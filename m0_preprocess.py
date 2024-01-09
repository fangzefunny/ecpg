import os 
import argparse 
import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from utils.fig_fn import * 

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set', '-d', help='which_data', type = str, default='exp1')
parser.add_argument('--method',   '-m', help='methods, mle or map', type = str, default='mle')
args = parser.parse_args()

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/analyses', f'{pth}/analyses/{args.data_set}',
        f'{pth}/analyses/{args.data_set}/human', f'{pth}/figures',
        f'{pth}/figures/{args.data_set}', f'{pth}/figures/{args.data_set}/human']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

# --------- Column and Value Dict ---------#

exp1_size = 4

columnName = { 
    'stimulus':  's', 
    'response':  'a',
    'screen_id': 'blockType',
    'act1':      'a_ava1',
    'act2':      'a_ava2',
    'acc':       'r',
    'corAct':    'a*',
}

def trial_Config(row):
    '''Get the trial config of each row'''
    s, a1, a2 = row['s'], row['a_ava1'], row['a_ava2']
    ls = [s]+list(np.sort([a1, a2]))
    return "".join([str(i) for i in ls])

def get_Group(row, stim_Lst, pool):
    if stim_Lst[row['s']] == 2:
        return 'control'
    elif row['config'] in pool:
        return 'trained'
    else: 
        if row['s'] == 4:
            return 'probe'
        else:
            return 'untrained'

# --------- Preprocess ---------#

def remake_cols_idx(data, seed=42):
    '''Remake the column name and values

    Args:
        data: raw data 
        seed: random seed
    '''
    # random generator
    rng = np.random.RandomState(seed)

    # remake the column
    data.rename(columns=columnName, inplace=True)

    # drop the pratice stage
    data = data.query('blockType!="practice"').reset_index(drop=True)
    garbage = ['start_time', 'payment', 'end_time', 'time_elapsed', 'blockType',
               'is_reward', 'n_Prac', 'raw_act1', 'raw_act2']
    for i in garbage: 
        if i in data.columns: data = data.drop(columns=[i])

    # turn trial into int
    data['trial'] = data['trial'].apply(lambda x: int(x))

    # change ass, gen into train and test
    data['stage'] = data['stage'].map({'ass': 'train', 'gen': 'test'})

    # turn stimuli and action into 0 - 3
    if args.data_set == 'exp1':
        data['s'] = data['s'].apply(lambda x: int(x) % exp1_size)
    data['a_ava1'] = data['a_ava1'].apply(lambda x: int(x) % exp1_size)
    data['a_ava2'] = data['a_ava2'].apply(lambda x: int(x) % exp1_size)

    # trial configuration
    data['config'] = data.apply(lambda x: trial_Config(x), axis=1)

    # the human action: str to index,
    # pick a random number if there is a nan
    data['a'] = data['a'].map(
        {np.nan: rng.choice(2), 'f': 0, 'j': 1})
    data['a'] = data.apply(lambda x: x[f"a_ava{x['a']+1}"], axis=1)
    data['r'] = data['r']

    # the correct action: str to index,
    # pick a random number if there is a nan
    data['a*'] = data['a*'].fillna(rng.choice(2))
    data['a*'] = data.apply(lambda x: x[f"a_ava{x['a*']+1}"], axis=1)

    # group type 
    pool = data.query('stage=="train"')['config'].unique()
    stim = [config[0] for config in pool]
    nS   = len(data['s'].unique())
    stim_Lst = [stim.count(str(i)) for i in range(nS)]
    data['group'] = data.apply(lambda x: get_Group(x, stim_Lst, pool), axis=1)

    # add block type
    if 'block_type' not in data.columns:
        data['block_type'] = 'cont'
    
    return data 

def splitTwoGroups(data, qs=[.25, .75]):
    assert len(qs)==2, 'qs should have a length of 2.'
    # find the well and bad group by median split 
    avg_train_acc = data.query('group=="untrained"').groupby(by=['sub_id']
                    ).mean(numeric_only=True).reset_index()
    split_bars = [avg_train_acc['r'].quantile(q) for q in qs]
    poor_group = avg_train_acc.query(f'r<={split_bars[0]}')['sub_id'].unique()
    norm_group = avg_train_acc.query(f'{split_bars[0]}<=r & r<={split_bars[1]}'
                                     )['sub_id'].unique()
    good_group = avg_train_acc.query(f'{split_bars[1]}<=r')['sub_id'].unique()
    def assign_group(x):
        if x in poor_group:
            return 'poor'
        elif x in norm_group:
            return 'norm'
        elif x in good_group:
            return 'good'
    data['goodPoor'] = data['sub_id'].apply(assign_group) 
    return data, good_group, norm_group, poor_group

def filter(data, threshold=.6):
    acc = data.query('stage=="train" & tps>=7')['r'].mean()

    return acc, acc>threshold

def data_stats(tot_acc, accepted_acc):
    ymax = 65
    rate = len(accepted_acc) / len(tot_acc)
    plt.figure(figsize=(5, 4))
    sns.histplot(tot_acc, bins=20)
    plt.vlines(x=.6, ymin=0, ymax=ymax, ls='--', color='k')
    plt.ylim([0, ymax])
    plt.title(f'Accept data {len(accepted_acc)}/{len(tot_acc)} \nAccepted rate: {rate:.2f}')
    plt.tight_layout()
    
def pre_process(data_set):

    # processed data
    for_save = {}
    processed_data = []
    tot_accs, accepted_accs = [], []
    
    ## Loop to preprocess each file
    # obtain all files under the exp1 list
    files = os.listdir(f'{pth}/data/{data_set}/')
    n_sub = 0
    for file in files:
        # skip the folder 
        n_sub += 1
        if not os.path.isdir(f'{pth}/data/{file}'): 
            # get the subid 
            sub_id = file.split('-')[0] 
            # get and remake the cols and index
            data = remake_cols_idx(
                pd.read_csv(f'{pth}/data/{data_set}/{file}'))  
            if 'block_id' not in data.columns:
                data.loc[:124, 'block_id'] = 0
                data.loc[124:, 'block_id'] = 1
            acc, accept = filter(data)
            tot_accs.append(acc)
            # assign blocks to subject
            if accept:
                accepted_accs.append(acc)
                data['sub_id'] = int(sub_id.split('_')[1])
                processed_data.append(data)
                
    process_data, _, _ ,_ = splitTwoGroups(
        pd.concat(processed_data, axis=0, sort=True), qs=[.25, .75])
    process_data.to_csv(f'{pth}/data/{data_set}-human.csv')

    sub_Lst = process_data['sub_id'].unique()
    for sub_id in sub_Lst:
        for_save[f'subj_{sub_id}'] = {}
        sub_data =process_data.query(f'sub_id == {sub_id}')
        for b in sub_data['block_id'].unique():
            block_data = sub_data.query(f'block_id=={b}').reset_index(drop=True)
            for_save[f'subj_{sub_id}'][b] = block_data
                    
    # save for fit 
    print(f'\tSubject Number: {len(for_save.keys())}/{n_sub}')
    with open(f'{pth}/data/{data_set}.pkl', 'wb')as handle:
        pickle.dump(for_save, handle)

    
    # -------- analyze the data  ------- #

    # base aggregation
    fname = f'{pth}/analyses/{args.data_set}/human/{args.method}-base.csv'
    agg(process_data, voi=['a']).to_csv(fname, index = False, header=True)

    # show acc distribution
    data_stats(tot_accs, accepted_accs)
    plt.savefig(f'{pth}/figures/{args.data_set}/dataStats-{args.data_set}.png', dpi=250)
    plt.close()

    #transfer effect
    viz_transfer(args.data_set, ['human'], method=args.method)
    plt.savefig(f'{pth}/figures/{args.data_set}/human/{args.method}-transfer.png', dpi=250)
    plt.close()

    # learning curve
    viz_lc(args.data_set, ['human'], method=args.method)
    plt.savefig(f'{pth}/figures/{args.data_set}/human/{args.method}-lc.png', dpi=250)
    plt.close()

if __name__ == '__main__':
    
    print(f'\nPreprocessing {args.data_set}...')
    pre_process(data_set=args.data_set)