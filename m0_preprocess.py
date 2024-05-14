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
parser.add_argument('--data_set', '-d', help='which_data', type = str, default='exp2')
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
    'corAct':    'cor_a',
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
    # add block type
    if 'block_type' not in data.columns: data['block_type'] = 'cont'

    # turn stimuli and action into 0 - 3
    task_id = args.data_set.split('-')[0]
    task_fn = eval(f'{task_id}_task')
    if task_id=='exp1': data['s'] = data['s'].apply(lambda x: int(x) % exp1_size)
    data['f'] = data.apply(lambda x: task_fn(x['block_type']).s2f(x['s']), axis=1)
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
    data['cor_a'] = data['cor_a'].fillna(rng.choice(2))
    data['cor_a'] = data.apply(lambda x: x[f"a_ava{x['cor_a']+1}"], axis=1)

    # group type 
    pool = data.query('stage=="train"')['config'].unique()
    stim = [config[0] for config in pool]
    nS   = len(data['s'].unique())
    stim_Lst = [stim.count(str(i)) for i in range(nS)]
    data['group'] = data.apply(lambda x: get_Group(x, stim_Lst, pool), axis=1)

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

def pretrain_exp1(thershold=.98):
    '''Cached to speed up model fitting 
    '''
    env = exp1_task('cont')
    nS = nZ = env.nS
    F = torch.FloatTensor(np.eye(nS))
    theta0 = torch.nn.Parameter(3.5*torch.ones(1))
    i, loss_prev, tol = 0, np.inf, 1e-8
    optim = torch.optim.Adam([theta0], lr=0.1)

    while True:
        ZHat = torch.softmax(F*theta0, dim=1)
        acc = (ZHat*torch.eye(nZ)).sum(1).mean()
        loss = .5*(acc - thershold).square()
        optim.zero_grad()
        loss.backward() 
        optim.step() 
        converge = (.5*np.abs(loss.detach().numpy() - loss_prev) < tol) or (i > 800)
        if converge: break
        loss_prev = loss.data.numpy()
        i += 1

    theta = np.eye(nS)*theta0.detach().numpy()
    fname = f'{pth}/data/exp1_ecpg_weight.pkl'
    with open(fname, 'wb')as handle: pickle.dump(theta, handle)

def pretrain_fea(block_type, threshold=.99):
    '''Cached to speed up model fitting 
    '''
    env = exp2_task(block_type)
    nS, nZ, nI = env.nS, env.nS, env.nI
    theta0 = (torch.ones(1)).requires_grad_()
    i, loss_prev, tol = 0, np.inf, 1e-10
    optim = torch.optim.Adam([theta0], lr=0.1)

    # forward
    while True:
        F = torch.FloatTensor(np.vstack([env.embed(env.s2f(s)) for s in range(nS)]))
        R = F * theta0
        Z = torch.softmax(torch.vstack([(R[r] * R).sum(1) for r in range(nZ)]), axis=1)
        acc = (Z*torch.eye(nZ)).sum(1).mean()
        loss = .5*(acc - threshold).square()
        optim.zero_grad()
        loss.backward() 
        optim.step()
        converge = (.5*np.abs(loss.data.numpy() - loss_prev) < tol) or (i > 800)
        if converge: break
        # cache 
        loss_prev = loss.data.numpy()
        i += 1

    # decide the encoder theta 
    F = np.vstack([env.embed(env.s2f(s)) for s in range(nS)])
    R = F * theta0.data.numpy()[0]
    tar = softmax(np.vstack([(R[r] * R).sum(1) for r in range(nZ)]), axis=1)

    # train a theta that do this classification 
    theta = torch.zeros([nI, nZ]).requires_grad_()
    optim = torch.optim.SGD([theta], lr=0.25)
    i, loss_prev = 0, np.inf
    tar = torch.FloatTensor(tar)

    while True:
        # forward
        ZHat = torch.softmax(torch.FloatTensor(F)@theta, dim=1)
        loss = (-tar*torch.log(ZHat+eps_)).sum(1).mean(0)
        # check convergence 
        converge = (.5*np.abs(loss.data.numpy() - loss_prev) < tol) or (i > 800)
        if converge: break
        # backward 
        optim.zero_grad()
        loss.backward()
        optim.step()
        # cache 
        loss_prev = loss.data.numpy()
        i += 1

    return theta.detach().numpy()  

def pretrain_exp2(threshold=.97):
    theta_dict = {}
    for block_type in ['cons', 'cont', 'conf']:
        theta_dict[block_type] = pretrain_fea(block_type, threshold)
    fname = f'{pth}/data/exp2_ecpg_weight.pkl'
    with open(fname, 'wb')as handle: pickle.dump(theta_dict, handle)

if __name__ == '__main__':
    
    print(f'\nPreprocessing {args.data_set}...')
    pre_process(data_set=args.data_set)
    if args.data_set.split('-')[0]=='exp1': pretrain_exp1()
    if args.data_set.split('-')[0]=='exp2': pretrain_exp2()
    