import os 
import pickle
import argparse 
import pandas as pd 

from utils.parallel import get_pool 
from utils.model import * 
from utils.fig_fn import * 
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',   '-d', help='choose data set', default='exp1')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='map')
parser.add_argument('--agent_name', '-n', help='choose agent', default='rdPG')
parser.add_argument('--params',     '-p', help='params', type=str, default='')
parser.add_argument('--n_sim',      '-f', help='f simulations', type=int, default=1)
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
args = parser.parse_args()
args.agent = eval(args.agent_name)
args.env_fn = eval(f'{args.data_set.split("-")[0]}_task')

# define path
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/analyses/{args.data_set}/{args.agent_name}', 
        f'{pth}/figures/{args.data_set}/{args.agent_name}']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

def evaluate(args):

    # load data
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)

    # define the subj
    model = wrapper(args.agent, env_fn=args.env_fn)

    ## Loop to choose the best model for simulation
    # the last column is the loss, so we ignore that
    fname  = f'{pth}/fits/{args.data_set}/'
    fname += f'fit_sub_info-{args.agent_name}-{args.method}.pkl'
    with open(fname, 'rb')as handle: 
        fit_sub_info = pickle.load(handle)
    sim_data = []
    for sub_idx in data.keys(): 
        params = fit_sub_info[sub_idx]['param']
        sim_sample = model.eval(data[sub_idx], params)
        sim_data.append(sim_sample)
    
    fname  = f'{pth}/analyses/{args.data_set}/'
    fname += f'{args.agent_name}/{args.method}-eval.csv'
    pd.concat(sim_data, axis=0).to_csv(fname, index = False, header=True)

def aggregate(args):
    fname  = f'{pth}/simulations/{args.data_set}/'
    fname += f'{args.agent_name}/sim-{args.method}.csv'
    agg_data = agg(pd.read_csv(fname), model=args.agent_name, voi=['a'])
    fname  = f'{pth}/analyses/{args.data_set}/'
    fname += f'{args.agent_name}/{args.method}-base.csv'
    agg_data.to_csv(fname, index = False, header=True)

def viz_effects(args):

     # transfer effect
    viz_transfer(args.data_set, ['human', args.agent_name], method=args.method)
    fname  = f'{pth}/figures/{args.data_set}/'
    fname += f'{args.agent_name}/{args.method}-transfer.png'
    plt.savefig(fname, dpi=250)
    plt.close()

    # learning curve
    viz_lc(args.data_set, [args.agent_name], method=args.method)
    fname  = f'{pth}/figures/{args.data_set}/'
    fname += f'{args.agent_name}/{args.method}-lc.png'
    plt.savefig(fname, dpi=250)
    plt.close()


if __name__ == '__main__':

    print(f'Analyzing {args.agent_name}')

    # STEP 1: EVALUATE THE DATA 
    evaluate(args)

    # # STEP 2: AGGREGATE THE DATA
    # aggregate(args)

    # # STEP 3: VIZUALIZE QUALITATIVE PROPERTY
    # viz_effects(args)


   