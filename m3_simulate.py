import os 
import pickle
import argparse 

import numpy as np 
import pandas as pd

from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import AEtask

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--agent_name', '-n', help='choose agent', default='rdPG_fea')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='mle')
parser.add_argument('--algorithm',  '-a', help='fitting algorithm', type = str, default='BFGS')
parser.add_argument('--data_set',   '-d', help='choose data set', default='exp2')
parser.add_argument('--n_sim',      '-f', help='f simulations', type=int, default=1)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=120)
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--params',     '-p', help='params', type=str, default='')
args = parser.parse_args()
args.agent = eval(args.agent_name)

# find the current path
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/simulations', f'{pth}/simulations/{args.data_set}', 
        f'{pth}/simulations/{args.data_set}/{args.agent_name}']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

# ---- Simulate to compare with the human data ------#

def sim_paral(pool, data, args):
    
    ## Simulate data for n_sim times 
    seed = args.seed 
    res = [pool.apply_async(simulate, args=(data, args, seed+5*i))
                            for i in range(args.n_sim)]
    for i, p in enumerate(res):
        sim_data = p.get() 
        fname  = f'{pth}/simulations/'
        fname += f'{args.data_set}/{args.agent_name}/sim-{args.method}-idx{i}.csv'
        sim_data.to_csv(fname, index=False)

# define functions
def simulate(data, args, seed):

    # define the subj
    model = wrapper(args.agent, env_fn=AEtask)

     # if there is input params 
    if args.params != '': 
        in_params = [float(i) for i in args.params.split(',')]
    else: in_params = None 

    ## Loop to choose the best model for simulation
    # the last column is the loss, so we ignore that
    sim_data = []
    fname  = f'{pth}/fits/{args.data_set}/fit_sub_info'
    fname += f'-{args.agent_name}-{args.method}.pkl'
    with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
    sub_lst = list(fit_sub_info.keys())
    if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
    for sub_idx in sub_lst: 
        if in_params is None:
            params = fit_sub_info[sub_idx]['param']
        else:
            params = in_params

        # synthesize the data and save
        rng = np.random.RandomState(seed)
        sim_sample = model.sim(data[sub_idx], params, rng=rng)
        sim_data.append(sim_sample)
        seed += 1

    return pd.concat(sim_data, axis=0, ignore_index=True)

def concat_sim_data(args):
    
    sim_data = [] 
    for i in range(args.n_sim):
        fname  = f'{pth}/simulations/{args.data_set}/'
        fname += f'{args.agent_name}/sim-{args.method}-idx{i}.csv'
        sim_datum = pd.read_csv(fname)
        sim_data.append(sim_datum)
        os.remove(fname) # delete the sample files

    sim_data = pd.concat(sim_data, axis=0, ignore_index=True)
    fname  = f'{pth}/simulations/{args.data_set}/'
    fname += f'{args.agent_name}/sim-{args.method}.csv'
    sim_data.to_csv(fname)
        
if __name__ == '__main__':
    
    ## STEP 0: GET PARALLEL POOL
    print(f'Simulating {args.agent_name}')
    pool = get_pool(args)

    # STEP 1: LOAD DATA 
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)

    # STEP 2: SYNTHESIZE DATA
    sim_paral(pool, data, args)
    concat_sim_data(args)

    pool.close()
