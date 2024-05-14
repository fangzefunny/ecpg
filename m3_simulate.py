import os 
import pickle
import argparse 

import numpy as np 
import pandas as pd

from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--agent_name', '-n', help='choose agent', default='ecPG')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='map')
parser.add_argument('--cross_valid','-v', help='is corss validated', type = bool, default=False)
parser.add_argument('--algorithm',  '-a', help='fitting algorithm', type = str, default='BFGS')
parser.add_argument('--data_set',   '-d', help='choose data set', default='exp1-cross_train')
parser.add_argument('--n_sim',      '-f', help='f simulations', type=int, default=1)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=120)
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--params',     '-p', help='params', type=str, default='')
args = parser.parse_args()
args.agent = eval(args.agent_name)
args.env_fn = eval(f'{args.data_set.split("-")[0]}_task')

# find the current path
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/simulations', f'{pth}/simulations/{args.data_set}', 
        f'{pth}/simulations/{args.data_set}/{args.agent_name}']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

# ---- Simulate to compare with the human data ------#

def sim_paral(pool, args):
    
    ## Simulate data for n_sim times 
    seed = args.seed 
    sim_fn = simulate_cross_valid if args.cross_valid else simulate
    res = [pool.apply_async(sim_fn, args=(args, seed+5*i))
                            for i in range(args.n_sim)]
    for i, p in enumerate(res):
        sim_data = p.get() 
        fname  = f'{pth}/simulations/'
        fname += f'{args.data_set}/{args.agent_name}/sim-{args.method}-idx{i}.csv'
        sim_data.to_csv(fname, index=False)

# define functions
def simulate(args, seed):

    # define the subj
    model = wrapper(args.agent, env_fn=args.env_fn)
    # load task data
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)

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

def simulate_cross_valid(args, seed):

    # define the subj
    model = wrapper(args.agent, env_fn=args.env_fn)

    ## Loop to choose the best model for simulation
    # the last column is the loss, so we ignore that
    sim_data = []
    # load data for parameter 
    fname  = f'{pth}/fits/{args.data_set}/fit_sub_info'
    fname += f'-{args.agent_name}-{args.method}.pkl'
    with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
    # load data for task
    data_set = args.data_set.split('-')[0]
    fname = f'{pth}/data/{data_set}-cross_test.pkl'
    with open(fname, 'rb')as handle: task_data = pickle.load(handle)
    sub_lst = list(fit_sub_info.keys())
    if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
    # synthesize the data and save
    rng = np.random.RandomState(seed)
    for sub_idx in sub_lst: 
        # load parameter 
        params = fit_sub_info[sub_idx]['param']
        sim_sample = model.sim(task_data[sub_idx], params, rng=rng)
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
    print(f'Simulating {args.data_set} using {args.agent_name}')
    pool = get_pool(args)

    # STEP 2: SYNTHESIZE DATA
    sim_paral(pool, args)
    concat_sim_data(args)

    pool.close()
