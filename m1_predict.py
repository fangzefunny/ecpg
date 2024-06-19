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
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='exp2')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='mle')
parser.add_argument('--agent_name', '-n', help='choose agent', default='ecPG_fea_sim')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=2420)
parser.add_argument('--params',     '-p', help='params', type=str, default='[30, 8, .1]')
parser.add_argument('--sub_id',    '-i', help='sub id', type=str, default='nan')
args = parser.parse_args()
args.agent = eval(args.agent_name)
args.env_fn = eval(f'{args.data_set.split("-")[0]}_task')
args.group = 'group' if args.method=='hier' else 'ind'

# find the current path
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/simulations', f'{pth}/simulations/{args.data_set}', 
        f'{pth}/simulations/{args.data_set}/{args.agent_name}']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

# ---- Simulate ------#
    
def sim_grid(args, n_sample_per_param=20):
    rng = np.random.RandomState(args.seed)
    poi_raw = args.agent.poi_raw
    n_sample = n_sample_per_param*args.agent.n_params 
    # create a summary table for indexing
    p_vals_lst = np.vstack([[np.round(poi[0]+rng.rand()*poi[1], 3) 
                  for poi in poi_raw] for _ in range(n_sample)])
    p_vals_table = pd.DataFrame(p_vals_lst, columns=args.agent.p_names)
    fname  = f'{pth}/simulations/{args.data_set}/'
    fname += f'{args.agent_name}/sim_configs.csv'
    p_vals_table.to_csv(fname)
    for i, p_vals in enumerate(p_vals_lst):
        p_vals = p_vals_lst[i, :]
        kwargs = {n: v for n, v in zip(args.agent.p_names, p_vals)}
        print(f'\nAt {i+1}/{n_sample}')
        block_types = ['cont'] if args.data_set=='exp1' \
                            else ['cons', 'cont', 'conf']
        for block_type in block_types:
            sim_subj_paral(pool, args, block_type, 
                           n_sim=200, **kwargs)
            
def sim_param(args):
    if args.sub_id=='nan': 
        kwargs = {n: v for n, v in zip(args.agent.p_names, eval(args.params))}
    else:
        kwargs = {}
    block_types = ['cont'] if args.data_set=='exp1' \
                            else ['cons', 'cont', 'conf']
    for block_type in block_types:
        sim_subj_paral(pool, args, block_type, 
                        n_sim=200, **kwargs)

# ---- Simulate to see insights ------#

def sim_subj_paral(pool, args, block_type, 
                    n_sim=50, seed=2022, **kwargs):
    
    # warp up a model 
    model = wrapper(args.agent, args.env_fn)
    trials = np.arange(eval(f'args.env_fn.n_{block_type}_train'))
    tars = args.agent.insights   
    voi = {tar: [0]*len(trials) for tar in tars}

    # prepare parameters: if subject parameters are passed
    # use subject's parameters
    if args.sub_id == 'nan':
        rparams  = [kwargs[n] for n in model.agent.p_names]
        param_info = '-'.join([f'{n}={p}' for n, p in zip(model.agent.p_names, rparams)])
        # conver to real parameter space 
        params = [fn(p) for p, fn in zip(rparams, model.agent.p_links)] 
        task = None 
    else:
        fname = f'{pth}/fits/{args.data_set}/fit_sub_info-{args.agent_name}'
        fname += f'-{args.method}.pkl'
        with open(fname, 'rb')as handle: fit_info = pickle.load(handle)
        params = fit_info[args.sub_id]['param'] 
        fname = f'{pth}/data/{args.data_set}.pkl'
        with open(fname, 'rb')as handle: datum = pickle.load(handle)
        tasks = datum[args.sub_id]
        for _, item in tasks.items():
            if item['block_type'].unique() == block_type:
                task = item 
        param_info = args.sub_id

    sim_data = []
    print(f'\tStimulating {block_type}, {param_info}')
    
    res = [pool.apply_async(sim_subj, 
                    args=(model, params, block_type, trials, seed, i, task))
                    for i in range(n_sim)]
    
    for _, p in enumerate(res):
        sim_sample, i_voi = p.get() 
        sim_data.append(sim_sample)  
        for tar in tars:
            for i, t in enumerate(trials):
                voi[tar][i] += (i_voi[tar][t] / n_sim) 

    # save
    sim_data = pd.concat(sim_data, ignore_index=True)
    fname  = f'{pth}/simulations/{args.data_set}/'
    fname += f'{args.agent_name}/simsubj-{block_type}_data-{param_info}.csv'
    sim_data.to_csv(fname, index=False, header=True)
    fname  = f'{pth}/simulations/{args.data_set}/'
    fname += f'{args.agent_name}/simsubj-{block_type}_voi-{param_info}.pkl'
    with open(fname, 'wb')as handle: pickle.dump(voi, handle)

def sim_subj(model, params, block_type, trials, seed, i, task=None):

    # decide what to collect
    tars = model.agent.insights
    rng = np.random.RandomState(seed+i)
    voi = {tar: [0]*len(trials) for tar in tars}

    # simulate block n times
    if task is None:
        task = args.env_fn(block_type).instan(seed=seed+3*i)
        model.register_hooks(*tars)
        sim_sample = model.sim_block(task, params, rng=rng)
        sim_sample['sub_id'] = f'sim{i}'
    else:
        block_data = task.copy()
        for v in model.env_fn.voi:
            if v in block_data.columns:
                block_data = block_data.drop(columns=v)
        model.register_hooks(*tars)
        sim_sample = model.sim_block(block_data, params, rng=rng)
    # inspect into the agent 
    for tar in tars:
        for i, t in enumerate(trials):
            voi[tar][i] = model.insights[tar][t]    
    
    return sim_sample, voi 
        
if __name__ == '__main__':
    
    ## STEP 0: GET PARALLEL POOL
    print(f'Simulating {args.agent_name}')
    pool = get_pool(args)

    # STEP 1: SIMULATE DATA FOR ANALYSIS
    if len(args.params):
        sim_param(args)
    else:
        sim_grid(args)
        
    # STEP 2: CLOSE POOL
    pool.close()