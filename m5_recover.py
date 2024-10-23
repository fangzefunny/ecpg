import argparse 
import os 
import pickle
import datetime 
import numpy as np
import pandas as pd
import subprocess
from copy import deepcopy


from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',     '-d', help='choose data set', default='exp2')
parser.add_argument('--agent_name',   '-n', help='data-generting agent', default='rmPG_fea')
parser.add_argument('--method',       '-m', help='methods, mle or map', type = str, default='map')
parser.add_argument('--algorithm',    '-a', help='fitting algorithm', type = str, default='BFGS')
parser.add_argument('--other_agents', '-o', help='fitted agent', nargs='+', required=True)
parser.add_argument('--n_cores',      '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',         '-s', help='random seed', type=int, default=420)
args = parser.parse_args()
args.agent = eval(args.agent_name)
args.env_fn = eval(f'{args.data_set.split("-")[0]}_task')
args.group = 'group' if args.method=='hier' else 'ind'
pth = os.path.dirname(os.path.abspath(__file__))

# -------------------------------- #
#          MODEL RECOVERY          #
# -------------------------------- #

def model_recover(args, n_sub=40, n_samp=10):

    ## STEP 0: GET PARALLEL POOL
    pool = get_pool(args)

    ## STEP 1: SYTHESIZE FAKE DATA FOR PARAM RECOVER
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
    syn_data_model_recover_paral(pool, data, args, n_sub=n_sub, n_samp=n_samp)
    pool.close() 

    ## STEP 2: REFIT THE OTHER MODEL TO THE SYTHESIZE DATA 
    for agent_name in args.other_agents:
        cmand = ['python', 'm2_fit.py', f'-d={args.data_set}-{args.agent_name}',
                f'-n={agent_name}', '-s=420', '-f=30', '-c=30', 
                f'-m={args.method}', f'-a={args.algorithm}']
        subprocess.run(cmand)

def syn_data_model_recover_paral(pool, data, args, n_sub=30, n_samp=10):

    # get parameters 
    fname  = f'{pth}/fits/{args.data_set}/fit_sub_info'
    fname += f'-{args.agent_name}-{args.method}.pkl'      
    with open(fname, 'rb')as handle: fit_info_orig = pickle.load(handle)

    ## create a sub list of subject list 
    sub_lst_orig = list(fit_info_orig.keys())
    if 'group' in sub_lst_orig: sub_lst_orig.pop(sub_lst_orig.index('group'))
    # select subject for recovery 
    rng = np.random.RandomState(args.seed)
    # Use random choice without replacement to select subjects
    sub_lst = rng.choice(sub_lst_orig, size=n_sub, replace=False)
    fit_param = {k: fit_info_orig[k]['param'] for k in sub_lst}

    # create the synthesize data for the chosen sub
    res = [pool.apply_async(syn_data_model_recover, 
                    args=(data, fit_param[sub_id], sub_id, args.seed*i, n_samp))
                    for i, sub_id in enumerate(sub_lst)]

    syn_data = {}
    for _, p in enumerate(res):
        sim_data_all = p.get() 
        for sub_id in sim_data_all.keys():
            syn_data[sub_id] = sim_data_all[sub_id]

    # save for fit 
    with open(f'{pth}/data/{args.data_set}-{args.agent_name}.pkl', 'wb')as handle:
        pickle.dump(syn_data, handle)
    print(f'  {n_sub} Syn data for {args.agent_name} has been saved!')

def syn_data_model_recover(task_data, param, sub_id, seed, n_samp=10):

    # create random state 
    rng = np.random.RandomState(seed)
    model = wrapper(args.agent, args.env_fn)

    # synthesize the data and save
    
    task_lst = rng.choice(list(task_data.keys()), size=n_samp, replace=False)
    
    sim_data_all = {}
    for i, task_id in enumerate(task_lst):
        sample_id = f'{sub_id}-{i}'
        sim_data = {} 
        block_ind = task_data[task_id]
        for block_id in block_ind:
            task = task_data[task_id][block_id]
            sim_sample = model.sim({i: task}, param, rng=rng)
            sim_sample = sim_sample.drop(columns=model.agent.voi)
            sim_sample['sample_id'] = sample_id
            sim_data[block_id] = sim_sample  
        sim_data_all[sample_id] = deepcopy(sim_data)

    return sim_data_all

if __name__ == '__main__':

    model_recover(args)