import argparse 
import os 
import pickle
import datetime 
import numpy as np
import pandas as pd

from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--n_fit',      '-f', help='fit times', type = int, default=1)
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='exp1')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='map')
parser.add_argument('--algorithm',  '-a', help='fitting algorithm', type = str, default='BFGS')
parser.add_argument('--agent_name', '-n', help='choose agent', default='LC')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=420)
args = parser.parse_args()
args.agent  = eval(args.agent_name)
args.env_fn = eval(f'{args.data_set.split("-")[0]}_task')
args.group  = 'group' if args.method=='hier' else 'ind'

# find the current path, create the folders if not existedl
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/fits', f'{pth}/fits/{args.data_set}']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

def fit(pool, data, args):
    '''Find the optimal free parameter for each model 
    '''
    ## declare environment 
    model = wrapper(args.agent, args.env_fn)

    ## fit list
    fname = f'{pth}/fits/{args.data_set}/fit_sub_info-{args.agent_name}-{args.method}.pkl'
    if os.path.exists(fname):
        # load the previous fit resutls
        with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
        fitted_sub_lst = [k for k in fit_sub_info.keys()]
    else:
        fitted_sub_lst = []
        fit_sub_info = {}

    ## Start 
    start_time = datetime.datetime.now()
    sub_start  = start_time

    ## Fit params to each individual 
    if args.group == 'ind':
        done_subj = len(fitted_sub_lst)
        all_subj  = len(data.keys())
        for sub_idx in data.keys(): 
            if sub_idx not in fitted_sub_lst:  
                print(f'Fitting {args.agent_name} subj {sub_idx}, progress: {(done_subj*100)/all_subj:.2f}%')
                fit_info = model.fit(data[sub_idx], args.method, args.algorithm, 
                                     pool=pool, seed=args.seed, n_fits=args.n_fit,
                                     verbose=False, init=False)
                fit_sub_info[sub_idx] = fit_info
                with open(fname, 'wb')as handle: 
                    pickle.dump(fit_sub_info, handle)
                sub_end = datetime.datetime.now()
                print(f'\tLOSS:{-fit_info["log_post"]:.4f}, using {(sub_end - sub_start).total_seconds():.2f} seconds')
                sub_start = sub_end
                done_subj += 1
    elif args.group == 'group':
        fit_sub_info = fit_hier(pool, data, model, fname,  
                                seed=args.seed, n_fits=args.n_fit)
        with open(fname, 'wb')as handle: 
            pickle.dump(fit_sub_info, handle)
                
    ## END!!!
    end_time = datetime.datetime.now()
    print('\nparallel computing spend {:.2f} seconds'.format(
            (end_time - start_time).total_seconds()))
    

def summary(data, args):

    ## Prepare storage
    n_sub    = len(data.keys())
    n_params = args.agent.n_params
    field    = ['log_post', 'log_like', 'aic', 'bic']
    res_mat  = np.zeros([n_sub, n_params+len(field)]) + np.nan 
    res_smry = np.zeros([2, n_params+len(field)]) + np.nan 
    fname =  f'{pth}/fits/{args.data_set}/'
    fname += f'fit_sub_info-{args.agent_name}-{args.method}.pkl'

    ## Loop to collect dat 
    with open(fname, 'rb')as handle: 
        fit_sub_info = pickle.load(handle)
    sub_lst = list(fit_sub_info.keys())
    if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
    for i, sub_idx in enumerate(sub_lst):
        sub_info = fit_sub_info[sub_idx]
        res_mat[i, :] = np.hstack([sub_info[p] 
                        for p in ['param']+field])
        if i==0: col = args.agent.p_names + field
    
    ## Compute and save the mean and sem
    res_smry[0, :] = np.mean(res_mat, axis=0)
    res_smry[1, :] = np.std(res_mat, axis=0) / np.sqrt(n_sub)
    fname =  f'{pth}/fits/{args.data_set}/'
    fname += f'/avg_params-{args.agent_name}-{args.method}.csv'
    pd.DataFrame(res_smry, columns=col).round(4).to_csv(fname)

if __name__ == '__main__':

    ## STEP 0: GET PARALLEL POOL
    pool = get_pool(args)

    ## STEP 1: LOAD DATA 
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
  
    ## STEP 2: FIT
    fit(pool, data, args)
    # summary the mean and std for parameters 
    summary(data, args)
    pool.close()