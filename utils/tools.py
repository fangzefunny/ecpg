
import pickle
import numpy as np 
import pandas as pd 
import pingouin as pg
from utils.model import *

def get_sim_data(data_set, model, method='mle'):
    if model=='human':
        fname = f'../data/{data_set}-human.csv'
    else:
        fname = f'../simulations/{data_set}/{model}/sim-{method}.csv'
    return pd.read_csv(fname)

def get_fit_param(data_set, model, method='mle', poi=None):
    fname = f'../fits/{data_set}/fit_sub_info-{model}-{method}.pkl'
    with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
    if poi is None: poi = eval(model).p_names
    sub_lst = list(fit_sub_info.keys())
    if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
    params = {p: [] for p in poi}
    params['sub_id'] = []
    for sub_id in sub_lst:
        params['sub_id'].append(sub_id)
        for p, fn in zip(poi, eval(model).p_trans):
            idx = fit_sub_info[sub_id]['param_name'].index(p) 
            pvalue = fit_sub_info[sub_id]['param'][idx].copy()
            params[p].append(fn(pvalue))
    return pd.DataFrame.from_dict(params)

def get_llh_score(data_set, models, method, 
                  if_bms=False, 
                  use_bic=False,
                  relative=True):
    '''Get likelihood socres

    Inputs:
        models: a list of models for evaluation
    
    Outputs:
        crs: nll, aic and bic score per model per particiant
        pxp: pxp score per model per particiant
    '''
    tar = models[0] 
    fit_sub_info = []
    for i, m in enumerate(models):
        with open(f'{pth}/../fits/{data_set}/fit_sub_info-{m}-{method}.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        log_post = [fit_info[idx]['log_post'] for idx in subj_lst]
        bic      = [fit_info[idx]['bic'] for idx in subj_lst]
        h        = [fit_info[idx]['H'] for idx in subj_lst] if if_bms else 0
        n_param  = fit_info[list(subj_lst)[0]]['n_param']
        fit_sub_info.append({
            'log_post': log_post, 
            'bic': bic, 
            'n_param': n_param, 
            'H': h,
        })
    # get bms 
    if if_bms: bms_results = fit_bms(fit_sub_info, use_bic=use_bic)

    ## combine into a dataframe 
    cols = ['NLL', 'AIC', 'BIC', 'model', 'sub_id']
    crs = {k: [] for k in cols}
    for m in models:
        with open(f'{pth}/../fits/{data_set}/fit_sub_info-{m}-{method}.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        nll = [-fit_info[idx]['log_like'] for idx in subj_lst]
        aic = [fit_info[idx]['aic'] for idx in subj_lst]
        bic = [fit_info[idx]['bic'] for idx in subj_lst]
        crs['NLL'] += nll
        crs['AIC'] += aic
        crs['BIC'] += bic
        crs['model'] += [m]*len(nll)
        crs['sub_id'] += list(subj_lst)
    crs = pd.DataFrame.from_dict(crs)
    for c in ['NLL', 'BIC', 'AIC']:
        tar_crs = len(models)*list(crs.query(f'model=="{tar}"')[c].values)
        subtrack = tar_crs if relative else 0
        crs[c] -= subtrack
    if if_bms: 
        pxp = pd.DataFrame.from_dict({'pxp': bms_results['pxp'], 'model': models})
    else:
        pxp = 0 
    return crs, pxp 