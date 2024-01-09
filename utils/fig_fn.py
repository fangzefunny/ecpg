import os 
import pickle 

import numpy as np
import pandas as pd 
import pingouin as pg

import matplotlib.pyplot as plt 
import seaborn as sns 

from utils.model import *
from utils.viz import viz
from utils.env_fn import AEtask
#viz.get_style()

# define path
pth = os.path.dirname(os.path.abspath(__file__))

# -------------- Axuiliary ---------------- #

def JSD(p, q):
    m = .5*(p+q)
    d1 = (p*(np.log(p+eps_)-np.log(m+eps_))).sum()
    d2 = (q*(np.log(q+eps_)-np.log(m+eps_))).sum()
    return np.sqrt(.5*d1+.5*d2)

def assoc(p, q, n=4):
    a = np.eye(n)[0]
    b = np.eye(n)[1]
    s = np.exp(-JSD(p, q)) 
    base = np.exp(-JSD(a, b))
    return (s-base)/(1-base)

def agg(data, model='human', sim_id=0, voi=[]):

    coi = ['r'] if model=='human' else ['acc']
    cols =['sub_id', 'tps', 'stage', 'group', 'goodPoor', 'block_type'] + voi
    sel_data = data[cols + coi]
    sel_data = sel_data.groupby(by=cols).mean().reset_index()
    sel_data['sim_id'] = sim_id
    sel_data.rename(columns={'r': 'acc'}, inplace=True)

    return sel_data

def transfer(ax, data, agents):
    sel_data = data.query(f'stage=="test"').reset_index()
    sel_data['is_untrained'] = sel_data['group'].map({
                                        'control': 0,
                                        'trained': 0,
                                        'untrained': 1,
                                        })
    sel_data = sel_data.groupby(by=['sub_id', 'is_untrained', 'agent']
                                        ).mean(numeric_only=True).reset_index()
    sel_data['agent'] = pd.Categorical(sel_data['agent'], categories=agents)
    palette = [eval(a).color for a in agents]

    v = sns.violinplot(x='is_untrained', y='acc', data=sel_data, 
                palette=palette, hue='agent', hue_order=agents, 
                legend=False, alpha=.1, inner=None, scale='width',
                ax=ax)
    plt.setp(v.collections, alpha=.35, edgecolor='none')
    s = sns.stripplot(x='is_untrained', y='acc', data=sel_data, 
                palette=palette, hue='agent', hue_order=agents, 
                edgecolor='gray', dodge=True, jitter=True, alpha=.7,
                legend=False, zorder=2,
                ax=ax)
    b = sns.barplot(x='is_untrained', y='acc', data=sel_data, 
                hue='agent', hue_order=agents, 
                errorbar='se', linewidth=1, 
                edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                capsize=.05, errwidth=2.5, errcolor=[.2, .2, .2],
                ax=ax)
    ax.legend().remove()
    ax.axhline(y=.5, xmin=0, xmax=1, lw=1, 
            color=[.2, .2, .2], ls='--')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Trained\n(learning)', 'Untrained\n(generalization)'])
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')

def lc(ax, exp, agents, method='mle'):

    data = []
    for agent in agents:
        df = pd.read_csv(f'{pth}/../analyses/{exp}/{agent}/{method}-base.csv')
        df['agent'] = agent
        data.append(df)
    data = pd.concat(data, axis=0)
    data['agent'] = data['agent'].apply(lambda x: eval(x).name)
    pal = [eval(a).color for a in agents]
    agents_rename = [eval(a).name for a in agents]    
    # get only the training data 
    q = f'group in ["control", "trained"] & agent in {agents_rename}'
    sel_data = data.query(q)
    # average over different trials
    sel_data = sel_data.groupby(by=['sub_id', 'tps', 'stage', 'agent']
                        )['acc'].mean().reset_index()
    sel_data['tps'] = sel_data.apply(
                lambda x: x['tps']+10 if x['stage']=='test' else x['tps'], axis=1)
    sns.lineplot(x='tps', y='acc', data=sel_data, lw=2,
                    err_style='bars', err_kws={'capsize': 3, 'elinewidth':2},
                    hue='agent', hue_order=agents_rename,
                    palette=pal,
                    errorbar='se', ax=ax)
    ax.legend().remove()
    ax.axvline(x=10, ymin=0, ymax=1, color='k', ls='--', lw=1.5)
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels(['1', '6', '11', '16'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylim([.4, 1])

def iSZ_reduct(exp, agents, method='mle'):
    data = []
    for agent in agents:

        # get complexity reduction 
        datum = pd.read_csv(f'{pth}/../simulations/{exp}/{agent}/sim-{method}.csv')
        red_datum = datum.query('trial in [0, 59]').reset_index(drop=True)
        if agent == 'rmPG': red_datum['i_SZ'] = 1
        sel_datum = red_datum.groupby(by=['sub_id', 'trial'])[
                            ['i_SZ']].mean().reset_index()
        d0  = sel_datum.query('trial==0').drop(columns='trial'
                            ).reset_index(drop=True)
        d59 = sel_datum.query('trial==59').drop(columns='trial'
                            ).reset_index(drop=True)
        
        iSZ_reduct = d59['i_SZ'] - d0['i_SZ']
        cat_datum = d0.merge(d59, on='sub_id')
        cat_datum['iSZ_reduct'] = iSZ_reduct
        cat_datum['agent'] = agent
        data.append(cat_datum[['sub_id', 'iSZ_reduct', 'agent']])

    comb_data = pd.concat(data, axis=0)
    comb_data['model'] = 'model'
    return comb_data

def cmp_rdPG_rdPG0(exp, cond, goodpoor=None):
    models = ['rdPG', 'rdPG0']
    models = models if exp=='exp1' else [m+'_fea' for m in models] 
    diffs = {}
    def class_err(theta0):
        nS = 4
        if exp=='exp1':
            psi = softmax(np.eye(nS)*theta0, axis=1)
            return (np.diagonal(psi)).mean()
        else:
            fn = AEtask(cond).embed
            F = np.vstack([fn(s) for s in range(nS)])
            R = F * theta0
            psi = softmax(np.vstack([(R[r] * R).sum(1) for r in range(nS)]), axis=1)
            return (np.diagonal(psi)).mean()

    for m in models:
        datum = pd.read_csv(f'analyses/{exp}/{m}/{exp}-eval.csv')
        datum = datum.query(f'block_type=="{cond}"')
        if goodpoor is not None: datum = datum.query(f'goodPoor=="{goodpoor}"')
        i_init = datum.query('stage=="train" & trial==0').groupby(by='sub_id')['i_SZ'].mean()
        i_end  = datum.query('stage=="train" & trial==59').groupby(by='sub_id')['i_SZ'].mean()
        diff = (i_end - i_init).reset_index().rename(columns={'i_SZ': m})
        # calculate the initial classification accuracy        
        datum = datum.groupby(by=['sub_id'])[['humanRew', 'theta0']
                ].mean().reset_index().rename(columns={'nll': m})
        err = datum['theta0'].apply(class_err)
        # calculate the generalization accuracy    
        datum = pd.read_csv(f'analyses/{exp}/{m}/{exp}-base.csv')
        datum = datum.query(f'block_type=="{cond}"')
        if goodpoor is not None: datum = datum.query(f'goodPoor=="{goodpoor}"')
        acc = datum.query('group=="untrained"').groupby(by='sub_id')['acc'].mean()
        diff[f'{m}_gen']  = acc.values
        diff[f'{m}_acc0'] = err
        diffs[m] = diff 
    data = diffs[models[0]].merge(diffs[models[1]], on='sub_id')
    return data

# ------ Quantitative Table -------- #

def get_table1(exp, agents, method='mle'):

    agents = agents if exp=='exp1' else [m+'_fea' for m in agents] 
    crs = {}
    for m in agents:
        n_param = eval(m).n_params
        fname = f'{pth}/../analyses/{exp}/{m}/{method}-eval.csv'
        data  = pd.read_csv(fname)
        data['nll'] = -data['ll']

        group_data = data
        subj_Lst = group_data['sub_id'].unique()
        nlls, aics, bics = [], [], [] 
        ass_nll, ass_aic, ass_bic = [], [], []
        gen_nll, gen_aic, gen_bic = [], [], []
        for sub_id in subj_Lst:
            inll = group_data.query(f'sub_id=={sub_id}')['nll'].sum() 
            inum = group_data.query(f'sub_id=={sub_id}')['nll'].shape[0]
            nlls.append(inll)
            aics.append(2*inll + 2*n_param)
            bics.append(2*inll + np.log(inum)*n_param)

            anll = group_data.query(f'sub_id=={sub_id} & stage=="train"'
                                )['nll'].sum() 
            anum = group_data.query(f'sub_id=={sub_id} & stage=="train"'
                                )['nll'].shape[0]
            ass_nll.append(anll)
            ass_aic.append(2*anll + 2*n_param)
            ass_bic.append(2*anll + np.log(anum)*n_param)

            gnll = group_data.query(f'sub_id=={sub_id} & stage=="test"'
                                )['nll'].sum() 
            gnum = group_data.query(f'sub_id=={sub_id} & stage=="train"'
                                )['nll'].shape[0]
            gen_nll.append(gnll)
            gen_aic.append(2*gnll + 2*n_param)
            gen_bic.append(2*gnll + np.log(gnum)*n_param)
        
        mat = {'tot_nll': nlls, 'tot_aic': aics, 'tot_bic': bics, 
               'ass_nll': ass_nll, 'ass_aic': ass_aic, 'ass_bic': ass_bic,
               'gen_nll': gen_nll, 'gen_aic': gen_aic, 'gen_bic': gen_bic}
        
        for k, v in mat.items(): mat[k] = np.mean(v)
        crs[m] = mat
        
        print(f'''
             # ------------------------------- {m} --------------------------------- #
                 Overall nll: {np.mean(nlls):.3f}      aic: {np.mean(aics):.3f}      bic: {np.mean(bics):.3f}
                 Train nll: {np.mean(ass_nll):.3f}         aic: {np.mean(ass_aic):.3f}      bic: {np.mean(ass_bic):.3f}            
                 Test nll: {np.mean(gen_nll):.3f}          aic: {np.mean(gen_aic):.3f}       bic: {np.mean(gen_bic):.3f} 
         ''')
    
    return pd.DataFrame.from_dict(crs).round(3)
        
def get_table2(exp, agents, method='mle'):
    crs = {}
    for m in agents:
        n_param = eval(m).n_params
        fname = f'{pth}/../analyses/{exp}/{m}/{method}-eval.csv'
        data  = pd.read_csv(fname)
        data['nll'] = -data['ll']
        
        group_data = data
        subj_Lst = group_data['sub_id'].unique()
        block_types = ['cons', 'cont', 'conf']
        nlls, aics, bics = [], [], []
        cons_nll, cons_aic, cons_bic  = [], [], []
        cont_nll, cont_aic, cont_bic = [], [], []
        conf_nll, conf_aic, conf_bic = [], [], []
        for sub_id in subj_Lst:
            inll = group_data.query(f'sub_id=={sub_id}')['nll'].sum() 
            inum = group_data.query(f'sub_id=={sub_id}')['nll'].shape[0]
            nlls.append(inll)
            aics.append(2*inll + 2*n_param)
            bics.append(2*inll + np.log(inum)*n_param)
                
            for b_type in block_types:

                inll = group_data.query(f'sub_id=={sub_id} & block_type=="{b_type}"'
                                    )['nll'].sum() 
                inum = group_data.query(f'sub_id=={sub_id} & block_type=="{b_type}"'
                                )['nll'].shape[0]
                eval(f'{b_type}_nll').append(inll)
                eval(f'{b_type}_aic').append(2*inll + 2*n_param)
                eval(f'{b_type}_bic').append(2*inll + np.log(inum)*n_param)
                
        
        mat = {'tot_nll': nlls, 'tot_aic': aics, 'tot_bic': bics,
               'cons_nll': cons_nll, 'cons_aic': cons_aic, 'cons_bic': cons_bic, 
               'cont_nll': cont_nll, 'cont_aic': cont_aic, 'cont_bic': cont_bic,
               'conf_nll': conf_nll, 'conf_aic': conf_aic, 'conf_bic': conf_bic,}
        
        for k, v in mat.items(): mat[k] = np.mean(v)
        crs[m] = mat

        print(f'''
             # --------------------- {m} --------------------- #
                 Tot nll: {np.mean(nlls):.3f}      aic: {np.mean(aics):.3f}      bic: {np.mean(bics):.3f}
                 cons nll: {np.mean(cons_nll):.3f}      aic: {np.mean(cons_aic):.3f}      bic: {np.mean(cons_bic):.3f}            
                 cont nll: {np.mean(cont_nll):.3f}      aic: {np.mean(cont_aic):.3f}      bic: {np.mean(cont_bic):.3f}
                 conf nll: {np.mean(conf_nll):.3f}      aic: {np.mean(conf_aic):.3f}      bic: {np.mean(conf_bic):.3f} 
         ''')
        
    return pd.DataFrame.from_dict(crs).round(3)

# ----- Human and model behaviors ----- #

def get_crs(exp, agents, method='mle'):
    
    cols = ['tot_nll', 'tot_aic', 'tot_bic', 
            'ass_nll', 'ass_aic', 'ass_bic', 
            'gen_nll', 'gen_aic', 'gen_bic', 
            'agent', 'sub_id']
    mat = {k: [] for k in cols}
    for m in agents:
        n_param = eval(m).n_params
        fname = f'{pth}/../analyses/{exp}/{m}/{method}-eval.csv'
        data  = pd.read_csv(fname)
        data['nll'] = -data['ll']
        
        group_data = data
        subj_Lst = group_data['sub_id'].unique()
        for sub_id in subj_Lst:
            
            # get subject and model info
            mat['sub_id'].append(sub_id)
            mat['agent'].append(m)

            # for all stage
            inll = group_data.query(f'sub_id=={sub_id}')['nll'].sum() 
            inum = group_data.query(f'sub_id=={sub_id}')['nll'].shape[0]
            mat['tot_nll'].append(inll)
            mat['tot_aic'].append(2*inll + 2*n_param)
            mat['tot_bic'].append(2*inll + np.log(inum)*n_param)

            # for the training stage
            anll = group_data.query(f'sub_id=={sub_id} & stage=="train"'
                                )['nll'].sum() 
            anum = group_data.query(f'sub_id=={sub_id} & stage=="train"'
                                )['nll'].shape[0]
            mat['ass_nll'].append(anll)
            mat['ass_aic'].append(2*anll + 2*n_param)
            mat['ass_bic'].append(2*anll + np.log(anum)*n_param)

            # for the testing stage
            gnll = group_data.query(f'sub_id=={sub_id} & stage=="test"'
                                )['nll'].sum() 
            gnum = group_data.query(f'sub_id=={sub_id} & stage=="test"'
                                )['nll'].shape[0]
            mat['gen_nll'].append(gnll)
            mat['gen_aic'].append(2*gnll + 2*n_param)
            mat['gen_bic'].append(2*gnll + np.log(gnum)*n_param)

    quant_mat = pd.DataFrame.from_dict(mat)
    crs_table = quant_mat.pivot(columns='agent', index='sub_id', 
            values=cols[:-2])
    
    return crs_table

def viz_model_cmp(crs_table, exp, agents, crs='bic'):
    
    sel_table = crs_table[f'tot_{crs}'].copy()
    sel_table[f'min_{crs}'] = sel_table.apply(
        lambda x: np.min([x[m] for m in agents]), 
        axis=1)
    sort_table = sel_table.sort_values(by=f'min_{crs}').reset_index()
    sort_table['sub_seq'] = sort_table.index

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    if ('rmPG_fea' in agents) or ('rmPG' in agents):
        m = 'rmPG' if exp=='exp1' else 'rmPG_fea'
        sns.scatterplot(x='sub_seq', y=m, 
                        data=sort_table, label=eval(m).name,
                        marker='s', color=[.7, .7, .7], 
                        s=20, alpha=.8,
                        edgecolor='none', ax=ax)
    if ('caPG_fea' in agents) or ('caPG' in agents):
        m = 'caPG' if exp=='exp1' else 'caPG_fea'
        sns.scatterplot(x='sub_seq', y=m, 
                        data=sort_table, label=eval(m).name,
                        marker='o', edgecolor=viz.r2, 
                        linewidth=1.1, s=20, alpha=.8,
                        facecolor='none', ax=ax)
    if ('LC' in agents):
        m = 'LC'
        sns.scatterplot(x='sub_seq', y=m, 
                        data=sort_table, label=eval(m).name,
                        marker='s', color=eval(m).color, 
                        s=20, alpha=.8,
                        edgecolor='none', ax=ax)
    if ('ACL' in agents):
        m = 'ACL'
        sns.scatterplot(x='sub_seq', y=m, 
                        data=sort_table, label=eval(m).name,
                        marker='x', color=eval(m).color, 
                        linewidth=1.1, s=20, alpha=.8,
                        ax=ax)
    if ('ecPG' in agents) or ('ecPG_fea' in agents):
        m = 'ecPG' if exp=='exp1' else 'ecPG_fea'
        sns.scatterplot(x='sub_seq', y=m, 
                        data=sort_table, label=eval(m).name,
                        marker='^', color=viz.Red, s=45,
                        edgecolor='none', ax=ax)
    n_data = (60+48)*2*2 if exp=='exp1' else (60+60)*3*2
    ax.axhline(y=-np.log(1/2)*n_data, xmin=0, xmax=1,
                    color='k', lw=1, ls='--')
    ax.set_xlim([-5, sort_table.shape[0]+15])
    ax.legend(loc='upper left')
    ax.spines['left'].set_position(('axes',-0.02))
    ax.set_xlabel(f'Participant index\n(sorted by the minimum {crs} score over all models)')
    ax.set_ylabel(crs.upper())
    fig.tight_layout()

def viz_corr(crs_table, exp, agents, crs='bic', method='mle'):
    
    sel_table = crs_table[f'tot_{crs}'].copy()
    sel_table[f'min_{crs}'] = sel_table.apply(
        lambda x: np.min([x[m] for m in agents]), 
        axis=1)
    sort_table = sel_table.sort_values(by=f'min_{crs}').reset_index()
    sort_table['sub_seq'] = sort_table.index
    # load human performance
    fname = f'{pth}/../analyses/{exp}/{agents[0]}/{method}-eval.csv'
    data  = pd.read_csv(fname)
    # get performance for different period 
    ass_acc  = data.query('group in ["control", "trained"]').groupby(
                        by=['sub_id'])['r'].mean()
    gen_acc  = data.query('group in ["untrained"]').groupby(
                        by=['sub_id'])['r'].mean()
    sort_table = sort_table.merge(ass_acc, on='sub_id'
                        ).rename(columns={'r':'train_acc'})
    sort_table = sort_table.merge(gen_acc, on='sub_id'
                        ).rename(columns={'r':'untrain_acc'})
    print(f'trained:\n{pg.corr(x=sort_table["sub_seq"], y=sort_table["train_acc"])}')
    print(f'untrained:\n{pg.corr(x=sort_table["sub_seq"], y=sort_table["untrain_acc"])}')

    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
    
    group  = ['Train', 'Untrained']
    colors = [viz.Blue, viz.Blue]
    for i, cr in enumerate(['train', 'untrain']):
        ax=axs[i]
        sns.scatterplot(x='sub_seq', y=f'{cr}_acc', 
                        data=sort_table, alpha=.7,
                        color=colors[i], s=15,
                        edgecolor=colors[i], ax=ax)

        ax.spines['left'].set_position(('axes',-0.05))
        ax.set_xlabel(f'Participant index')
        ax.set_ylabel(f'{group[i]} acc.')
        ax.set_box_aspect(1.3)
    fig.tight_layout()

def viz_pxp(exp, agents, method='mle'):

    palette = [eval(agent).color for agent in agents]

    # get bic 
    def getbic(exp, agents):
        fit_info = []
        for m in agents:
            n_param = eval(m).n_params
            fname = f'../analyses/{exp}/{m}/{method}-eval.csv'
            data  = pd.read_csv(fname)
            data['nll'] = -data['ll']
            
            group_data = data
            subj_Lst = group_data['sub_id'].unique()
            fit_sub_info = {'bic': []}
            for sub_id in subj_Lst:
                inll = group_data.query(f'sub_id=={sub_id}')['nll'].sum() 
                inum = group_data.query(f'sub_id=={sub_id}')['nll'].shape[0]
                fit_sub_info['bic'].append(2*inll + np.log(inum)*n_param)

            fit_info.append(fit_sub_info)
        
        return fit_info
    
    fit_info = getbic(agents=agents, exp=exp)  
    bms_res = fit_bms(fit_info, use_bic=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.4))
    print(bms_res['pxp'].round(2))
    sns.barplot(x=agents, y=bms_res['pxp'],
                edgecolor=palette, alpha=.8, 
                width=.5,
                lw=5, palette=palette, 
                ax=ax)
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_yticks([0, .5, 1])
    ax.set_ylabel('PXP')
    ax.set_xticks([i for i in range(len(agents))])
    ax.set_xticklabels([eval(agent).name for agent in agents])
    fig.tight_layout()

def viz_reduct(comb_data, agents):
    palette = [eval(agent).color for agent in agents]
    fig, ax = plt.subplots(1, 1, figsize=(2.9, 2.3), 
                            sharex=True, sharey=True)
    v = sns.violinplot(x='model', y='iSZ_reduct', data=comb_data, 
            palette=palette, hue='agent', hue_order=agents, 
            legend=False, alpha=.1, inner=None, scale='width',
            ax=ax)
    plt.setp(v.collections, alpha=.35, edgecolor='none')
    s = sns.stripplot(x='model', y='iSZ_reduct', data=comb_data, 
            palette=palette, hue='agent', hue_order=agents, 
            edgecolor='gray', dodge=True, jitter=True, alpha=.7,
            legend=False, zorder=2,
            ax=ax)
    b = sns.barplot(x='model', y='iSZ_reduct', data=comb_data, 
            hue='agent', hue_order=agents, 
            errorbar='se', linewidth=1, 
            edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
            capsize=.1, errwidth=2.5, errcolor=[.2, .2, .2],
            ax=ax)
    ax.legend().remove()
    ax.axhline(y=0, xmin=0, xmax=1, lw=1, 
        color=[.2, .2, .2], ls='--')
    ax.set_ylim([-.7, .3])
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_xlabel('')
    ax.set_xticks([-.25, 0, .25])
    ax.set_ylabel('bits reduce', fontsize=10)
    ax.set_box_aspect(1.4)
    fig.tight_layout()

def viz_transfer(exp, agents, method='mle'):
    data = []
    for agent in agents:
        df = pd.read_csv(f'{pth}/../analyses/{exp}/{agent}/{method}-base.csv')
        df['agent'] = agent
        data.append(df)
    data = pd.concat(data, axis=0)
    # rename the agent for figures
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.7))
    transfer(ax, data, agents)
    fig.tight_layout()

def viz_lc(exp, agents, method='mle'):
    n = len(agents)
    fig, axs = plt.subplots(n, 1, figsize=(4, n*1.8), sharex=True)
    for i, agent in enumerate(agents):
        ax = axs[i] if n>1 else axs
        lc(ax, exp, ['human', agent], method)
    fig.tight_layout()

def viz_transfer_cond(exp, agents, method='mle', goodPoor=None):

    conds = ['cons', 'cont', 'conf']
    title_map = {
        'cons': 'Consistent', 
        'cont': 'Control',
        'conf': 'Conflict'
    }
    
    data = [] 
    for agent in agents:
        df = pd.read_csv(f'{pth}/../analyses/{exp}/{agent}/{method}-base.csv')
        df['agent'] = agent
        data.append(df)
    data = pd.concat(data, axis=0)

    nr, nc = 2, 2
    ind = [0, 2, 3]
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharex=True, sharey=True)
    for i, cond in enumerate(conds):
        ax = axs[ind[i]//nc, ind[i]%nc]
        sel_cond =  f'stage=="test" & block_type=="{cond}" & group!="probe"'
        if goodPoor is not None: sel_cond += f'& goodPoor=="{goodPoor}"'
        sel_data = data.query(sel_cond)
        transfer(ax, sel_data, agents)
        ax.set_title(title_map[cond])
        if i: ax.set_ylabel('')
    axs[0, 1].set_axis_off()
    fig.tight_layout()
    
def viz_lc_cond(exp, agents, method='mle'):
    data = [] 
    conds = ['cons', 'cont', 'conf']
    for agent in agents:
        df = pd.read_csv(f'{pth}/../analyses/{exp}/{agent}/{method}-base.csv')
        df['agent'] = agent
        data.append(df)
    data = pd.concat(data, axis=0)
    data['block_type'] = data['block_type']

    nr, nc = len(agents), 1
    fig, axs = plt.subplots(nr, nc, figsize=(4, 1.8*nr), sharey=True, sharex=True)
    for i, agent in enumerate(agents):
        ax = axs[i]
        sel_data = data.query(f'group in ["trained", "control"] & agent=="{agent}"')
        sel_data = sel_data.groupby(by=['sub_id', 'tps', 'stage',
                    'agent', 'block_type']).mean(numeric_only=True).reset_index()
        sel_data['tps'] = sel_data.apply(
                lambda x: x['tps']+10 if x['stage']=='test' else x['tps'], axis=1)
        sns.lineplot(x='tps', y='acc', data=sel_data, 
                    err_style='bars', err_kws={'capsize': 3, 'elinewidth':2},
                    hue='block_type', hue_order=conds, alpha=.7,
                    palette=viz.Pal_type, lw=2.5,
                    errorbar='se', ax=ax)
        ax.vlines(x=10, ymin=.4, ymax=1, color='k', ls='--', lw=1.5)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xticklabels(['1', '6', '11', '16'])
        ax.set_xlabel('')
        ax.set_ylabel('Acc. (%)')
        ax.set_ylim([.4, 1])
        ax.spines['left'].set_position(('axes',-0.05))
        ax.set_title(f'{eval(agents[i]).name}')
        ax.get_legend().remove()
    fig.tight_layout()

def viz_Exp2Ans():
    xgroup = [1, 0, 1, 0]
    ygroup = [0, 1, 0, 1]
    titles = [r'Ans. for $x$', r"Ans. for $y'$"]
    fig, axs = plt.subplots(2, 1, figsize=(1*3, 2*2.6), sharey=True, sharex=True)
    for i, group in enumerate([xgroup, ygroup]):
        ax = axs[i]
        sns.barplot(x=[1, 2, 3, 4], y=group, color=viz.oGrey, ax=ax)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'])
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Proportion') 
        ax.set_xlim([-1, 4])
        ax.set_title(f'{titles[i]}')
        ax.spines['left'].set_position(('axes',-0.05))
    fig.tight_layout()

def viz_Probe(exp, agents, method, goodPoor=None):
    n = len(agents) 
    fig, axs = plt.subplots(n, 3, figsize=(2.5*3, n*2.3), sharey=True)
    for i, agent in enumerate(agents):
        if agent == 'human':
            fname = f'{pth}/../data/{exp}-human.csv'
            s = 1
        else:
            fname = f'{pth}/../simulations/{exp}/{agent}/sim-{method}.csv'
            s = 10
        data = pd.read_csv(fname, index_col=0)
        sel_data = data.query('group=="probe"').reset_index()

        for j, cond in enumerate(['cons', 'cont', 'conf']):
            ax = axs[i, j] if n > 1 else axs[j]
            sdata = sel_data.query(f'block_type=="{cond}"')
            if goodPoor is not None: sdata = sdata.query(f'goodPoor=="good"') 
            gdata = sdata.groupby(by=['sub_id', 'a', 'block_type']).count()['r'].reset_index()
            ptable = gdata.pivot_table(values='r', index='sub_id', columns='a').fillna(0) / (6*s)
            ptable.columns = [0, 1, 2, 3]
            ptable = ptable.reset_index()
            ptable = ptable.melt(id_vars='sub_id', value_vars=[0, 1, 2, 3]
                        ).rename(columns={'variable': 'a', 'value':'prop'})
        
            sns.stripplot(x='a', y='prop', data=ptable, 
                        color=viz.Pal_type[j], #dodge=True, 
                        edgecolor='gray', size=2.2,
                        jitter=True, alpha=.7,
                        legend=False, zorder=2,
                        ax=ax)
            v = sns.violinplot(x='a', y='prop', data=ptable,
                        errorbar='se', linewidth=1, 
                        inner=None, 
                        capsize=.3, errwidth=1.5, alpha=.45,
                        color=viz.Pal_type[j],
                        ax=ax)
            plt.setp(v.collections, alpha=.35, edgecolor='none')
            sns.barplot(x='a', y='prop', data=ptable,
                        errorbar='se', linewidth=2, width=.75,
                        edgecolor=viz.Pal_type[j], 
                        facecolor=viz.Pal_type[j].tolist()+[.2],
                        capsize=.05, errwidth=2.5, errcolor=[.2, .2, .2],
                        color='w', ax=ax)
            ax.set_box_aspect(.9)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_ylim([0, 1.1])
            if j==1: ax.set_title(eval(agent).name)
            ax.spines['left'].set_position(('axes',-0.05))
            ax.set_xticklabels([r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'])
    fig.tight_layout()

def viz_probe_kld(exp, agents, method, goodPoor=None):
    agents = ['human'] + agents

    # get probe policies
    probes = {}
    for agent in agents:
        agent_prob ={}
        if agent == 'human':
            fname = f'data/{exp}-human.csv'
            s = 1
        else:
            fname = f'simulations/{exp}/{agent}/sim-{method}.csv'
            s = 10
        data = pd.read_csv(fname, index_col=0)
        if goodPoor is not None: data = data.query(f'goodPoor=="{goodPoor}"')
        sel_data = data.query('group=="probe"').reset_index()

        for j, cond in enumerate(['cons', 'cont', 'conf']):
            sdata = sel_data.query(f'block_type=="{cond}"')
            gdata = sdata.groupby(by=['sub_id', 'a', 'block_type']).count()['r'].reset_index()
            ptable = gdata.pivot_table(values='r', index='sub_id', columns='a').fillna(0) / (6*s)
            ptable.columns = [a for a in [0, 1, 2, 3]]
            ptable = ptable.reset_index()
            agent_prob[cond] = ptable
        probes[agent] = agent_prob
    
    klds = {'diff': [], 'cond': [], 'agent': []}
    for cond in ['cons', 'cont', 'conf']:
        for agent in agents[1:]:
            tar = probes['human'][cond].values
            hat = probes[agent][cond].values
            kld = (tar*(np.log(tar+1e-13)-np.log(hat+1e-12))).sum(1).tolist()
            klds['diff']  += kld
            klds['cond']  += [cond]*len(kld)
            klds['agent'] += [agent]*len(kld)
    diff = pd.DataFrame.from_dict(klds)

    fig, ax = plt.subplots(1, 1, figsize=(4,2))
    palette = [eval(agent).color for agent in agents[1:]]
    sns.barplot(x='cond', y='diff', data=diff,
                hue='agent', hue_order=agents[1:],
                errorbar='se', capsize=.1, errwidth=2,
                alpha=.9, palette=palette)
    ax.get_legend().remove()
    ax.set_xlabel('')
    fig.tight_layout()

def viz_acc0_gen(data, agents):
    if 'rlPG' in agents:
        data['rlPG_gen'] = .5
        data['rlPG_acc0'] = 1
    fig, axs = plt.subplots(1, 3, figsize=(7, 3),
                        sharex=True, sharey=True)
    for i, m in enumerate(agents):
        ax = axs[i]
        sns.scatterplot(x=f'{m}_gen', y=f'{m}_acc0', data=data,
                    s=50, edgecolor=viz.Pal_agent[1+i], linewidth=1,
                    alpha=.7,
                    color=viz.Pal_agent[1+i], ax=ax)
        sns.regplot(x=f'{m}_gen', y=f'{m}_acc0', data=data,
                    scatter=False, truncate=False,
                    line_kws={'lw': 2.5}, ci=0,
                    color=[.1, .1, .1], ax=ax)
        if m != 'rlPG':
            print(pg.corr(x=data[f'{m}_gen'], y=data[f'{m}_acc0'], method='pearson'))
        ax.axvline(x=.5, ymin=0, ymax=1, lw=1, ls='--', color='k')
        ax.set_box_aspect(1)
        ax.set_xlim([.0, 1.1])
        ax.set_ylim([.2, 1.1])
        ax.spines['left'].set_position(('axes',-0.05))
        ax.set_xlabel('Untrained acc.')
        ax.set_ylabel('Init classification acc.')
    fig.tight_layout()

def viz_cmp_rdPG_rdPG0(exp, data):
    if exp=='exp1':
        m1, m2 = 'rdPG', 'rdPG0'
    else:
        m1, m2 = 'rdPG_fea', 'rdPG0_fea'
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    ax = axs[0, 0]
    v = sns.violinplot(data.loc[:,[m1, m2]], ax=ax, inner=None,
                    palette=[viz.Red, viz.r2])
    plt.setp(v.collections, alpha=.35, edgecolor='none')
    sns.stripplot(data.loc[:,[m1, m2]], edgecolor='w', 
                    ax=ax, alpha=.8, zorder=2,
                    palette=[viz.Red, viz.r2])
    sns.barplot(data.loc[:,[m1, m2]], 
                    errorbar='se', linewidth=1, 
                    edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                    capsize=.1, errwidth=2.5, errcolor=[.2, .2, .2],
                    ax=ax)
    #ax.legend().remove()
    ax.axhline(y=0, xmin=0, xmax=1, lw=1, ls='--', color='k')
    ax.set_ylim([-1.5, 1.2])
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_box_aspect(1)
    ax = axs[0, 1]
    sns.scatterplot(x=f'{m1}_gen', y=f'{m1}_acc0', data=data,
                    legend=False, s=50, edgecolor='none',
                    color=viz.Red, ax=ax)
    #ax.axhline(y=.25, xmin=0, xmax=1, lw=1, ls='--', color='k')
    ax.axvline(x=.5, ymin=0, ymax=1, lw=1, ls='--', color='k')
    ax.set_box_aspect(1)
    ax.set_xlabel('')
    ax.set_xlim([.25, 1.1])
    ax.set_ylim([.2, 1.1])
    ax.spines['left'].set_position(('axes',-0.05))
    print(pg.corr(x=data[f'{m1}_gen'], y=data[f'{m1}_gen'], method='pearson'))
    ax = axs[0, 2]
    sns.scatterplot(x=f'{m2}_gen', y=f'{m2}_acc0', data=data, 
                    legend=False, s=50, edgecolor='none',
                    color=viz.r2, ax=ax)
    #ax.axhline(y=.25, xmin=0, xmax=1, lw=1, ls='--', color='k')
    ax.axvline(x=.5, ymin=0, ymax=1, lw=1, ls='--', color='k')
    ax.set_box_aspect(1)
    ax.set_xlabel('')
    ax.set_xlim([.25, 1.1])
    ax.set_ylim([.2, 1.1])
    ax.spines['left'].set_position(('axes',-0.05))
    print(pg.corr(x=data[f'{m2}_gen'], y=data[f'{m2}_acc0'], method='pearson'))
    ax = axs[1, 0]
    ax.set_axis_off()
    ax = axs[1, 1]
    sns.scatterplot(x=f'{m1}_gen', y=m1, data=data,
                    hue=f'{m1}_acc0', legend=False, s=50, edgecolor='none',
                    palette=sns.diverging_palette(220, 20, as_cmap=True), ax=ax)
    ax.axhline(y=0, xmin=0, xmax=1, lw=1, ls='--', color='k')
    ax.axvline(x=.5, ymin=0, ymax=1, lw=1, ls='--', color='k')
    ax.set_box_aspect(1)
    ax.set_xlabel('')
    ax.set_xlim([.25, 1.1])
    ax.set_ylim([-1.5, 1.2])
    ax = axs[1, 2]
    sns.scatterplot(x=f'{m2}_gen', y=m2, data=data, 
                    hue=f'{m2}_acc0', legend=False, s=50, edgecolor='none',
                    palette=sns.diverging_palette(220, 20, as_cmap=True), ax=ax)
    ax.axhline(y=0, xmin=0, xmax=1, lw=1, ls='--', color='k')
    ax.axvline(x=.5, ymin=0, ymax=1, lw=1, ls='--', color='k')
    ax.set_box_aspect(1)
    ax.set_xlabel('')
    ax.set_xlim([.25, 1.1])
    ax.set_ylim([-1.5, 1.2])
    fig.tight_layout()

# ------- Model-based analyses -------- #

def sim_comp_time(sim_data, trials):
    sel_data = sim_data.query('stage=="train"').groupby(by=['trial']
                    ).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.5))
    sns.lineplot(x='trial', y='i_SZ', data=sel_data, 
                color=viz.Green, ax=ax)
    for t in trials:
        row = sel_data.query(f'trial=={t}')
        xr = row['trial'].values[0]
        yr = row['i_SZ'].values[0]
        ax.scatter(xr, yr, marker='x', color='k', s=40, zorder=10)
        ax.text(xr-4, yr+.12, f'{yr:.3f}', 
                    horizontalalignment='left', fontsize=12,
                    color='k', zorder=15)
    ax.set_ylim([.5, 2])
    ax.set_xlabel('Trial')
    ax.set_ylabel(f'Representation \ncomplexity (bits)')
    ax.spines['left'].set_position(('axes',-0.05))
    fig.tight_layout()

def get_assoc(enc):
    return .5*assoc(enc[0, :], enc[1, :]) + \
           .5*assoc(enc[2, :], enc[3, :])

def sim_Insight(voi, sim_data, tar, trials):

    if tar == 'enc':
        xl, yl = 'S', 'Z'
        xt, yt = [r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$'], [r"$x $ ", r"$x'$", r"$y $ ", r"$y'$"]
        cmap = viz.BluesMap
    elif tar == 'dec':
        xl, yl = 'Z', 'A'
        xt, yt = [r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'], [r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$']
        cmap = viz.GreensMap
    elif tar =='pol':
        xl, yl = 'S', 'A'
        xt, yt = [r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'],  [r"$x $ ", r"$x'$", r"$y $ ", r"$y'$"]
        cmap = viz.YellowsMap

    if tar == 'enc':
        i_SZs = sim_data.query('stage=="train"').groupby(by='trial'
                    ).mean(numeric_only=True).reset_index()['i_SZ'].values
        i_SZ  = [i_SZs[t] for t in trials]
    target = [voi[tar][t] for t in trials]
    if tar == 'enc': ass_val = [get_assoc(target[i]) for i in range(len(trials))] 
    
    nr, nc = 1, len(trials)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*2.8, nr*3.1), 
                            sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.95, .23, .01, .57])
    for idx in range(nc):
        ax = axs[idx]
        t = trials[idx]
        sns.heatmap(target[idx], square=True, lw=.5, 
                    cmap=cmap, vmin=0, vmax=1, 
                    cbar_ax=cbar_ax, ax=ax)
        
        if tar=='enc':
            ax.set_title(f"t={t}")#\nI(S;Z) = {i_SZ[idx]:.3f}")
            #ax.set_title(f"t={t}\nAssoc. = {ass_val[idx]:.3f}")
        else:
            ax.set_title(f't={t}')
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=target[idx].shape[1], color='k',lw=5)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=target[idx].shape[0], color='k',lw=5)
        ax.set_xticks([.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(xt)
        ax.xaxis.set_tick_params(length=0)
        ax.set_yticks([.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(yt, rotation=0)
        ax.yaxis.set_tick_params(length=0)

    plt.subplots_adjust(left=.15, bottom=.2, right=.85, top=.7)
    
def trainInfo():

    train_mat = np.array([[1,  0, 1, 0],
                          [.5,.5, 1, 0],
                          [0,  1, 0, 1],
                          [0,  1,.5,.5]])
    
    test_mat  = np.array([[1, 0, 1, 0],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 1, 0, 1]])

    xt, yt = [r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'],  [r"$x$ ", r"$x'$", r"$y$ ", r"$y'$"]
    
    fig, axs = plt.subplots(2, 1, figsize=(3, 2*3.1))
    for i, mat in enumerate([train_mat, test_mat]):
        ax = axs[i]
        sns.heatmap(mat, square=True, lw=.5, 
                        cmap=viz.RedsMap, vmin=0, vmax=1, 
                        cbar_kws={"shrink": .6}, ax=ax)
        ax.axhline(y=0, color='k',linewidth=5)
        ax.axhline(y=mat.shape[1], color='k',linewidth=5)
        ax.axvline(x=0, color='k',linewidth=5)
        ax.axvline(x=mat.shape[0], color='k',linewidth=5)
        ax.set_xticks([.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(xt)
        ax.xaxis.set_tick_params(length=0)
        # ax.set_xlabel(xl)
        ax.set_yticks([.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(yt, rotation=0)
        ax.yaxis.set_tick_params(length=0)
        # ax.set_ylabel(yl, rotation=0)
    plt.tight_layout()

def sim_Attn(exp, agent_name, block_types):
    lws = [6.5, 6.5, 3.5]
    voi ={}
    for block_type in block_types:
        fname = f'{pth}/../simulations/{exp}/{agent_name}/simsubj-{block_type}_voi.pkl'
        with open(fname, 'rb')as handle: voi[block_type] = pickle.load(handle)

    block_names = ['Consistent', 'Control', 'Conflict']
    features = ['Shape', 'Color', 'Appendage']
    tars = ['trial', 'attn', 'block_type', 'feature']
    df = {tar: [] for tar in tars}
    features = ['Shape', 'Color', 'Appendage']
    for i, b in enumerate(block_types):
        attn  = np.vstack(voi[b]['attn'])
        T = len(voi[b]['attn'])
        for j, f in enumerate(features):
            df['trial'] += list(range(T))
            df['attn'] += list(attn[:, j])
            df['block_type'] += [b]*T
            df['feature'] += [f]*T
    data = pd.DataFrame.from_dict(df)

    fig, axs = plt.subplots(1, 3, figsize=(5*3, 2.5), sharex=True)
    for k, b in enumerate(block_types):
        ax = axs[k]
        for i, f in enumerate(features):
            sns.lineplot(x='trial', y='attn', lw=lws[i],
                        data=data.query(f'block_type=="{b}" & feature=="{f}"'),
                        color=viz.Pal_fea[i], ax=ax, label=f)
            #ax.hlines(y=0, xmin=0, xmax=60, lw=1, ls='--', color='k')
        ax.legend().remove()
        ax.set_title(f'{block_names[k]}')
        #ax.set_ylim([.0, 5])
        ax.set_ylabel('Attention (a.u.)') 
        ax.set_box_aspect(.45)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.spines['left'].set_position(('axes',-0.05))
    
    fig.tight_layout()

def sim_Probe(exp, agent_name, block_types):

    voi ={}
    for block_type in block_types:
        fname = f'{pth}/../simulations/{exp}/{agent_name}/simsubj-{block_type}_voi-lmbda=0.1.pkl'
        with open(fname, 'rb')as handle: voi[block_type] = pickle.load(handle)
 
    nr, nc = 1, len(block_types)*2
    fig, axs = plt.subplots(nr, nc, figsize=(5*3, 3))
    for b, block_type in enumerate(block_types):
        ax = axs[b*2]
        if agent_name[:2] != 'rl':
            enc = voi[block_type]['enc'][59]
            sns.heatmap(enc, square=True, lw=.5, cbar=False,
                        cmap=viz.BluesMap, vmin=0, vmax=1, 
                        ax=ax)
            xl, yl = 'S', 'Z'
            xt, yt = [r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$'], [r"$x $", r"$x'$", r"$y $", r"$y'$", "probe"]
            ax.axhline(y=0, color='k',lw=5)
            ax.axhline(y=enc.shape[0], color='k',lw=5)
            ax.axhline(y=enc.shape[0]-1, color='k',lw=2)
            ax.axvline(x=0, color='k',lw=5)
            ax.axvline(x=enc.shape[1], color='k',lw=5)
            ax.set_xticks([.5, 1.5, 2.5, 3.5])
            ax.set_xticklabels(xt)
            ax.xaxis.set_tick_params(length=0)
            ax.set_yticks([.5, 1.5, 2.5, 3.5, 4.5])
            ax.set_yticklabels(yt, rotation=0)
            ax.yaxis.set_tick_params(length=0)
            ax.set_title(f'Encoder')
        else:
            ax.set_axis_off()

        ax = axs[b*2+1]
        pol = voi[block_type]['pol'][59]
        sns.heatmap(pol, square=True, lw=.5, cbar=False,
                    cmap=viz.YellowsMap, vmin=0, vmax=1, 
                    ax=ax)
        xl, yl = 'S', 'A'
        xt, yt = [r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'],  [r"$x$"+" ", r"$x'$", r"$y$"+" ", r"$y'$", "probe"]
        ax.axhline(y=0, color='k',lw=5)
        ax.axhline(y=pol.shape[0], color='k',lw=5)
        ax.axhline(y=pol.shape[0]-1, color='k',lw=2)
        ax.axvline(x=0, color='k',lw=5)
        ax.axvline(x=pol.shape[1], color='k',lw=5)
        ax.set_xticks([.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(xt)
        ax.xaxis.set_tick_params(length=0)
        ax.set_yticks([.5, 1.5, 2.5, 3.5, 4.5])
        ax.set_yticklabels([], rotation=0)
        ax.yaxis.set_tick_params(length=0)
        ax.set_title(f'Policy')
    fig.tight_layout()

def corr_matrix(exp, agents = ['rlPG_fea', 'cascade_fea', 'rdPG_fea'], method='mle', goodPoor=None):
    p_tables = {}
    conds = ['cons', 'cont', 'conf']
    for i, agent in enumerate(['human']+agents):
        if agent == 'human':
            fname = f'{pth}/../data/{exp}-human.csv'
            s = 1
        else:
            fname = f'{pth}/../simulations/{exp}/{agent}/sim-{method}.csv'
            s = 10
        data = pd.read_csv(fname, index_col=0)
        sel_data = data.query('group=="probe"').reset_index()
        pp = {}
        for j, cond in enumerate(['cons', 'cont', 'conf']):
            sdata = sel_data.query(f'block_type=="{cond}"')
            if goodPoor is not None: sdata = sdata.query(f'goodPoor=="good"') 
            gdata = sdata.groupby(by=['sub_id', 'a', 'block_type']).count()['r'].reset_index()
            ptable = gdata.pivot_table(values='r', index='sub_id', columns='a').fillna(0) / (6*s)
            ptable.columns = [0, 1, 2, 3]
            ptable = ptable.reset_index()
            pp[cond] = ptable.sort_values(by='sub_id').loc[:, [0, 1, 2, 3]].values
        p_tables[agent] = pp
    corr_data = {'corr': [], 'agent':[], 'cond':[]}
    for i, cond in enumerate(conds):
        y = p_tables['human'][cond][:, 0::2].reshape([-1])
        for agent in agents:
            x = p_tables[agent][cond][:, 0::2].reshape([-1])
            corr_lm = pg.corr(x, y, method='spearman')
            #print(corr_lm)
            r = corr_lm["r"][0]
            corr_data['corr'].append(r)
            corr_data['agent'].append(agent)
            corr_data['cond'].append(cond)
    corr_data = pd.DataFrame.from_dict(corr_data).pivot_table(
                    values='corr', index='cond', columns='agent'
                    ).loc[['cons', 'cont', 'conf'], agents]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.heatmap(corr_data, cmap="coolwarm",  center=.3, vmax=1.2,
                square=True, annot=True, cbar_kws={"shrink": .5},
                lw=1, ax=ax)
    ax.set_xticklabels([eval(agent).name for agent in agents])
    ax.set_yticklabels(['Consistent', 'Control', 'Conflict'])
    ax.set_xlabel('')
    ax.set_ylabel('')


