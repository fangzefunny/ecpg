import pickle 
import numpy as np
import warnings
import time 

from functools import partial
from scipy.special import softmax, psi, gammaln
from scipy.stats import gamma, norm 
from scipy.optimize import minimize
from pybads.bads import BADS

eps_ = 1e-13
max_ = 1e+13

# -----------------------------------------------#
#           Hierarchical optimization            #
# -----------------------------------------------#

def fit_hier(pool, data, model, fname, n_fits=20, 
             seed=2020, tol=1e-4, max_iter=10, 
             init=None, verbose=True):
    '''Hierarchical model fitting, searching for prior

    ----------------------------------------------------------------
    REFERENCES:
    
    Huys, Q. J., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, 
    R. J., & Dayan, P. (2011). Disentangling the roles of approach, 
    activation and valence in instrumental and pavlovian responding. 
    PLoS computational biology, 7(4), e1002028.
    -------------------------------------------------------------------
    Based on: https://github.com/sjgershm/mfit

    @ ZF
    '''

     # number of parameter, and possible bound
    n_param = model.agent.n_params
    m_data  = list(data.items())[0][1][0].shape[0]
    sub_lst = list(data.keys())
    n_sub   = len(sub_lst)
    plb = np.array([b[0] for b in model.agent.p_pbnds])
    pub = np.array([b[1] for b in model.agent.p_pbnds])

    # init group-level parameters
    if init is not None:
        mus = init[0]
        vs  = init[1]
    else:    
        mus = plb + .5*(pub-plb)
        vs  = pub-plb

    # run EM until converge
    epi = 0 
    lme = 0
    while True:
        epi += 1
        prev_lme = lme 
        print(f'\nGroup-level Iteration: {epi}')

        # construct prior
        p_priors = [norm(mu, np.sqrt(v)) for mu, v in zip(mus, vs)]
        
        # E-step: optimize individual parameters
        fit_info = {}
        for i, sub_idx in enumerate(sub_lst):
            start_time = time.time()
            sub_fit = model.fit(data[sub_idx], 'map', 'BFGS', pool, p_priors,
                                seed=seed, n_fits=n_fits,
                                verbose=False, init=False)
            fit_info[sub_idx] = sub_fit
            end_time = time.time()
            if verbose: 
                interval = end_time - start_time
                print(f'SUB:{sub_idx}, progress: {(i*100/n_sub):2f}%')
                print(f'\tNLL:{-sub_fit["log_like"]:.4f}, using: {interval:.2f}')
                
        # transform the parameter to Gaussian space,
        # using link function 
        params = []
        for _, item in fit_info.items():
            params.append(item['param'])
        params = np.vstack(params) # n_sub x n_param

        # M-step: update group-level parameters 
        # u = 1/N \sum_i m_i
        mus  = np.mean(params, axis=0)
        # v2 = 1/N \sum_i [m_i^2 + ∑^2] - mu^2
        vs = 0
        group_ll, good_h = [], []
        for i, (_, item) in enumerate(fit_info.items()):
            vs += (params[i, :])**2 + np.diag(item['H_inv'])
            try:
                log_h = np.linalg.slogdet(item['H'])[1]
                l = item['log_post'] + .5*(n_param*np.log(2*np.pi) - log_h)
                gh = 1
            except:
                warnings.warn('Hessian could not be calculated')
                l = np.nan 
                gh = 0
                continue
            group_ll.append(l)
            good_h.append(gh)
        # make sure the variance is not to small 
        vs = np.clip(vs/n_sub - mus**2, a_min=1e-5, a_max=np.inf)
        lme = np.sum(group_ll)-n_param*np.log(m_data*n_sub)

        # finish this round and preprare for the next
        print(f'Finish {epi}-th iteration: \tThe group LME is {lme:.3f}')
        with open(fname, 'wb')as handle: pickle.dump(fit_info, handle)
            
        # check convergence
        done = (np.abs(lme - prev_lme) < tol) or (epi >= max_iter)
        if done: 
            group_fit = {
                'group_lme': lme,
                'group_mu':  mus, 
                'group_var': vs,
            }
            fit_info['group'] = group_fit
            break 
        
    return fit_info

# -----------------------------------------------#
#         Maximum likelihoood Estimation         #
# -----------------------------------------------#

def fit(loss_fn, data, bnds, pbnds, p_name, p_priors,
        method='mle', alg='Nelder-Mead', init=False, seed=2021, 
        verbose=False):
    '''Fit the parameter using optimization 

    Args: 

        loss_fn: a function; log likelihood function
        data:  a dictionary, each key map a dataframe
        bnds: parameter bound
        pbnds: possible bound, used to initialize parameter
        priors: a list of scipy random variable, used to
                calculate log prior
        p_name: the names of parameters
        method: decide if we use the prior -'mle', -'map', -'hier'
        alg: the fiting algorithm, currently we can use 
            - 'Nelder-Mead': a simplex algorithm,
            - 'BFGS': a quasi-Newton algorithm, return hessian,
                        but only works on unconstraint problem
            - 'bads': bayesian optimization problem
        init:  input the init parameter if needed 
        seed:  random seed; used when doing parallel computing
        verbose: show the optimization details or not. 
    
    Return:
        result: optimization results

    @ZF
    '''
    # get some value
    n_params = len(p_name)
    if method=='mle': p_priors=None 
    if alg=='BFGS': bnds=None
    # get the number of trial 
    n_rows = np.sum([data[k].shape[0] for k in data.keys()])

    # Init params
    if init:
        # if there are assigned params
        param0 = init
    else:
        # random init from the possible bounds 
        rng = np.random.RandomState(seed)
        param0 = [pbnd[0] + (pbnd[1] - pbnd[0]
                    ) * rng.rand() for pbnd in pbnds]
                    
    ## Fit the params 
    if verbose: print('init with params: ', param0) 
    if alg=='bads':
        def loss_fn_bads(x): 
            return loss_fn(x, data, p_priors)
        lb  = np.array([bnd[0] for bnd in bnds])
        ub  = np.array([bnd[1] for bnd in bnds])
        plb = np.array([pbnd[0] for pbnd in pbnds])
        pub = np.array([pbnd[1] for pbnd in pbnds])
        bads = BADS(loss_fn_bads, param0, lb, ub, plb, pub, 
                    options={"display" : 'off'})
        result = bads.optimize()
        x_min = result['x']
        f_min = result['fval']

    else:
        result = minimize(loss_fn, param0, args=(data, p_priors), 
                    bounds=bnds, method=alg,
                    options={'disp': verbose})
        x_min = result.x
        f_min = result.fun
    if verbose: print(f'''  Fitted params: {x_min}, 
                                Loss: {f_min}''')
            
    ## Save the optimize results 
    fit_res = {}
    fit_res['log_post']   = -f_min
    fit_res['log_like']   = -loss_fn(x_min, data, None)
    fit_res['param']      = x_min
    fit_res['param_name'] = p_name
    fit_res['n_param']    = n_params
    fit_res['aic']        = n_params*2 - 2*fit_res['log_like']
    fit_res['bic']        = n_params*np.log(n_rows) - 2*fit_res['log_like']
    if alg == 'BFGS':
        fit_res['H'] = np.linalg.pinv(result.hess_inv)
        fit_res['H_inv'] = result.hess_inv

    return fit_res

# ------------------------------------------------------#
#         Maximum likelihoood Estimation parallel       #
# ------------------------------------------------------#

def fit_parallel(pool, loss_fn, data, bnds, pbnds, p_name,              
                 p_priors, method='mle', alg='Nelder-Mead', 
                 init=False, seed=2021, verbose=False, n_fits=40):
    '''Fit the parameter using optimization, parallel 

    Args: 
        pool:  computing pool; mp.pool
        loss_fn: a function; log likelihood function
        data:  a dictionary, each key map a dataframe
        bnds: parameter bound
        pbnds: possible bound, used to initialize parameter
        priors: a list of scipy random variable, used to
                calculate log prior
        p_name: the names of parameters
        method: decide if we use the prior -'mle', -'map', -'hier'
        alg: the fiting algorithm, currently we can use 
            - 'Nelder-Mead': a simplex algorithm,
            - 'BFGS': a quasi-Newton algorithm, return hessian,
                        but only works on unconstraint problem
            - 'bads': bayesian optimization problem
        init:  input the init parameter if needed 
        seed:  random seed; used when doing parallel computing
        n_fits: number of fit 
        verbose: show the optimization details or not. 
    
    Return:
        result: optimization results

    @ZF
    '''
    results = [pool.apply_async(fit, 
                    args=(loss_fn,
                          data, 
                          bnds,
                          pbnds, 
                          p_name, 
                          p_priors,             
                          method, 
                          alg, 
                          init, 
                          seed+2*i,    
                          verbose)
                    ) for i in range(n_fits)]
    opt_val   = np.inf 
    losses, tol = [], 1e-2,
    for p in results:
        res = p.get()
        losses.append(-res['log_post'])
        if -res['log_post'] < opt_val:
            opt_val = -res['log_post']
            opt_res = res
    n_low = (np.abs(np.array(losses)-opt_val)<tol).sum()
    print(f'\tNum of lowest loss: {n_low}/{len(losses)}')
            
    return opt_res 

# ------------------------------------------------------#
#             Bayesian group level comparison           #
# ------------------------------------------------------#

def fit_bms(all_sub_info, use_bic=False, tol=1e-4):
    '''Fit group-level Bayesian model seletion
    Nm is the number of model
    Args: 
        all_sub_info: [Nm, list] a list of model fitting results
        use_bic: use bic to approximate lme
        tol: 
    Outputs:
        BMS result: a dict including 
            -alpha: [1, Nm] posterior of the model probability
            -p_m1D: [nSub, Nm] posterior of the model 
                     assigned to the subject data p(m|D)
            -E_r1D: [nSub, Nm] expectation of E[p(r|D)]
            -xp:    [Nm,] exceedance probabilities
            -bor:   [1] Bayesian Omnibus Risk, the probability
                    of choosing null hypothesis: model frequencies are equal
            -pxp:   [Nm,] protected exceedance probabilities
    ----------------------------------------------------------------
    REFERENCES:
    
    Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
    
    Rigoux, L, Stephan, KE, Friston, KJ and Daunizeau, J. (2014)
    Bayesian model selection for group studiesRevisited.
    NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
    -------------------------------------------------------------------
    Based on: https://github.com/sjgershm/mfit
    @ ZF
    '''
    ## get log model evidence
    if use_bic:
        lme = np.vstack([-.5*np.array(fit_info['bic']) for fit_info in all_sub_info]).T
    else:
        lme = np.vstack([calc_lme(fit_info) for fit_info in all_sub_info]).T
    
    ## get group-level posterior
    Nm = lme.shape[1]
    alpha0, alpha = np.ones([1, Nm]), np.ones([1, Nm])

    while True:
        
        # cache previous α
        prev = alpha.copy()

        # compute the posterior: Nsub x Nm
        # p(m|D) (p, k) = exp[log p(D(p,1))|m(p,k)) + Psi(α(1,k)) - Psi(α'(1,1))]
        log_u = lme + psi(alpha) - psi(alpha.sum())
        u = np.exp(log_u - log_u.max(1, keepdims=True)) # the max trick 
        p_m1D = u / u.sum(1, keepdims=True)

        # compute beta: 1 x Nm
        # β(k) = sum_p p(m|D)
        B = p_m1D.sum(0, keepdims=True)

        # update alpha: 1 x Nm
        # α(k) = α0(k) + β(k) 
        alpha = alpha0 + B 

        # check convergence 
        if np.linalg.norm(alpha - prev) < tol:
            break 
    
    # get the expected posterior 
    E_r1D = alpha / alpha.sum()

    # get the exeedence probabilities 
    xp = dirchlet_exceedence(alpha)

    # get the Bayesian Omnibus risk
    bor = calc_BOR(lme, p_m1D, alpha, alpha0)
    
    # get the protected exeedence probabilities
    pxp=(1-bor)*xp+bor/Nm

    # out BMS fit 
    BMS_result = { 'alpha_post': alpha, 'p_m1D': p_m1D, 
                   'E_r1D': E_r1D, 'xp': xp, 'bor': bor, 'pxp': pxp}

    return BMS_result

def calc_lme(fit_info):
    '''Calculate Log Model Evidence
    Turn a list of fitting results of different
    model into a matirx lme. Ns means number of subjects, 
    Nm is the number of models.
    Args:
        fit_info: [dict,] A dict of model's fitting info
            - log_post: opt parameters
            - log_like: log likelihood
            - param: the optimal parameters
            - n_param: the number of parameters
            - aic
            - bic
            - H: hessian matrix 
    
    Outputs:
        lme: [Ns, Nm] log model evidence 
                
    '''
    lme  = []
    for s in range(len(fit_info['log_post'])):
        # log|-H|
        h = np.log(np.linalg.det(fit_info['H'][s]))
        # log p(D,θ*|m) + .5(log(d) - log|-H|) 
        l = fit_info['log_post'][s] + \
            .5*(fit_info['n_param']*np.log(2*np.pi)-h)
        lme.append(l)
        
    # use BIC if any Hessians are degenerate 
    ind = np.isnan(lme) | np.isinf(lme)| (np.imag(lme)!=0)
    if any(ind.reshape([-1])): 
        warnings.warn("Hessians are degenerated, use BIC")
        lme = -.5 * np.array(fit_info['bic'])
            
    return np.array(lme)

def dirchlet_exceedence(alpha_post, nSample=1e6):
    '''Sampling to calculate exceedence probability
    Args:
        alpha: [1,Nm] dirchilet distribution parameters
        nSample: number of samples
    Output: 
    '''
    # the number of categories
    Nm = alpha_post.shape[1]
    alpha_post = alpha_post.reshape([-1])

    # sampling in blocks
    blk = int(np.ceil(nSample*Nm*8 / 2**28))
    blk = np.floor(nSample/blk * np.ones([blk,]))
    blk[-1] = nSample - (blk[:-1]).sum()
    blk = blk.astype(int)

    # sampling 
    xp = np.zeros([Nm,])
    for i in range(len(blk)):

        # sample from a gamma distribution and normalized
        r = np.vstack([gamma(a).rvs(size=blk[i]) for a in alpha_post]).T

        # use the max decision rule and count 
        xp += (r == np.amax(r, axis=1, keepdims=True)).sum(axis=0)

    return xp / nSample

# -------- Bayesian Omnibus Risk -------- #

def calc_BOR(lme, p_m1D, alpha_post, alpha0):
    '''Calculate the Bayesian Omnibus Risk
     Args:
        lme: [Nsub, Nm] log model evidence
        p_r1D: [Nsub, Nm] the posterior of each model 
                        assigned to the data
        alpha_post:  [1, Nm] H1: alpha posterior 
        alpha0: [1, Nm] H0: alpha=[1,1,1...]
    Outputs:
        bor: the probability of selection the null
                hypothesis.
    '''
    # calculte F0 and F1
    f0 = F0(lme)
    f1 = FE(lme, p_m1D, alpha_post, alpha0)

    # BOR = 1/(1+exp(F1-F0))
    bor = 1 / (1+ np.exp(f1-f0))
    return bor 

def F0(lme):
    '''Calculate the negative free energy of H0
    Args:
        lme: [Nsub, Nm] log model evidence
    Outputs:
        f0: negative free energy as an approximation
            of log p(D|H0)
    '''
    Nm = lme.shape[1]
    qm = softmax(lme, axis=1)    
    f0 = (qm * (lme - np.log(Nm) - np.log(qm + eps_))).sum()                                  
    return f0
    
def FE(lme, p_m1D, alpha_post, alpha0):
    '''Calculate the negative free energy of H1
    Args:
        lme: [Nsub, Nm] log model evidence
        p_m1D: [Nsub, Nm] the posterior of each model 
                        assigned to the data
        alpha_post:  [1, Nm] H1: alpha posterior 
        alpha0: [1, Nm] H0: alpha=[1,1,1...]
    Outputs:
        f1: negative free energy as an approximation
            of log p(D|H1)
    '''
    E_log_r = psi(alpha_post) - psi(alpha_post.sum())
    E_log_rmD = (p_m1D*(lme+E_log_r)).sum() + ((alpha0 -1)*E_log_r).sum()\
                + gammaln(alpha0.sum()) - (gammaln(alpha0)).sum()
    Ent_p_m1D = -(p_m1D*np.log(p_m1D + eps_)).sum()
    Ent_alpha  = gammaln(alpha_post).sum() - gammaln(alpha_post.sum()) \
                                        - ((alpha_post-1)*E_log_r).sum()
    f1 = E_log_rmD + Ent_p_m1D + Ent_alpha
    return f1
