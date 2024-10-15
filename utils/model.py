import os 
import numpy as np 
import pandas as pd 
from copy import deepcopy

from functools import lru_cache
from scipy.special import softmax 
from scipy.stats import halfnorm, uniform

from utils.fit import *
from utils.env_fn import *
from utils.viz import *

pth = os.path.dirname(os.path.abspath(__file__))

eps_ = 1e-13
max_ = 1e+13

# ------------------------------#
#          Axuilliary           #
# ------------------------------#

def mask_fn(nA, a_ava):
    return (np.eye(nA)[a_ava, :]).sum(0, keepdims=True)

def MI(p_X, p_Y1X, p_Y):
    return (p_X*p_Y1X*(np.log(p_Y1X+eps_)-np.log(p_Y.T+eps_))).sum()

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    return np.exp(x) 

def step(w, lr):
    w.data -= lr*w.grad.data
    w.grad.data.zero_()

# ------------------------------#
#         Agent wrapper         #
# ------------------------------#

class wrapper:
    '''Agent wrapper

    We use the wrapper to

        * Fit
        * Simulate
        * Evaluate the fit 
    '''

    def __init__(self, agent, env_fn):
        self.agent  = agent
        self.env_fn = env_fn
        self.use_hook = False
    
    # ------------ fit ------------ #

    def fit(self, data, method, alg, pool=None, p_priors=None,
            init=False, seed=2021, verbose=False, n_fits=40):
        '''Fit the parameter using optimization 
        '''

        # get functional inputs 
        fn_inputs = [self.loss_fn, 
                     data, 
                     self.agent.p_bnds,
                     self.agent.p_pbnds, 
                     self.agent.p_names,
                     self.agent.p_priors if p_priors is None else p_priors,
                     method,
                     alg, 
                     init,
                     seed,
                     verbose]
        
        if pool:
            sub_fit = fit_parallel(pool, *fn_inputs, n_fits=n_fits)
        else: 
            sub_fit = fit(*fn_inputs)  

        return sub_fit      

    def loss_fn(self, params, sub_data, p_priors=None):
        '''Total likelihood

        Fit individual:
            Maximum likelihood:
            log p(D|θ) = log \prod_i p(D_i|θ)
                       = \sum_i log p(D_i|θ )
            or Maximum a posterior 
            log p(θ|D) = \sum_i log p(D_i|θ ) + log p(θ)
        '''
        # negative log likelihood
        tot_loglike_loss  = -np.sum([self.loglike(params, sub_data[key])
                    for key in sub_data.keys()])
        # negative log prior 
        if p_priors==None:
            tot_logprior_loss = 0 
        else:
            p_trans = [fn(p) for p, fn in zip(params, self.agent.p_trans)]
            tot_logprior_loss = -self.logprior(p_trans, p_priors)
        # sum
        return tot_loglike_loss + tot_logprior_loss

    def loglike(self, params, block_data):
        '''Likelihood for one sample
        -log p(D_i|θ )
        In RL, each sample is a block of experiment,
        Because it is independent across experiment.
        '''
        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)
        ll   = 0
       
        ## loop to simulate the responses in the block 
        for t, row in block_data.iterrows():

            # predict stage: obtain input
            ll += env.eval_fn(row, subj)

        return ll
          
    def logprior(self, params, p_priors):
        '''Add the prior of the parameters
        '''
        lpr = 0
        for pri, param in zip(p_priors, params):
            lpr += np.max([pri.logpdf(param), -max_])
        return lpr

    # ------------ evaluate ------------ #

    def eval(self, data, params):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            sim_data.append(self.eval_block(block_data, params))
        return pd.concat(sim_data, ignore_index=True)
    
    def eval_block(self, block_data, params):

        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = ['ll'] + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # record some insights of the model
            for v in self.agent.voi:
                pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # simulate the data 
            ll = env.eval_fn(row, subj)
            
            # record the stimulated data
            pred_data.loc[t, 'll'] = ll

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)

    # ------------ simulate ------------ #

    def sim(self, data, params, rng):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            for v in self.env_fn.voi:
                if v in block_data.columns:
                    block_data = block_data.drop(columns=v)
            sim_data.append(self.sim_block(block_data, params, rng))
        
        return pd.concat(sim_data, ignore_index=True)

    def sim_block(self, block_data, params, rng):

        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = self.env_fn.voi + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # record some insights of the model
            for i, v in enumerate(self.agent.voi):
                pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # if register hook to get the model insights
            if self.use_hook:
                for k in self.insights.keys():
                    self.insights[k].append(eval(f'subj.get_{k}()'))

            # simulate the data 
            subj_voi = env.sim_fn(row, subj, rng)
            
            # record the stimulated data
            for i, v in enumerate(env.voi): 
                pred_data.loc[t, v] = subj_voi[i]

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)
    
    def register_hooks(self, *args):
        self.use_hook = True 
        self.insights = {k: [] for k in args}

# ------------------------------#
#         Memory buffer         #
# ------------------------------#

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = {k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

# ------------------------------#
#             Base              #
# ------------------------------#

class base_agent:
    '''Base Agent'''
    name     = 'base'
    p_bnds   = None
    p_pbnds  = []
    p_names  = []  
    p_priors = []
    p_trans  = []
    p_links  = []
    n_params = 0 
    # value of interest, used for output
    # the interesting variable in simulation
    voi      = []
    insights = ['pol']
    
    def __init__(self, env, params):
        self.env = env 
        self.nS  = env.nS
        self.nA  = env.nA 
        self.nD  = env.nD 
        self.nF  = env.nF
        self.nI  = env.nI
        self.nProbe = env.nProbe
        self.load_params(params)
        self._init_embed()
        self._init_buffer()
        self._init_agent()
        
    def load_params(self, params): 
        return NotImplementedError
    
    def _init_embed(self):
        self.embed = self.env.embed
        self.s2f   = self.env.s2f
        self.F     = np.vstack([self.embed(self.s2f(s)) for s in self.env.stimuli])

    def _init_buffer(self):
        self.mem = simpleBuffer()
    
    def _init_agent(self):
        return NotImplementedError
    
    def learn(self): 
        return NotImplementedError

    def policy(self, s, **kwargs): 
        return NotImplementedError

    # --------- Insights -------- #

    def get_pol(self):
        acts = [[0, 1], [2, 3]]
        pi = np.zeros([self.nS, self.nA])
        for s in range(self.nS):
            for act in acts:
                a1, a2 = act
                pi[s, :] += self.policy(self.s2f(s), a_ava=[a1, a2])   
        return pi

class human:
   name  = 'Human'
   color = viz.Blue    

class fea_base:

    def get_pol(self):
        acts = [[0, 1], [2, 3]]
        n = self.nS+self.nProbe
        pi = np.zeros([n, self.nA])
        for s in range(n):
            for act in acts:
                a_ava1, a_ava2 = act
                f = self.s2f(s)
                pi[s, :] += self.policy(f, s=s,
                            a_ava=[a_ava1, a_ava2])   
        return pi  

# ------------------------------#
#         Classic models        #
# ------------------------------#

class rmPG(base_agent):
    name     = 'RMPG'
    p_names  = ['alpha'] 
    p_bnds   = [(0, 1000)]
    p_pbnds  = [(-2, 2)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_names)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_names)
    n_params = len(p_names)
    voi      = ['i_SZ']
    insights = ['pol']
    color    = np.array([200, 200, 200]) / 255
    marker   = 's'
    size     = 20
    alpha    = .8

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_pi = params[0]
        self.b        = 1/2
    
    def _init_agent(self):
        self.phi    = np.zeros([self.nS, self.nA]) 
    
    def learn(self):
        self._learn_dec()

    def _learn_dec(self):

        # get data 
        fstr, a_ava, a, r = self.mem.sample('f', 'a_ava', 'a', 'r')
        
        # forward
        f = self.embed(fstr)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1s = softmax(f@self.phi-(1-m_A)*max_, axis=1)
        u_sa = r - self.b

        # backward
        gPhi = -f.T@(np.eye(self.nA)[a, :] - p_A1s)*u_sa
        self.phi -= self.alpha_pi*gPhi

    def policy(self, fstr, **kwargs):
        f   = self.embed(fstr)
        m_A = mask_fn(self.nA, kwargs['a_ava'])
        pi  = softmax(f@self.phi-(1-m_A)*max_, axis=1)
        return pi.reshape([-1])

    def get_i_SZ(self): return 1

class ecPG_sim(base_agent):
    '''Efficient coding policy gradient (analytical)

    We create two mathematical equivalent versions for ECPG, each
    aim at tackling different numerical problems. 
    The only difference exist in calculating sTheta.

    The analytical version, we do:
        sTheta = (u*p_a1Z.T - self.lmbda*log_dif)
    and in the fitting version, we do:
        sTheta = (u*p_a1Z.T/(self.lmdba+eps_) - log_dif)

    The analytical version is more consistent with the objective function.
    It is easy to illustrate the model behaviors with varying
    λ, because it will not divided by 0. 

    However, this implementation causes high correlation
    between parameters α_ψ and λ，bring difficulties in parameter
    estimation. 

    The same logics applied to fECPG.
    '''
    name     = 'ECPG'
    p_names  = ['alpha_psi', 'alpha_rho', 'lmbda']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 3), (-2, 3), (-6, 1.5)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_names)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_names)
    n_params = len(p_names)
    voi      = ['i_SZ']
    insights = ['enc', 'dec', 'pol']
    color    = viz.Red
    marker   = '^'
    size     = 125
    alpha    = 1

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.lmbda      = params[2]
        self.b          = .5

    def _init_agent(self):
        self.nZ = self.nS
        fname = f'{pth}/../data/exp1_ecpg_weight.pkl'
        with open(fname, 'rb')as handle: theta = pickle.load(handle)
        self.theta = deepcopy(theta)
        self.phi   = np.zeros([self.nS, self.nA]) 
        self._learn_pZ()

    def _learn_pZ(self):
        self.p_Z1S = softmax(self.F@self.theta, axis=1)
        self.p_S   = np.ones([self.nS, 1]) / self.nS  # nSx1 
        self.p_Z   = self.p_Z1S.T @ self.p_S  

    def policy(self, fstr, **kwargs):
        f = self.embed(fstr)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, kwargs['a_ava'])
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        # renormalize to avoid numeric problem
        pi = (p_Z1s@p_A1Z).reshape([-1])
        return pi / pi.sum()
        
    def learn(self):
        self._learn_enc_dec()
        self._learn_pZ()

    def _learn_enc_dec(self):
        # get data 
        fstr, a_ava, a, r = self.mem.sample('f', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = self.embed(fstr)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        # note: in derviation we wrote 
        #          log_dif = log p(Z|S) - log p(Z) - 1
        # However, substracting the constant 1 will not affect the 
        # numerical value of gradeint, due to the normalization 
        # term in calculating gTheta.
        log_dif = np.log(p_Z1s+eps_)-np.log(self.p_Z.T+eps_)  
        sTheta = (u*p_a1Z.T - self.lmbda*log_dif)
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi
    
    # --------- some predictions ----------- #
        
    def get_i_SZ(self):
        psi_Z1S = softmax(self.F@self.theta, axis=1)
        return MI(self.p_S, psi_Z1S, self.p_Z)
    
    def get_i_ZA(self):
        rho_A1Z = softmax(self.phi, axis=1)
        p_A = (self.p_Z.T @ rho_A1Z).T
        return MI(self.p_Z, rho_A1Z, p_A)
        
    def get_enc(self):
        return softmax(self.F@self.theta, axis=1)
    
    def get_dec(self):
        acts = [[0, 1], [2, 3]]
        rho  = 0
        for act in acts:
            m_A   = mask_fn(self.nA, act)
            p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
            p_A1Z  /= p_A1Z.sum(1, keepdims=True)
            rho += p_A1Z
        return rho 
    
class ecPG(ecPG_sim):
    '''Efficient coding policy gradient (fitting)

    This fitting version is created to decorrelatete 
    parameters α_ψ and λ, accelerating the model fitting process
    and improving precision. However, it comes across the
    divided by 0 problem. 

    The same logics applied to fECPG.
    '''

    def _learn_enc_dec(self):
        # get data 
        fstr, a_ava, a, r = self.mem.sample('f', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = self.embed(fstr)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        log_dif = np.log(p_Z1s+eps_)-np.log(self.p_Z.T+eps_)  
        sTheta = (u*p_a1Z.T/(self.lmbda+eps_) - log_dif)
        gTheta  = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

class caPG(ecPG):
    name     = 'CAPG'
    p_names  = ['alpha_psi', 'alpha_rho']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 2.5), (-2, 2.5)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_names)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_names)
    n_params = len(p_names)
    voi      = ['i_SZ']
    insights = ['enc', 'dec', 'pol']
    color    = viz.r2
    marker   = 'o'
    size     = 30
    alpha    = .8

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.b          = 1/2

    def _learn_enc_dec(self):
        # get data 
        fstr, a_ava, a, r = self.mem.sample('f', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = self.embed(fstr)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        sTheta = (u*p_a1Z.T)
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

class l2PG(ecPG):
    name     = 'L2PG'
    p_names  = ['alpha_psi', 'alpha_rho', 'lmbda']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 2.5), (-2, 2.5), (-10, 1)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_bnds)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_bnds)
    poi_raw  = [(.01, 15), (.01, 15), (.01, 2)]     
    n_params = len(p_names)
    voi      = []
    insights = ['encoder', 'decoder', 'policy', 'attn', 'theta']
    marker   = 'o'
    size     = 30
    alpha    = .8
    color    = np.array([155, 153, 176]) / 255

    def _learn_enc_dec(self):
        # get data 
        fstr, a_ava, a, r = self.mem.sample('f', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = self.embed(fstr)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        l2_loss = self.theta/np.sqrt(np.square(self.theta+eps_).sum())
        sTheta = (u*p_a1Z.T)/(self.lmbda+eps_)
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T)) + l2_loss
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

class l1PG(l2PG):
    name     = 'L1PG'
    color    = np.array([154, 171, 165]) / 255

    def _learn_enc_dec(self):
        # get data 
        fstr, a_ava, a, r = self.mem.sample('f', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = self.embed(fstr)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        l1_loss = np.sign(self.theta)
        sTheta = (u*p_a1Z.T)/(self.lmbda+eps_)
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T)) + l1_loss
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

class dcPG(l2PG):
    name     = 'DCPG'
    p_names  = ['alpha_psi', 'alpha_rho', 'lmbda']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 2.5), (-2, 2.5), (-10, 1)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_bnds)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_bnds)
    poi_raw  = [(.01, 15), (.01, 15), (.01, 2)]     
    n_params = len(p_names)
    voi      = []
    insights = ['encoder', 'decoder', 'policy', 'attn', 'theta']
    marker   = 'o'
    size     = 30
    alpha    = .8
    color    = np.array([155, 150, 192]) / 255

    def _learn_pZ(self):
        self.p_Z1S = softmax(self.F@self.theta, axis=1)
        self.p_S   = np.ones([self.nS, 1]) / self.nS  # nSx1 
        self.p_Z   = self.p_Z1S.T @ self.p_S  
        p_A1Z      = softmax(self.phi, axis=1)
        self.p_A   = p_A1Z.T @ self.p_Z 

    def _learn_enc_dec(self):
        # get data 
        fstr, a_ava, a, r = self.mem.sample('f', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = self.embed(fstr)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        sTheta = u*p_a1Z.T
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        log_diff = np.log(p_A1Z+eps_) - np.log(self.p_A.T+eps_)
        sPhi = (u/(self.lmbda+eps_) - log_diff[:, [a]])*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

# ------------------------------#
#      Feature-based models     #
# ------------------------------# 

class rmPG_fea(rmPG):
    name     = 'fRMPG'
    voi      = []
    insights = ['pol']

    def _init_agent(self):
        self.phi    = np.zeros([self.nI, self.nA]) 

    def get_pol(self): fea_base.get_pol(self)

class ecPG_fea_sim(ecPG_sim):
    '''Feature efficient coding policy gradient (analytical)
    '''
    insights = ['enc', 'dec', 'pol', 'attn']
    name     = 'fECPG'

    def _init_agent(self):
        self.nZ = self.nS
        fname = f'{pth}/../data/exp2_ecpg_weight.pkl'
        with open(fname, 'rb')as handle: theta_dict = pickle.load(handle)
        self.theta = deepcopy(theta_dict[self.env.block_type])
        self.phi   = np.zeros([self.nZ, self.nA]) 
        self._learn_pZ()

    def _learn_pZ(self):
        self.p_Z1S = softmax(self.F@self.theta, axis=1)
        self.p_S = np.ones([self.nS, 1]) / self.nS 
        self.p_S = np.vstack([self.p_S, 0])
        self.p_Z   = self.p_Z1S.T @ self.p_S  
    
    def get_attn(self):
        '''Perturbation-based attention
        '''
        attn = np.zeros([self.nD])
        for d in range(self.nD): 
            attn_d = 0 
            for s in range(self.nS):
                f_orig = self.embed(self.s2f(s))
                pi_orig = softmax(f_orig@self.theta, axis=1)
                nD = f_orig.reshape([self.nD, -1]).shape[1]
                pi_pert = [] 
                for i in range(nD):
                    f_pert = f_orig.reshape([self.nD, -1]).copy()
                    f_pert[d, :] = np.eye(nD)[i, :]
                    f_pert = f_pert.reshape([1, -1])
                    pi_pert.append(softmax(f_pert@self.theta, axis=1).reshape([-1]))
                pi_pert = np.vstack(pi_pert)
                # kld for each stimuli 
                attn_d += (pi_orig* (np.log(pi_orig+eps_) - 
                                     np.log(pi_pert+eps_))).sum(1).mean()
            attn[d] = attn_d

        return attn
    
    def get_pol(self): return fea_base.get_pol(self)

class ecPG_fea(ecPG_fea_sim):
    '''Feature efficient coding policy gradient (fitting)
    '''
    def _learn_enc_dec(self): ecPG._learn_enc_dec(self)
              
class caPG_fea(caPG):
    name     = 'fCAPG'

    def _learn_pZ(self): ecPG_fea_sim._learn_pZ(self)

    def _init_agent(self): ecPG_fea_sim._init_agent(self)
     
    def get_attn(self): return ecPG_fea_sim.get_attn(self)

class l2PG_fea(l2PG):
    name     = 'fL2PG'

    def _learn_pZ(self): ecPG_fea_sim._learn_pZ(self)

    def _init_agent(self): ecPG_fea_sim._init_agent(self)
     
    def get_attn(self): return ecPG_fea_sim.get_attn(self)

class l1PG_fea(l2PG_fea):
    name     = 'fL1PG'
    color    = l1PG.color

    def _learn_enc_dec(self): l1PG._learn_enc_dec(self)

class dcPG_fea(l2PG_fea):
    name     = 'fDCPG'
    color    = dcPG.color

    def _learn_pZ(self):
        self.p_Z1S = softmax(self.F@self.theta, axis=1)
        self.p_S = np.ones([self.nS, 1]) / self.nS 
        self.p_S = np.vstack([self.p_S, 0])
        self.p_Z   = self.p_Z1S.T @ self.p_S  
        p_A1Z      = softmax(self.phi, axis=1)
        self.p_A   = p_A1Z.T @ self.p_Z 

    def _learn_enc_dec(self): dcPG._learn_enc_dec(self)

class rndPG_fea(caPG_fea):
    name     = 'RNDPG'
    p_names  = ['alpha_psi', 'alpha_rho', 'lambda']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 2.5), (-2, 2.5), (-2, 2.5)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_names)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_names)
    n_params = len(p_names)
    voi      = ['i_SZ']
    insights = ['enc', 'dec', 'pol']
    color    = viz.r2
    marker   = 'o'
    size     = 30
    alpha    = .8

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.lmbda      = params[2]
        self.n_samples  = 30
        self.b          = 1/2

    def policy(self, fstr, **kwargs):
        f = self.embed(fstr)
        # Use map to accelerate the computation of pi_ensemble
        def compute_pi(noise):
            p_Z1s = softmax(f@(self.theta+noise), axis=1)
            m_A   = mask_fn(self.nA, kwargs['a_ava'])
            p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
            pi = (p_Z1s@p_A1Z).reshape([-1])
            pi /= pi.sum()
            return pi, p_Z1s, p_A1Z
        pi_ensemble = []
        self.p_Z1s_ensemble = []
        self.p_A1Z_ensemble = []
        for _ in range(self.n_samples):
            pi, p_Z1s, p_A1Z = compute_pi(self.lmbda * np.random.randn(*self.theta.shape))
            pi_ensemble.append(pi)
            self.p_Z1s_ensemble.append(p_Z1s)
            self.p_A1Z_ensemble.append(p_A1Z)
        return np.mean(pi_ensemble, axis=0)
        
    def _learn_enc_dec(self):

        # get data 
        fstr, a, r = self.mem.sample('f', 'a', 'r')
        f = self.embed(fstr)
        
        # Use map to accelerate the loop
        def compute_grad(p_Z1s, p_A1Z):

            p_a1Z = p_A1Z[:, [a]]
            u = np.array([r - self.b])[:, np.newaxis] 
            
            # backward
            sTheta = (u*p_a1Z.T)
            gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                        sTheta - p_Z1s@sTheta.T))
            
            sPhi = u*p_Z1s.T
            gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi
            
            return gTheta, gPhi

        results = list(map(lambda x: compute_grad(*x), zip(self.p_Z1s_ensemble, self.p_A1Z_ensemble)))
        
        grad_theta = np.mean([res[0] for res in results], axis=0)
        grad_phi = np.mean([res[1] for res in results], axis=0)

        self.theta -= self.alpha_psi * grad_theta
        self.phi   -= self.alpha_rho * grad_phi


# ----------------------------------------#
#      Representation learning models     #
# ----------------------------------------# 
          
class LC(base_agent):
    '''Latenc cause model

    modified based:
    Gershman, S. J., Monfils, M. H., Norman, K. A., & Niv, Y. (2017). 
    The computational nature of memory modification. Elife, 6, e23763.

    Note, we modified the original model because
    it cannot generalize at all in the exp2 control case. 
    '''
    name     = 'LC'
    p_names  = ['eta', 'alpha', 'beta']
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 2), (-1, 2), (-1, 2)]
    p_priors = []
    p_trans  = [lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x),
                lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_)-np.log(1-x),
                lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_)]
    n_params = len(p_names)
    voi = ['last_z']
    insights = ['pol', 'p_Z1S']
    color = np.array([181, 228, 140]) / 255

    def __init__(self, env, params):
        super().__init__(env, params)

    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(LC.p_trans, params)]
        self.eta    = params[0] # the learning rate of value
        self.alpha  = params[1] # the concentration of prior 
        self.beta   = params[2] # the inverse temperature
        self.p      = .1 #params[3]
        self.tau   = 1         # the time scale parameters for CRP
        self.max_iter = 10     # maximum iteration for EM
         
    def _init_agent(self):
        self.nZ, self.t = 0, 0
        self.z = 0
        self.ZHistory = [] 
        self.fHistory = []
        self.p_f1Z = 1
        self.W_ZA = np.ones([1, self.nA])/self.nA
        self._learn_p_Z()

    def _learn_p_Z(self):
        # update prior 
        cat_zH = np.eye(self.nZ+1)[self.ZHistory]
        t = len(self.ZHistory)
        tH = np.arange(t) 
        f_z1zH = ((((1/(t-tH))**self.tau)).reshape([1, -1])@cat_zH).reshape([-1])
        f_z1zH[self.nZ] = self.alpha
        self.p_Z = f_z1zH.reshape([-1, 1]) / f_z1zH.sum()

    def policy(self, fstr, **kwargs):
        '''the forward function
        π(a|st) = softmax(βQ(st,a))
                = softmax(β\sum_z p(z,st)Q(z,a))
        '''
        # get the feature of s
        f = self.embed(fstr)
        self.fHistory.append(f.copy())
        # get p(s=f|Z) => p(s|Z): nZxnF @ nFx1 = nZx1
        p_s1f = f.reshape([1, self.nD, -1])
        self.p_s1Z = (self.p_f1Z*p_s1f).sum(2).prod(1, keepdims=True)
        # get p(r=1|Z, A) = Q(Z, A): nZxnA 
        p_r1ZA = self.W_ZA
        # p(r=1|s, A) = \sum_z p(r=1, Z|s, A)
        #             ∝ \sum_z p(r=1, Z, s|A)
        #             = \sum_z p(Z)p(s|Z)p(r=1|Z, A)
        #             = \sum_z p(Z, s)p(r=1|Z, A)
        # nZx1 * nZx1 * nZxnA = nZ*nA
        f_Zs = (self.p_Z*self.p_s1Z)
        self.p_Zs = f_Zs / (f_Zs+eps_).sum()
        Q_sA = (self.p_Zs*p_r1ZA).sum(0)
        # add mask
        m_A  = mask_fn(self.nA, kwargs['a_ava'])
        # p(a|s) = softmax(βp(r=1|s, A))
        logit = self.beta*Q_sA - (1-m_A)*max_
        self.t += 1
        return softmax(logit.reshape([-1]))
            
    def learn(self):
        self._learn_Q_ZA()
        self._learn_p_f1Z()
        self._learn_p_Z()

    def _learn_p_f1Z(self):
        cat_zH = np.eye(self.nZ+1)[self.ZHistory]
        f_f1Z = (cat_zH.T@np.vstack(self.fHistory)
            ).reshape([-1, self.nD, self.nF])+self.p
        self.p_f1Z = f_f1Z / f_f1Z.sum(2, keepdims=True) 
        assert (np.abs(self.p_f1Z.sum(axis=(1,2))-self.nD)<1e-5).all(), f'p(f|Z) does not sum to {self.nD}'

    def _learn_Q_ZA(self):
        # get data 
        a, r = self.mem.sample('a', 'r')
        # get feature
        # nZx1 
        old_w = 0 
        for _ in range(self.max_iter):
            # E-step: p(z|s, a, r) = p(Z,s)p(r|Z,a)
            #                      = p(Z,s)p(Q|Z,a)p(r|Q)
            # p(Q|Z,a) = Q(Z,a): nZ,
            q_Za = self.W_ZA[:, a]
            # p(r|Z, a) = p(Q|Z,a)p(r|Q) = Q**(r)*Q**(1-r)
            p_r1Za = q_Za**r*(1-q_Za)**(1-r)
            # p(Z|s, a, r) ∝ p(Z, s, r| a)
            #              = p(Z, s)p(r|Z, a)
            f_Z1sar = self.p_Zs.reshape([-1])*p_r1Za
            p_Z1sar = f_Z1sar / (f_Z1sar+eps_).sum()
            # M-step: W = W + ηδ
            # δ = p(z|fs, a, r)*(r - Q(s,a)): nZ,
            delta = p_Z1sar*(r - q_Za)
            # W_Za = W_Za + η*f*δ
            self.W_ZA[:, a] += self.eta*delta.reshape([-1])
            # check covergence
            if np.abs((self.W_ZA - old_w).sum()) < 1e-5: break
            # cache the old W
            old_w = self.W_ZA.copy() 
        
        # pick a z given posterior 
        self.latent_cause_policy(p_Z1sar)

    def latent_cause_policy(self, p_Zs):
        '''latent cause policy + update prior 
            p(Z|Z_{1:t-1})
        '''
        # pick the maximumize cluster
        self.z = np.argmax(p_Zs)
        self.ZHistory.append(self.z)
        # if a new cluster is selected, init
        # new set of policy
        if self.z==self.nZ: 
            new_w = np.ones([1, self.nA])/self.nA
            self.W_ZA = np.vstack([self.W_ZA, new_w])
            self.nZ += 1

    def get_last_z(self):
        return self.z
            
    def get_p_Z1S(self):
        F = self.F.reshape([-1, self.nD, self.nF])
        p_S1Z = np.zeros([self.nZ+1, F.shape[0]])
        for i in range(F.shape[0]):
            p_s1Z = (self.p_f1Z*F[[i]]).sum(2).prod(1)
            p_S1Z[:, i] = p_s1Z 
        f_SZ = p_S1Z*self.p_Z
        return f_SZ / f_SZ.sum(1, keepdims=True)

class MA(base_agent):
    '''Memory association model

    Created based on reviewers' comments
    '''
    name     = 'MA'
    p_names  = ['alpha_Q', 'alpha_assoc', 'beta_assoc', 'beta']
    p_bnds   = [(0, np.log(1000))]*len(p_names)
    p_pbnds  = [(-3, 3), (-3, 3), (-3, 10), (-3, 10)]    
    p_priors = [uniform(0, 1), uniform(0, 1),
                halfnorm(0, 40), halfnorm(0, 40)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x))]*2 +\
                [lambda x: clip_exp(x)]*2
    p_links  = [lambda x: np.log(x+eps_)-np.log(1-x+eps_)]*2+\
                [lambda x: np.log(x+eps_)]*2
    n_params = len(p_names)
    voi = []
    insights = ['pol', 'attn']
    color = np.array([153, 217, 140]) / 255
     
    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_Q     = params[0] # the value learning rate
        self.alpha_assoc = params[1] # the association learning rate
        self.beta_assoc  = params[2] # the association inverse temperature
        self.beta        = params[3] # the policy inverse temperature

    def _init_agent(self):
        self.q_FA    = np.zeros([self.nI, self.nA])  # the feature-action value 
        self.w_assoc = np.eye(self.nS+self.nProbe) # the association matrix
        self.T       = [] # memory about the trained pairs
        self.T_table = np.zeros([self.nS+self.nProbe, self.nA]) 
    
    def policy(self, fstr, **kwargs):
        f = self.embed(fstr)
        s = int(kwargs['s'])
        config = tuple([s]+kwargs['a_ava'])
        if config in self.T: # if trained
            q_hat = f@self.q_FA 
        else: # if untrained
            stimuli = list(range(self.nS+self.nProbe))
            phi = softmax([self.beta_assoc*self.w_assoc[s, i] 
                            for i in stimuli]).reshape([-1, 1])
            q   = np.vstack([self.embed(self.s2f(i))@self.q_FA 
                            for i in stimuli])
            q_hat = np.sum(phi*q, axis=0, keepdims=True)
        m_A = mask_fn(self.nA, kwargs['a_ava'])
        pi = softmax(self.beta*q_hat - (1-m_A)*max_, axis=1)
        return pi.reshape([-1]) 

    def learn(self):
        self._learn_value()
        self._learn_assoc()
        self._learn_train_table()

    def _learn_value(self):
        fstr, a, r = self.mem.sample('f', 'a', 'r')
        f = self.embed(fstr)
        q_hat = f@self.q_FA
        rpe = r - q_hat[0, a]
        # for the chosen feature & action
        f = f.reshape([-1]) 
        q_selected = self.q_FA[f>0, a] + self.alpha_Q*rpe
        # update q table 
        self.q_FA[f>0, a] = q_selected

    def _learn_assoc(self):
        k = self.F@self.F.T
        q_SA = (self.F@self.q_FA)*self.T_table
        w_tar = q_SA@q_SA.T 
        for s in range(self.nS+self.nProbe): w_tar[s, s] = 1
        self.w_assoc += self.alpha_assoc*(1+k)*(w_tar - self.w_assoc)

    def _learn_train_table(self):
        s, a_ava = self.mem.sample('s', 'a_ava')
        config = tuple([int(s)]+a_ava)
        self.T.append(config)
        self.T_table[int(s), a_ava] = 1

class MA2(MA):
    '''Memory association model

    Created based on reviewers' comments
    '''
    name     = 'MA'
    p_names  = ['alpha_Q', 'alpha_assoc', 'beta_assoc', 'beta', 's']
    p_bnds   = [(0, np.log(1000))]*len(p_names)
    p_pbnds  = [(-3, 3), (-3, 3), (-3, 10), (-3, 10), (-3, 10)]    
    p_priors = [uniform(0, 1), uniform(0, 1),
                halfnorm(0, 40), halfnorm(0, 40), halfnorm(0, 40)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x))]*2 +\
                [lambda x: clip_exp(x)]*3
    p_links  = [lambda x: np.log(x+eps_)-np.log(1-x+eps_)]*2+\
                [lambda x: np.log(x+eps_)]*3
    n_params = len(p_names)
     
    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_Q     = params[0] # the value learning rate
        self.alpha_assoc = params[1] # the association learning rate
        self.beta_assoc  = params[2] # the association inverse temperature
        self.beta        = params[3] # the policy inverse temperature
        self.s           = params[4] # the scale variable for k

    def _learn_assoc(self):
        k = self.F@self.F.T
        q_SA = (self.F@self.q_FA)*self.T_table
        w_tar = q_SA@q_SA.T 
        for s in range(self.nS+self.nProbe): w_tar[s, s] = 1
        self.w_assoc += self.alpha_assoc*(1+self.s*k)*(w_tar - self.w_assoc)

class ACL(base_agent):
    '''ACL model with value attention

    Adapted from: 
        Leong, Y. C., Radulescu, A., Daniel, R., DeWoskin, V., & Niv, Y. (2017). 
        Dynamic interaction between reinforcement learning and attention in 
        multidimensional environments. Neuron, 93(2), 451-463.
    '''
    name     = 'ACL'
    p_names  = ['eta', 'beta', 'eta_attn', 'epsilon', 'beta_attn']
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-3, 3), (-3, 10), (-3, 3), (-3, 3), (-3, 10)]    
    p_priors = [uniform(0, 1), halfnorm(0, 40), 
                uniform(0, 1), uniform(0, 1), halfnorm(0, 40)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x),]
    p_links  = [lambda x: np.log(x+eps_)-np.log(1-x+eps_),
                lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_)-np.log(1-x+eps_),
                lambda x: np.log(x+eps_)-np.log(1-x+eps_),
                lambda x: np.log(x+eps_),]
    n_params = len(p_names)
    voi = []
    insights = ['q_attn', 'phi_attn']
    color = np.array([118, 200, 147]) / 255
        
    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.eta       = params[0] # the value learning rate
        self.beta      = params[1] # the inverse temperature
        self.eta_attn  = params[2] # the learning rate of attention Q
        self.eps       = params[3] # the decay rate of attention Q
        self.beta_attn = params[4] # the inverse temperature of attention Q

    def _init_agent(self):
        self.q_choice = np.zeros([self.nI, self.nA]) # the Q for the choice model 
        self.q_attn   = np.zeros([self.nI, self.nA]) # the Q for the attn model
        self.phi      = np.ones([self.nD]) / self.nD

    def get_phi(self):
        # the maximum feature value in each dimension 
        # was then passed through a softmax function to 
        # obtain the predicted attention vector
        v = self.q_attn.max(1) 
        xi = (v.reshape([self.nD, self.nF])).max(1) 
        return softmax(self.beta_attn*xi).reshape([-1, 1])

    def policy(self, fstr, **kwargs):
        f = self.embed(fstr).reshape([self.nD, self.nF])
        self.phi = self.get_phi()
        weight_f = (self.phi*f).reshape([-1, self.nI])
        q_hat = weight_f@self.q_choice
        m_A = mask_fn(self.nA, kwargs['a_ava'])
        pi = softmax(self.beta*q_hat - (1-m_A)*max_, axis=1)
        return pi.reshape([-1]) 

    def learn(self):
        self._learn_choice_model()
        self._learn_attn_model()
        
    def _learn_choice_model(self):
        fstr, a, r = self.mem.sample('f', 'a', 'r')
        # embeding, prediction error
        f = self.embed(fstr).reshape([self.nD, self.nF])
        weight_f = (self.phi*f).reshape([-1, self.nI])
        q_hat = weight_f@self.q_choice
        rpe = r - q_hat[0, a]
        # for the chosen feature & action  
        f = f.reshape([-1])
        q_selected = self.q_choice[f>0, a] + self.eta*self.phi.reshape([-1])*rpe
        # update q table 
        self.q_choice[f>0, a] = q_selected

    def _learn_attn_model(self):
        fstr, a, r = self.mem.sample('f', 'a', 'r')
        # embeding, prediction error
        f = self.embed(fstr).reshape([self.nD, self.nF])
        weight_f = (self.phi*f).reshape([-1, self.nI])
        q_hat = weight_f@self.q_attn
        rpe = r - q_hat[0, a]
        # for the chosen feature & action  
        f = f.reshape([-1])
        q_selected = self.q_attn[f>0, a] + self.eta_attn*rpe
        # for the unchosen feature & action
        self.q_attn += self.eps*(0 - self.q_attn) 
        # update q table 
        self.q_attn[f>0, a] = q_selected

    def get_q_attn(self):
        return self.q_attn.copy()
    
    def get_phi_attn(self):
        return self.phi.copy()