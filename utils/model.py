import numpy as np 
import pandas as pd 
import torch
 
from functools import lru_cache
from scipy.special import softmax 
from scipy.stats import gamma, uniform, beta

from utils.fit import *
from utils.env_fn import AEtask
from utils.viz import *

eps_ = 1e-13
max_ = 1e+13

# ------------------------------#
#          Axuilliary           #
# ------------------------------#

def mask_fn(nA, a1, a2):
    return (np.eye(nA)[[a1, a2], :]).sum(0, keepdims=True)

def MI(p_X, p_Y1X, p_Y):
    return (p_X*p_Y1X*(np.log(p_Y1X+eps_)-np.log(p_Y.T+eps_))).sum()

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    return np.exp(x) 

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
                     self.agent.p_name,
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
        tot_logprior_loss = 0 if p_priors==None else \
            -self.logprior(params, p_priors)
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
        for _, row in block_data.iterrows():

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
    p_name   = []  
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
        self.load_params(params)
        self._init_embed()
        self._init_Believes()
        self._init_Buffer()
        
    def load_params(self, params): 
        return NotImplementedError
    
    def orig_params(self, params):
        return [tran(p) for p, tran in zip(params, self.p_trans)]
    
    def _init_embed(self):
        def onehot(s):
            return np.eye(self.nS)[[s], :]
        self.embed = onehot

    def _init_Buffer(self):
        self.mem = simpleBuffer()
    
    def _init_Believes(self):
        self._init_Critic()
        self._init_Actor()
        self._init_Dists()

    def _init_Critic(self):
        self.q_SA = np.ones([self.nS, self.nA]) / self.nA

    def _init_Actor(self):
        self.pi_A1S = np.ones([self.nS, self.nA]) / self.nA

    def _init_Dists(self): 
        pass

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
                pi[s, :] += self.policy(s, a_ava1=a1, a_ava2=a2)   
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
                a1, a2 = act
                pi[s, :] += self.policy(s, a_ava1=a1, a_ava2=a2)   
        return pi  

# ------------------------------#
#         Classic models        #
# ------------------------------#

@lru_cache(typed=False)
def pretrain(thershold=.99):
    '''Cached to speed up model fitting 
    '''
    env = AEtask('cont')
    nS = env.nS
    F = np.eye(nS)
    Z = np.eye(nS)*(thershold-(1-thershold)/nS) \
        + np.ones([nS, nS])*(1-thershold)/nS
    theta0 = 3.5
    i, loss_prev, lr, tol = 0, np.inf, .15, 1e-8

    while True:
        # forward
        ZHat = softmax(F*theta0, axis=1)
        loss = (-Z*np.log(ZHat+eps_)).sum(1).mean(0)
        # check convergence 
        converge = (.5*np.abs(loss - loss_prev) < tol) or (i > 600)
        if converge: break
        # backward 
        grad = F*(ZHat - Z) / F.shape[0]
        theta0 -= lr*grad.sum()
        # cache 
        loss_prev = loss
        i += 1

    # decide the encoder theta 
    theta = np.eye(nS)*theta0

    return theta 

class rmPG(base_agent):
    name     = 'RMPG'
    p_bnds   = [(np.log(eps_), np.log(50))]
    p_pbnds  = [(-2, 2)]
    p_name   = ['α']  
    p_priors = []
    p_poi    = ['α'] 
    p_trans  = [lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_)]
    n_params = len(p_name)
    insights = ['pol']
    color    = np.array([200, 200, 200]) / 255

    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_pi = params[0]
        self.b        = 1/2
    
    def _init_Believes(self):
        self._init_Actor()

    def _init_Actor(self):
        self.phi    = np.zeros([self.nS, self.nA]) 
        self.pi_A1S = softmax(self.phi, axis=1)
    
    def learn(self):
        self._learnActor()

    def _learnActor(self):
        # get data 
        s, a1, a2, a, r = self.mem.sample('s', 'a_ava1', 'a_ava2', 'a', 'r')
        
        # one-hot encode the stimuli 
        f = self.embed(s)
        
        # forward 
        m_A   = mask_fn(self.nA, a1, a2)
        p_A1s = softmax(f@self.phi-(1-m_A)*max_, axis=1)
        u_sa = r - self.b

        # backward
        gPhi = -f.T@(np.eye(self.nA)[a, :] - p_A1s)*u_sa
        self.phi -= self.alpha_pi*gPhi

        self.update_pol()

    def update_pol(self):
        pass 

    def policy(self, s, **kwargs):
        '''The forward function
        '''
        m_A   = mask_fn(self.nA, kwargs['a_ava1'], kwargs['a_ava2'])
        f = self.embed(s)
        p_A1s = softmax(f@self.phi -(1-m_A)*max_, axis=1)
        return p_A1s.reshape([-1])

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
    p_bnds   = [(np.log(eps_), np.log(50)), 
                (np.log(eps_), np.log(50)),
                (np.log(eps_), np.log(50))]
    p_pbnds  = [(-2, 2), (-2, 2), (-10, 2)]
    p_name   = ['α_ψ', 'α_ρ', 'λ']  
    p_poi    = p_name
    p_priors = []
    p_trans  = [lambda x: clip_exp(x),
                lambda x: clip_exp(x),
                lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_),]
    n_params = len(p_name)
    voi      = ['i_SZ', 'i_ZA']
    insights = ['enc', 'dec', 'pol']
    color    = viz.Red

    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.lmbda      = params[2]
        self.b          = .5

    def _init_Believes(self):
        self._init_Actor()
        self._init_Dists()

    def _init_Actor(self):
        self.nZ = self.nS 
        self.theta   = pretrain().copy()
        self.psi_Z1S = softmax(self.theta, axis=1) # nSxnZ 
        self.phi     = np.zeros([self.nS, self.nA]) 
        self.rho_A1Z = softmax(self.phi, axis=1)
    
    def _init_Dists(self):
        self.p_S = np.ones([self.nS, 1]) / self.nS  # nSx1 
        self.p_Z = self.psi_Z1S.T @ self.p_S        # nZxnS @ nSx1 
    
    def learn(self):
        self._learnActor()
        self._learnPz()

    def _learnActor(self):
        # get data 
        s, a1, a2, a, r = self.mem.sample('s', 'a_ava1', 'a_ava2', 'a', 'r')
        # one-hot encode the stimuli 
        f = self.embed(s)  
       
        # prediction 
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a1, a2)
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
        gTheta  = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

        self.update_psi_rho()
    
    def update_psi_rho(self):
        self.psi_Z1S = softmax(self.theta, axis=1)
        self.rho_A1Z = softmax(self.phi,   axis=1)
        
    def _learnPz(self):
        self.p_Z = self.psi_Z1S.T @ self.p_S

    def policy(self, s, **kwargs):
        f = self.embed(s)
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, kwargs['a_ava1'], kwargs['a_ava2'])
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_A1s = (p_Z1s@p_A1Z).reshape([-1])
        p_A1s /= p_A1s.sum()
        return p_A1s.reshape([-1])
    
    # --------- some predictions ----------- #
        
    def get_i_SZ(self):
        return MI(self.p_S, self.psi_Z1S, self.p_Z)

    def get_i_ZA(self):
        p_A = (self.p_Z.T @ self.rho_A1Z).T
        return MI(self.p_Z, self.rho_A1Z, p_A)
        
    def get_enc(self):
        return self.psi_Z1S
    
    def get_dec(self):
        acts = [[0, 1], [2, 3]]
        rho  = 0
        for act in acts:
            a1, a2 = act
            m_A   = mask_fn(self.nA, a1, a2)
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

    def _learnActor(self):
        # get data 
        s, a1, a2, a, r = self.mem.sample('s', 'a_ava1', 'a_ava2', 'a', 'r')
        # one-hot encode the stimuli 
        f = self.embed(s)  
       
        # prediction 
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a1, a2)
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

        self.update_psi_rho()

class caPG(ecPG):
    name     = 'CAPG'
    p_bnds   = [(np.log(eps_), np.log(50)), 
                (np.log(eps_), np.log(50))]
    p_pbnds  = [(-2, 2), (-2, 2)]
    p_name   = ['α_ψ', 'α_ρ']  
    p_poi    = p_name
    p_priors = []
    p_trans  = [lambda x: clip_exp(x),
                lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_)]
    n_params = len(p_name)
    voi      = ['i_SZ', 'i_ZA']
    insights = ['enc', 'dec', 'pol']
    color    = viz.r2

    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.b          = 1/2

    def _learnActor(self):
        # get data 
        s, a1, a2, a, r = self.mem.sample('s', 'a_ava1', 'a_ava2', 'a', 'r')

        # one-hot encode the stimuli 
        f = self.embed(s) 
       
        # prediction 
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a1, a2)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        sTheta = u*p_a1Z.T
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

        # update the encoder 
        self.update_psi_rho()

# ------------------------------#
#      Feature-based models     #
# ------------------------------# 

@lru_cache(typed=False)
def pretrain_fea(block_type, threshold=.99):
    '''Cached to speed up model fitting 
    '''
    env = AEtask(block_type)
    nS, nZ, nF = env.nS, env.nS, env.nF
    theta0 = (torch.ones(1)).requires_grad_()
    lr, tol = .1, 1e-10
    i, loss_prev = 0, np.inf

    # forward
    while True:
        F = torch.FloatTensor(np.vstack([env.embed(s) for s in range(nS)]))
        R = F * theta0
        Z = torch.softmax(torch.vstack([(R[r] * R).sum(1) for r in range(nZ)]), axis=1)
        acc = (Z*torch.eye(nS)).sum(1).mean()
        loss = .5*(acc - threshold).square()
        loss.backward() 
        theta0.data -= lr * theta0.grad.data
        theta0.grad.data.zero_()
        converge = (.5*np.abs(loss.data.numpy() - loss_prev) < tol) or (i > 800)
        if converge: break
        # cache 
        loss_prev = loss.data.numpy()
        i += 1

    # decide the encoder theta 
    theta = np.zeros([nS, nZ])
    F = np.vstack([env.embed(s) for s in range(nS)])
    R = F * theta0.data.numpy()[0]
    tar = softmax(np.vstack([(R[r] * R).sum(1) for r in range(nZ)]), axis=1)

    # train a theta that do this classification 
    theta = np.zeros([nF, nZ])
    lr = .21
    i, loss_prev = 0, np.inf

    while True:
        # forward
        ZHat = softmax(F@theta, axis=1)
        loss = (-tar*np.log(ZHat+eps_)).sum(1).mean(0)
        # check convergence 
        converge = (.5*np.abs(loss - loss_prev) < tol) or (i > 800)
        if converge: break
        # backward 
        grad = F.T@(ZHat - tar) / F.shape[0]
        theta -= lr*grad
        # cache 
        loss_prev = loss
        i += 1

    return theta 

class rmPG_fea(fea_base, rmPG):
    name     = 'fRMPG'
    p_bnds   = [(np.log(eps_), np.log(50))]
    p_pbnds  = [(-2, 2)]
    p_name   = ['α']  
    p_poi    = p_name
    p_priors = []
    p_trans  = [lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_)]
    n_params = len(p_name)
    voi      = []
    insights = ['pol']
    color    = viz.g

    def _init_Actor(self):
        self.nProbe = 1 
        self.F = np.vstack([self.embed(s) for s in range(self.nS+self.nProbe)]) 
        self.phi = np.zeros([self.nF, self.nA])
        self.pi_A1S = softmax(self.F@self.phi, axis=1)

    def _init_embed(self):
        self.embed = self.env.embed

    def update_pol(self):
        self.pi_A1S = softmax(self.F@self.phi, axis=1)

class ecPG_fea_sim(fea_base, ecPG_sim):
    '''Feature efficient coding policy gradient (analytical)
    '''
    name     = 'fECPG'
    p_bnds   = [(np.log(eps_), np.log(50)), 
                (np.log(eps_), np.log(50)),
                (np.log(eps_), np.log(50))]
    p_pbnds  = [(-2, 2), (-2, 2), (-10, -.15)]
    p_name   = ['α_ψ', 'α_ρ', 'λ']  
    p_poi    = p_name
    p_priors = []
    p_trans  = [lambda x: clip_exp(x),
                lambda x: clip_exp(x),
                lambda x: clip_exp(x),]
    p_links  = [lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_),]
    n_params = len(p_name)
    voi      = ['i_SZ', 'i_ZA']
    insights = ['enc', 'dec', 'pol', 'attn']
    color    = viz.Red

    def _init_embed(self):
        self.embed = self.env.embed
        
    def _init_Dists(self):
        self.p_S = np.ones([self.nS, 1]) / self.nS  # nSx1 
        self.p_S = np.vstack([self.p_S, 0])
        self.p_Z = self.psi_Z1S.T @ self.p_S        # nZxnS @ nSx1 

    def _init_Actor(self):
        self.nZ = self.nS 
        self.nProbe  = 1
        block_type   = self.env.block_type
        self.theta   = pretrain_fea(block_type).copy()
        self.F       = np.vstack([self.embed(s) for s in range(self.nS+self.nProbe)]) #nSxnF
        self.psi_Z1S = softmax(self.F@self.theta, axis=1) # nFxnZ 
        self.phi     = np.zeros([self.nS, self.nA]) 
        self.rho_A1Z = softmax(self.phi, axis=1)
    
    def update_psi_rho(self):
        self.psi_Z1S = softmax(self.F@self.theta, axis=1) # nFxnZ 
        self.rho_A1Z = softmax(self.phi, axis=1)

    def get_attn(self):
        '''Perturbation-based attention
        '''
        attn = np.zeros([3])
        for d in range(3): 
            attn_d = 0 
            for s in range(self.nS):
                f_orig = self.embed(s)
                pi_orig = softmax(f_orig@self.theta, axis=1)
                nD = f_orig.reshape([3, -1]).shape[1]
                pi_pert = [] 
                for i in range(nD):
                    f_pert = f_orig.reshape([3, -1]).copy()
                    f_pert[d, :] = np.eye(nD)[i, :]
                    f_pert = f_pert.reshape([1, -1])
                    pi_pert.append(softmax(f_pert@self.theta, axis=1).reshape([-1]))
                pi_pert = np.vstack(pi_pert)
                # kld for each stimuli 
                attn_d += (pi_orig* (np.log(pi_orig+eps_) - 
                                     np.log(pi_pert+eps_))).sum(1).mean()
            attn[d] = attn_d

        return attn

class ecPG_fea(ecPG_fea_sim):
    '''Feature efficient coding policy gradient (fitting)
    '''
    def _learnActor(self):
        # get data 
        s, a1, a2, a, r = self.mem.sample('s', 'a_ava1', 'a_ava2', 'a', 'r')
        # one-hot encode the stimuli 
        f = self.embed(s)  
       
        # prediction 
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a1, a2)
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

        self.update_psi_rho()

class caPG_fea(ecPG_fea):
    name     = 'fCAPG'
    p_bnds   = [(np.log(eps_), np.log(50)), 
                (np.log(eps_), np.log(50))]
    p_pbnds  = [(-2, 2), (-2, 2)]
    p_name   = ['α_ψ', 'α_ρ'] 
    p_poi    = p_name 
    p_priors = []
    p_trans  = [lambda x: clip_exp(x),
                lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_)]
    n_params = len(p_name)
    voi      = ['i_SZ', 'i_ZA']
    insights = ['enc', 'dec', 'pol', 'attn']
    color    = viz.r2

    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.b          = 1/2

    def _learnActor(self):
        # get data 
        s, a1, a2, a, r = self.mem.sample('s', 'a_ava1', 'a_ava2', 'a', 'r')

        # one-hot encode the stimuli 
        f = self.embed(s) 
       
        # prediction 
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a1, a2)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        sTheta = u*p_a1Z.T
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi

        # update the encoder 
        self.update_psi_rho()

# ----------------------------------------#
#      Representation learning models     #
# ----------------------------------------# 

class ACL(base_agent):
    '''ACL model with value attention

    Adapted from: 
        Leong, Y. C., Radulescu, A., Daniel, R., DeWoskin, V., & Niv, Y. (2017). 
        Dynamic interaction between reinforcement learning and attention in 
        multidimensional environments. Neuron, 93(2), 451-463.
    '''
    name     = 'ACL'
    p_bnds   = [(np.log(eps_), np.log(50)), 
                (np.log(eps_), np.log(50)),
                (np.log(eps_), np.log(50)),
                (np.log(eps_), np.log(50))]
    p_pbnds  = [(-3, 3), (-1, 3), (-3, 3), (-3, 3)]
    p_name   = ['η', 'β', 'ε', 'η_a']
    p_poi    = p_name
    p_priors = []
    p_trans  = [lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x))]
    p_links  = [lambda x: np.log(x+eps_)-np.log(1-x),
                lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_)-np.log(1-x),
                lambda x: np.log(x+eps_)-np.log(1-x)]
    n_params = len(p_name)
    voi = []
    insights = ['pol', 'attn']
    color = np.array([201, 173, 167]) / 255

    def __init__(self, env, params):
        super().__init__(env, params)
        
    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.eta   = params[0] # the value learning rate
        self.beta  = params[1] # the inverse temperature
        self.eps   = params[2] # the decay rate
        self.eta_a = params[3] # the learning rate of attention 

    def _init_Critic(self):
        self.q_SA = np.zeros([self.nF, self.nA]) / (self.nA*self.nD)
        self.q_sA = np.ones([self.nA,]) / self.nA

    def _init_Dists(self):
        self.w   = np.ones([self.nD,]) / self.nD
        self.phi = np.ones([self.nF, 1]) / self.nD

    def policy(self, s, **kwargs):
        '''the forward function'''
        # normalize the weight
        m_A  = mask_fn(self.nA, kwargs['a_ava1'], kwargs['a_ava2'])
        # get the weighted value 
        f = self.embed(s)
        self.q_sA = (self.phi.T*f)@self.q_SA
        p_A1s = softmax(self.beta*self.q_sA - (1-m_A)*max_, axis=1)
        return p_A1s.reshape([-1]) 

    def _init_embed(self):
        self.embed = self.env.embed 

    def learn(self):
        self._learn_critic()
        self._learn_attn()
        
    def _learn_critic(self):
        # get data 
        s, a, r = self.mem.sample('s', 'a', 'r')
        # embeding, prediction error
        f = self.embed(s).reshape([-1])
        delta = r - self.q_sA[0, a]
        # for the chosen feature & action  
        q_selected = self.q_SA[f>0, a] + self.eta*self.w*delta
        # for the unchosen feature & action
        self.q_SA += self.eps*(0 - self.q_SA) 
        # update q table 
        self.q_SA[f>0, a] = q_selected

    def _learn_attn(self):
        # the maximum value in each dimension was
        # then passed through a softmax function
        # to obtain the attention vector 
        v = self.q_SA.max(1)
        tar_w = softmax(v.reshape([self.nD, -1]).max(axis=-1))
        self.w += self.eta_a*(tar_w - self.w)
        # broadcast to each features 
        self.phi = np.repeat(self.w, int(self.nF/self.nD))
    
    def get_attn(self):
        return self.w.copy()
          
class LC(base_agent):
    '''Latenc cause model

    modified based:
    Gershman, S. J., Monfils, M. H., Norman, K. A., & Niv, Y. (2017). 
    The computational nature of memory modification. Elife, 6, e23763.

    Note, we modified the original model because
    it cannot generalize at all in the exp2 control case. 
    '''
    name     = 'LC'
    p_bnds   = [(np.log(eps_), np.log(50)), 
                (np.log(eps_), np.log(50)),
                (np.log(eps_), np.log(50)),
                (-50, 50),]
    p_pbnds  = [(-2, 2), (-1, 2), (-1, 2), (-2, 2)]
    p_name   = ['η', 'α', 'β', 'w']
    p_poi    = ['η', 'α', 'β']
    p_priors = []
    p_trans  = [lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x),
                lambda x: clip_exp(x),
                lambda x: x,]
    p_links  = [lambda x: np.log(x+eps_)-np.log(1-x),
                lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_),
                lambda x: x]
    n_params = len(p_name)
    voi = ['last_z']
    insights = ['pol', 'p_Z1S']
    color    =  np.array([154, 140, 152]) / 255

    def __init__(self, env, params):
        super().__init__(env, params)

    def load_params(self, params):
        # from gaussian space to actual space  
        params = [f(p) for f, p in zip(LC.p_trans, params)]
        self.eta   = params[0] # the learning rate of value
        self.alpha = params[1] # the concentration of prior 
        self.beta  = params[2] # the inverse temperature
        self.w0    = params[3] # the initial w
        self.tau   = 1         # the time scale parameters for CRP
        self.max_iter = 10     # maximum iteration for EM
         
    def _init_embed(self):
        self.embed = self.env.embed 

    def _init_Critic(self):
        self.W_ZSA = np.zeros([1, 1, self.nA])+self.w0

    def _init_Dists(self):
        self.z   = 0
        self.zH  = []
        self.nZ  = 0
        self.cat_zH = np.eye(self.nZ+1)[self.zH]
        self.p_Z = 1
        self.p_Z1sar = np.array([1])
        self.p_S1Z  = 1
        # the parameter for categorical distribution
        self.fH  = []
        self.t = 0

    def policy(self, s, **kwargs):
        '''the forward function'''
        # update deceide the class for previous stimulus 
        if self.t>0: self.class_policy()
        # get the feature of s
        f = self.embed(s)
        self.fH.append(f.copy())
        # get p(s=f|Z) => p(s|Z): nZxnF @ nFx1 = nZx1
        self.p_s1Z = (self.p_S1Z * f.reshape([1, self.nD, -1])
                      ).sum(2).prod(1, keepdims=True)
        # get p(r=1|s, Z, A) = Q(Z, s, A): nZxnA 
        q_ZsA = (f[:, :, np.newaxis]*self.W_ZSA).sum(1)
        # p(r=1|s, A) = \sum_z p(r=1, Z|s, A)
        #             ∝ \sum_z p(r=1, Z, s|A)
        #             = \sum_z p(Z)p(s|Z)p(r=1|Z, s, A)
        # nZx1 * nZx1 * nZxnA = nZ*nA
        f_Zs = (self.p_Z*self.p_s1Z)
        self.p_Zs = f_Zs / f_Zs.sum()
        q_sA = (self.p_Zs*q_ZsA).sum(0)
        # add mask
        m_A  = mask_fn(self.nA, kwargs['a_ava1'], kwargs['a_ava2'])
        # p(a|fs) = softmax(βQ(a|fs))
        logit = self.beta*q_sA - (1-m_A)*max_
        self.t += 1
        return softmax(logit.reshape([-1]))
    
    def class_policy(self):
        '''class policy + update prior 
            p(Z|Z_{1:t-1})
        '''
        # assign latent cluster 
        self.z = np.argmax(self.p_Zs.reshape([-1]))
        self.zH.append(self.z)
        # if a new cluster is selected 
        if self.z==self.nZ: 
            self.nZ += 1
            new_w = np.zeros([1, 1, self.nA])+self.w0
            self.W_ZSA = np.vstack([self.W_ZSA, new_w])
        # update prior 
        self.cat_zH = np.eye(self.nZ+1)[self.zH]
        t = len(self.zH)
        tH = np.arange(t) 
        f_z1zH = ((((1/(t-tH))**self.tau)).reshape([1, -1])@self.cat_zH).reshape([-1])
        f_z1zH[self.nZ] = self.alpha
        self.p_Z = f_z1zH.reshape([-1, 1]) / f_z1zH.sum()
        # update likelihood
        # get p(S|Z): nTxnZ.T @ nTxnF = nZxnF
        f_S1Z = (self.cat_zH.T@np.vstack(self.fH)
            ).reshape([-1, self.nD, int(self.nF/self.nD)]) + .1
        self.p_S1Z = f_S1Z / f_S1Z.sum(2, keepdims=True) 
        assert (np.abs(self.p_S1Z.sum(axis=(1,2))-3)<1e-5).all(), 'p(S|Z) does not sum to 3'
        
    def learn(self):
        self._learn_critic()

    def _learn_critic(self):
        '''Update using EM algorithm
        '''
        # get data 
        s, a, r = self.mem.sample('s', 'a', 'r')
        # get feature
        f = self.embed(s)
        # nZx1 
        old_w = 0 
        for _ in range(self.max_iter):
            # E-step: p(z|s, a, r) = p(Z, s)p(Q|Z,s,a)p(r|Q)
            # p(r=1 |Z, s, a) = Q(Z, s, a): nZ,
            q_Zsa = (f[:, :, np.newaxis]*self.W_ZSA).sum(1)[:, a]
            # p(r|Z, s, a) = Q**(r)*Q**(1-r)
            p_r1Zsa = q_Zsa**r*(1-q_Zsa)**(1-r)
            # p(Z|s, a, r) ∝ p(Z, s, r| a)
            #              = p(Z, s)p(r|Z, s, a)
            # nZ, 
            f_Z1sar = self.p_Zs.reshape([-1])*p_r1Zsa
            p_Z1sar = f_Z1sar / f_Z1sar.sum()
            # M-step: W = W + ηfδ
            # δ = p(z|fs, a, r)*(r - Q(a|fs, Z)): nZ,
            delta = p_Z1sar*(r - q_Zsa)
            # W_ZFa = W_ZFa + η*f*δ
            self.W_ZSA[:, :, a] += self.eta*delta.reshape([-1, 1])
            # check covergence
            if np.abs((self.W_ZSA - old_w).sum()) < 1e-5: break
            # cache the old W
            old_w = self.W_ZSA.copy() 
        
        # get posterior 
        self.p_Zs = p_Z1sar

    def get_last_z(self):
        return self.z
            
    def get_p_Z1S(self):
        f_SZ = self.p_S1Z*self.p_Z
        return f_SZ / f_SZ.sum(1)
