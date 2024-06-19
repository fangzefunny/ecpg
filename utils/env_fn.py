import numpy as np 
import pandas as pd 

eps_ = 1e-13
max_ = 1e+13
    
class exp1_task:
    name = 'AE task basic'
    stimuli = [0, 1, 2, 3]
    nS = len(stimuli)
    nA = 4
    nProbe = 0
    nD = nS
    nF = 1
    nI = 4
    dims = [1, 1, 1, 1]
    dim_name = ['d1', 'd2', 'd3', 'd4'] 
    voi = ['a', 'acc', 'r']
    n_cont_train = 6*10
    block_types = ['cont', 'cons', 'conf']

    def __init__(self, block_type='cont'):
        self.block_type = block_type

    @staticmethod
    def embed(fstr):
        o = [int(f) for f in fstr]
        return np.hstack(o).reshape([1, -1])
    
    @staticmethod
    def eval_fn(row, subj):
    
        # see state 
        stage  = row['stage']
        s      = int(row['s'])
        f      = row['f']
        a_ava1 = row['a_ava1']
        a_ava2 = row['a_ava2']
        pi     = subj.policy(f, s=s,
                    a_ava=[a_ava1, a_ava2])
        a      = int(row['a'])
        r      = row['r'] 
        ll     = np.log(pi[a]+eps_)

        # save the info and learn 
        if stage == 'train':
            subj.mem.push({
                's': s, 
                'f': f,
                'a_ava': [a_ava1, a_ava2],
                'a': a, 
                'r': r,
            })
            subj.learn()

        return ll
    
    @staticmethod
    def sim_fn(row, subj, rng):
        # ---------- Stage 1 ----------- #

        # see state 
        stage  = row['stage']
        s      = int(row['s'])
        f      = row['f']
        a_ava1 = row['a_ava1']
        a_ava2 = row['a_ava2']
        pi     = subj.policy(f, s=s,
                    a_ava=[a_ava1, a_ava2])
        a      = int(rng.choice(exp1_task.nA, p=pi)) 
        cor_a  = row['cor_a']     
        r      = 1.*(a==cor_a)

        # save the info and learn 
        if stage == 'train':
            subj.mem.push({
                's': s, 
                'f': f,
                'a_ava': [a_ava1, a_ava2],
                'a': a, 
                'r': r,
            })
            subj.learn()

        return a, pi[cor_a].copy(), r

    def s2f(self, s):
        lst = np.eye(self.nS)[int(s)]
        return ''.join([str(int(i)) for i in lst])

    def instan(self, seed=1234):
        '''Instantiate the environment
        '''

        rng = np.random.default_rng(seed)
        nass = 10
        ngen  = 6
        train = [[0, 0, 1, 0, 0], [2, 0, 1, 1, 0],
                 [0, 2, 3, 0, 0], [3, 0, 1, 1, 0],
                 [1, 2, 3, 0, 0], [2, 2, 3, 1, 0],]
        test  = [[0, 0, 1, 0, 0], [2, 0, 1, 1, 0],
                 [1, 0, 1, 0, 1], [3, 0, 1, 1, 0],
                 [0, 2, 3, 0, 0], [2, 2, 3, 1, 0],
                 [1, 2, 3, 0, 0], [3, 2, 3, 1, 1],]
        flrate = 0
        
        # create miniblock of the trial
        ass_data = np.vstack(train) 
        gen_data = np.vstack(test)
        nflip    = int(10*flrate)
        ass_flip = []
        for _ in range(len(train)):
            lst = [1]*nflip + [0]*(nass-nflip)
            rng.shuffle(lst)
            ass_flip.append(lst.copy())
        ass_flip = np.vstack(ass_flip)
        gen_flip = np.zeros([len(test), ngen])

        # repeat and shuffle the miniblock
        block_data = {}
        tps = []
        for sta in ['ass', 'gen']:
            ind   = list(range(eval(f'{sta}_data').shape[0]))
            data = []
            for t in range(eval(f'n{sta}')):
                # shuffle the index
                rng.shuffle(ind)
                data.append(np.hstack([eval(f'{sta}_data')[ind, :], 
                        eval(f'{sta}_flip')[ind, t].reshape([-1, 1])]))
                tps.append([t]*len(ind))
            block_data[sta] = np.vstack(data)
        
        block = {
            's':         np.hstack([block_data['ass'][:, 0], block_data['gen'][:, 0]]),
            'a_ava1':    np.hstack([block_data['ass'][:, 1], block_data['gen'][:, 1]]),
            'a_ava2':    np.hstack([block_data['ass'][:, 2], block_data['gen'][:, 2]]),
            'freqAct':   np.hstack([block_data['ass'][:, 3], block_data['gen'][:, 3]]), 
            'untrained': np.hstack([block_data['ass'][:, 4], block_data['gen'][:, 4]]), 
            'flip'  :    np.hstack([block_data['ass'][:, 5], block_data['gen'][:, 5]]), 
            'tps'   :    np.hstack(tps),
            'trial':     np.hstack([np.arange(ass_data.shape[0]*nass), np.arange(gen_data.shape[0]*ngen)]),
            'stage':     ['ass']*nass*ass_data.shape[0] + ['gen']*ngen*gen_data.shape[0]
        }

        block_data = pd.DataFrame.from_dict(block)

        # get feature
        block_data['f'] = block_data['s'].apply(self.s2f)
        block_data['group'] = block_data['untrained'].apply(
            lambda x: 'untrained' if x else 'trained'
        )

        # get correct Act index
        block_data['cor_a'] = block_data.apply(
                    lambda x: x[f'a_ava{1+int((x["flip"] - x["freqAct"])**2)}'], axis=1)

        for v in ['s', 'a_ava1', 'a_ava2', 'cor_a']:
            block_data[v] = block_data[v].apply(lambda x: int(x))

        # assign
        block_data['block_type'] = self.block_type

        # rename stage
        block_data['stage'] = block_data['stage'].map(
            {'ass': 'train', 'gen': 'test'}
        )
        
        return block_data
    
class exp2_task(exp1_task):
    name = 'AE task feature'
    stimuli = [0, 1, 2, 3, 4]
    nS = 4
    nProbe = 1
    nA = 4
    nD = 3
    nF = 5
    nI = nD*nF
    dims = [5, 5, 5]
    dim_name = ['shape', 'color', 'appendage'] 
    voi = ['a', 'acc', 'r']
    n_cons_train = 6*10
    n_cont_train = 6*10
    n_conf_train = 6*10
    block_types = ['cont', 'cons', 'conf']

    def __init__(self, block_type='cont'):
        self.block_type = block_type

    @staticmethod
    def embed(fstr):
        o = [np.eye(exp2_task.dims[i])[int(f)] 
                for i, f in enumerate(fstr)]
        return np.hstack(o).reshape([1, -1])
    
    def s2f(self, s):
        '''
            r1: shape
            r2: color
            r3: appendage
        '''
        s = int(s)
        if self.block_type== 'cont':
            f = '304' if s==4 else ''.join([f'{s}']*3) 

        elif self.block_type == 'cons':
            f = '304' if s==4 else f'{s}{s//2}{s}'

        elif self.block_type == 'conf':
            f = '304' if s==4 else f'{s}{s%2}{s}'

        return f
