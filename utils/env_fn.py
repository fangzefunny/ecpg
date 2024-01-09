import numpy as np 
import pandas as pd 

eps_ = 1e-13
max_ = 1e+13
    
class AEtask:
    name = 'AE task'
    nS = 4
    nA = 4
    nD = 3
    nF = 15
    voi = ['a', 'acc', 'r']
    n_cont_train = 6*10
    n_cons_train = 6*10
    n_conf_train = 6*10
    block_types  = ['cont', 'cons', 'conf']

    def __init__(self, block_type='cont'):
        assert block_type in ['cont', 'cons', 'conf']
        self.block_type = block_type

    # ---------- Initialization ---------- #

    def embed(self, s):
        '''
            r1: shape
            r2: color
            r3: headdress
        '''
        f = np.zeros([3, 5])
        if self.block_type== 'cont':
            if s == 4:
                f[0, 3] = 1
                f[1, 0] = 1
                f[2, 4] = 1
            else:
                f[:, s] = 1

        elif self.block_type == 'cons':
            if s == 4:
                f[0, 3] = 1
                f[1, 0] = 1
                f[2, 4] = 1
            else:
                f[1, s//2]   = 1
                f[[0, 2], s] = 1

        elif self.block_type == 'conf':
            if s == 4:
                f[0, 3] = 1
                f[1, 0] = 1
                f[2, 4] = 1
            else:
                f[1, s%2]    = 1
                f[[0, 2], s] = 1

        return f.reshape([1, -1]) 
        
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

        # get correct Act index
        block_data['a*'] = block_data.apply(
                    lambda x: x[f'a_ava{1+int((x["flip"] - x["freqAct"])**2)}'], axis=1)

        for v in ['s', 'a_ava1', 'a_ava2', 'a*']:
            block_data[v] = block_data[v].apply(lambda x: int(x))

        # assign
        block_data['block_type'] = self.block_type

        # rename stage
        block_data['stage'] = block_data['stage'].map(
            {'ass': 'train', 'gen': 'test'}
        )
        
        return block_data

    # ---------- Interaction functions ---------- #
    @staticmethod
    def eval_fn(row, subj):
    
        # see state 
        stage  = row['stage']
        s      = int(row['s'])
        a_ava1 = row['a_ava1']
        a_ava2 = row['a_ava2']
        pi     = subj.policy(s, 
                    a_ava1=a_ava1, 
                    a_ava2=a_ava2)
        a      = int(row['a'])
        r      = row['r'] 
        ll     = np.log(pi[a]+eps_)

        # save the info and learn 
        if stage == 'train':
            subj.mem.push({
                's': s, 
                'a_ava1': a_ava1,
                'a_ava2': a_ava2,
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
        a_ava1 = row['a_ava1']
        a_ava2 = row['a_ava2']
        pi     = subj.policy(s, 
                    a_ava1=a_ava1, 
                    a_ava2=a_ava2)
        a      = int(rng.choice(AEtask.nA, p=pi)) 
        a_aster= row['a*']     
        r      = 1.*(a == a_aster)

        # save the info and learn 
        if stage == 'train':
            subj.mem.push({
                's': s, 
                'a_ava1': a_ava1,
                'a_ava2': a_ava2,
                'a': a, 
                'r': r,
            })
            subj.learn()

        return a, pi[a_aster].copy(), r
