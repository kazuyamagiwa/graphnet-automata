#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import collections
from scipy.stats import entropy
from statistics import mean
import optuna

#import warnings; warnings.simplefilter('ignore')

#%%
KERNEL = np.zeros(9, dtype=np.uint8).reshape((3,3))
#print(KERNEL)

#%%
# use of convolution, courtesy of salt-die
class generator_state():
    def __init__(self, nodes, prob, KERNEL):
        self.nodes = nodes
        self.prob = prob
        self.KERNEL = KERNEL
        self.seed = nx.to_numpy_matrix(nx.erdos_renyi_graph(self.nodes, self.prob, seed = 1, directed=True))
    
    def next_state(self):
        seed = self.seed
        KERNEL = self.KERNEL
        seed = np.pad(seed, (1, 1), 'constant')
        neighbor_count = nd.convolve(seed, KERNEL, mode="constant")
        self.seed = np.where(((seed == 0) & (neighbor_count > 0) & (neighbor_count <= 4)) |
                                 ((seed == 1) & (neighbor_count > 4)), 1, 0)
        return(self.seed)
    
    def run(self):
        while True:
            for _ in range(300):
                self.next_state()
            return(self.seed)


#%%
def function_ga(node_num, prob_num):
    ent_list =[]
    for i in range(0, 512, 1):

        kernel_seed = f'{i:09b}'
        KERNEL_ = np.array(list(kernel_seed), dtype=np.uint8).reshape((3,3))

        gen = generator_state(node_num, prob_num, KERNEL_)
        gen_g1 = nx.from_numpy_matrix(gen.run())

        degree_sequence = sorted([d for n, d in gen_g1.degree()], reverse=True)
        degreeCount = collections.Counter(degree_sequence)

        degreeCount_init = collections.Counter(degree_sequence)    
        deg_init, cnt_init = zip(*degreeCount_init.items())
        cnt_avg_init_0 = cnt_init[-1]/sum(cnt_init)

        del degreeCount[0] # exclude graphs with zero edges
    
        if deg_init[-1] == 0 and cnt_avg_init_0 > 0.10: # exclude graphs with edge-less nodes over 10%
            ent = 100
        else:
            if len(degreeCount) > 0:
                deg, cnt = zip(*degreeCount.items())
                cnt_avg = [cnt[i]/sum(cnt) for i in range(len(cnt))]
                ent = entropy(cnt_avg)
            else:
                ent = 100            

    #    print(ent)

        if ent < 2:
            ent_list.append(ent)

    if len(ent_list) != 0:
        avg_ent = mean(ent_list)
    else:
        avg_ent = 100

    return(avg_ent)

#%%
#function_ga(33, 0.01)

# %%
def objective(trial):
    node_num = trial.suggest_int('node_num', 2, 100)
    prob_num = trial.suggest_loguniform('prob_num', 0.001, 1)
    return function_ga(node_num, prob_num)


# %%
study = optuna.create_study()

# %%
study.optimize(objective, n_trials=100)

# %%
best_para = study.best_params
best_val = study.best_value

with open('optimized.txt', 'w') as f:
    print(best_para, file = f)
    print(best_val, file = f)

f.close()

# %%

x = [trial.params['node_num'] for trial in study.trials]
y = [trial.params['prob_num'] for trial in study.trials]
z = [trial.value for trial in study.trials]

f, ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].tripcolor(x,y,z)
ax[1].tricontourf(x,y,z, 20)
ax[1].plot(x,y, 'ko ')
ax[0].plot(x,y, 'ko ')
plt.savefig('contour.png')

# %%
