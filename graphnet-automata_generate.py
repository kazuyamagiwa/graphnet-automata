#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import collections
import h5py
from statistics import mean

import warnings; warnings.simplefilter('ignore')

#%%
KERNEL = np.zeros(9, dtype=np.uint8).reshape((3,3))
#print(KERNEL)

#%%
# use of convolution, courtesy of salt-die
class generator_state:
    seed = nx.to_numpy_matrix(nx.erdos_renyi_graph(13, 0.05, seed = 1, directed=True))
    
    def next_state(self):
        seed = self.seed
        seed = np.pad(seed, (1, 1), 'constant')
        neighbor_count = nd.convolve(seed, KERNEL, mode="constant")
        self.seed = np.where(((seed == 0) & (neighbor_count > 0) & (neighbor_count <= 4)) |
                                 ((seed == 1) & (neighbor_count > 4)), 1, 0)
        return(self.seed)
    
    def run(self):
        while True:
            for _ in range(100):
                self.next_state()
            return(self.seed)

#%%
def adj2avgcnt(adj):
    """
    calculate degree count average from adjacency matrix 
    """
    gen = nx.from_numpy_matrix(adj)
    degree_sequence = sorted([d for n, d in gen.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cnt_avg = mean(cnt)
#    print(cnt_avg)
    return(cnt_avg)

#%%

output_file = 'GA_seed_13_0.05_50.h5'
h5file = h5py.File(output_file, 'w')

for i in range(0, 512, 1):
    print(i)

    kernel_seed = f'{i:09b}'
    KERNEL = np.array(list(kernel_seed), dtype=np.uint8).reshape((3,3))

    gen = generator_state()
    adj_g1 = gen.run()

    cnt_avg_g1 = adj2avgcnt(adj_g1)
    print(cnt_avg_g1)

    dir = 'seed_'+str(np.int(i))
    h5file.create_group(dir)
    h5file.create_dataset(dir+'/adjacency_matrix',data= adj_g1, compression='gzip', compression_opts=9)
    h5file.create_dataset(dir+'/average_count',data= cnt_avg_g1)
    h5file.flush()

h5file.flush()
h5file.close()

# %%
input_file = 'GA_seed_13_0.05_50.h5'
h5file = h5py.File(input_file,"r")

folder= 'seed_'  + str(448)
adj1  = h5file[folder+"/adjacency_matrix"].value

g1 = nx.from_numpy_matrix(adj1)
nx.draw(g1, node_size=10, alpha=0.5)
#plt.savefig('torus.png')
#plt.savefig('disordered.png')
plt.show()

# %%
h5file.close()
# %%
