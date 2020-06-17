import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community
import scipy.ndimage as nd
import collections

import warnings; warnings.simplefilter('ignore')

KERNEL = np.array([[1, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.uint8)

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
            for _ in range(300):
                self.next_state()
            return(self.seed)

gen = generator_state()
gen_g1 = nx.from_numpy_matrix(gen.run())

G = gen_g1

degree_sequence = sorted([d for n, d in gen_g1.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

plt.show()