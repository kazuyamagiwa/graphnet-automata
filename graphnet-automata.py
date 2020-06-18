#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community
import scipy.ndimage as nd
import collections

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
            for _ in range(300):
                self.next_state()
            return(self.seed)


#%%
for i in range(0, 512, 1):
    print(i)

    kernel_seed = f'{i:09b}'
    KERNEL = np.array(list(kernel_seed), dtype=np.uint8).reshape((3,3))

    gen = generator_state()
    gen_g1 = nx.from_numpy_matrix(gen.run())

    degree_sequence = sorted([d for n, d in gen_g1.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    if np.average(cnt) > 10:
        print('hit')
        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)       
        # draw graph in inset
        plt.axes([0.4, 0.4, 0.5, 0.5])
        Gcc = gen_g1.subgraph(sorted(nx.connected_components(gen_g1), key=len, reverse=True)[0])
        pos = nx.spring_layout(gen_g1)
        plt.axis('off')
        nx.draw_networkx_nodes(gen_g1, pos, node_size=20)
        nx.draw_networkx_edges(gen_g1, pos, alpha=0.4)
        plt.savefig('%s_degree_historgram.png' % i)
