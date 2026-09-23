"""Streamlit demo for graphnet-automata.

Evolve a seed graph with a chosen 3x3 binary kernel and inspect the result.
Keep ``steps`` modest for interactive use — full 512-kernel sweeps belong in
the CLI tools, not this UI.
"""

from __future__ import annotations

import collections

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

from graphnet_automata import GeneratorState, kernel_from_index

st.set_page_config(
    page_title="graphnet-automata",
    layout="wide",
)

st.title("graphnet-automata")
st.caption(
    "Evolve a seed graph with cellular-automaton-like adjacency updates. "
    "Choose a kernel, run a modest number of steps, and inspect the result."
)

with st.sidebar:
    st.header("Seed graph")
    nodes = st.slider("Nodes", min_value=2, max_value=40, value=13)
    prob = st.slider("Edge probability", min_value=0.001, max_value=1.0, value=0.05, format="%.3f")
    graph_seed = st.number_input("RNG seed", min_value=0, max_value=10_000, value=1, step=1)
    directed = st.checkbox("Directed seed graph", value=True)

    st.header("Evolution")
    kernel_index = st.slider("Kernel index (0–511)", min_value=0, max_value=511, value=21)
    steps = st.slider("Steps", min_value=1, max_value=150, value=50)
    show_communities = st.checkbox("Color Louvain communities", value=True)

    run = st.button("Evolve", type="primary", use_container_width=True)

kernel = kernel_from_index(int(kernel_index))

col_kernel, col_info = st.columns([1, 2])
with col_kernel:
    st.subheader("Kernel")
    fig_k, ax_k = plt.subplots(figsize=(2.4, 2.4))
    ax_k.imshow(kernel, cmap="gray_r", vmin=0, vmax=1)
    ax_k.set_xticks(range(3))
    ax_k.set_yticks(range(3))
    ax_k.set_title(f"index {kernel_index} ({kernel_index:09b})")
    st.pyplot(fig_k, clear_figure=True)
    plt.close(fig_k)

with col_info:
    st.subheader("How it works")
    st.markdown(
        """
1. Build an Erdős–Rényi **seed** graph and convert it to an adjacency matrix.
2. Each step **pads** the matrix (new nodes appear around the border).
3. A 3×3 **kernel** counts local structure; birth/survival thresholds update cells.
4. Convert the final matrix back to a graph for drawing and degree stats.

There are \(2^9 = 512\) binary kernels. This demo runs **one** kernel at a time.
"""
    )

if not run:
    st.info("Adjust parameters in the sidebar, then click **Evolve**.")
    st.stop()


@st.cache_data(show_spinner=False)
def evolve_graph(
    nodes: int,
    prob: float,
    kernel_index: int,
    steps: int,
    graph_seed: int,
    directed: bool,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[int, int]]:
    kernel_local = kernel_from_index(kernel_index)
    gen = GeneratorState(
        nodes=nodes,
        prob=prob,
        kernel=kernel_local,
        steps=steps,
        graph_seed=graph_seed,
        directed=directed,
    )
    adjacency = gen.run()
    graph = nx.from_numpy_array(adjacency)
    degrees = dict(graph.degree())
    edges = list(graph.edges())
    return adjacency, edges, degrees


with st.spinner("Evolving graph…"):
    adjacency, edges, degrees = evolve_graph(
        int(nodes),
        float(prob),
        int(kernel_index),
        int(steps),
        int(graph_seed),
        bool(directed),
    )

graph = nx.Graph()
graph.add_nodes_from(range(adjacency.shape[0]))
graph.add_edges_from(edges)

n_nodes = graph.number_of_nodes()
n_edges = graph.number_of_edges()
degree_sequence = sorted(degrees.values(), reverse=True)
degree_count = collections.Counter(degree_sequence)
avg_degree = float(np.mean(degree_sequence)) if degree_sequence else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Nodes", n_nodes)
m2.metric("Edges", n_edges)
m3.metric("Avg degree", f"{avg_degree:.2f}")
m4.metric("Matrix size", f"{adjacency.shape[0]}×{adjacency.shape[1]}")

partition = None
if show_communities and n_edges > 0:
    try:
        import community as community_louvain

        partition = community_louvain.best_partition(graph)
    except Exception:
        partition = None

left, right = st.columns(2)

with left:
    st.subheader("Evolved graph")
    fig_g, ax_g = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(graph, seed=int(graph_seed))
    node_color = (
        [partition[n] for n in graph.nodes()] if partition is not None else "#4C78A8"
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax_g,
        node_size=18,
        node_color=node_color,
        cmap=plt.cm.RdYlBu if partition is not None else None,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax_g, alpha=0.35, width=0.6)
    ax_g.set_axis_off()
    st.pyplot(fig_g, clear_figure=True)
    plt.close(fig_g)
    if partition is None and show_communities:
        st.caption("Community coloring unavailable for this graph.")

with right:
    st.subheader("Degree histogram")
    if degree_count:
        deg, cnt = zip(*sorted(degree_count.items()))
        fig_h, ax_h = plt.subplots(figsize=(6, 6))
        ax_h.bar(deg, cnt, width=0.8, color="#4C78A8")
        ax_h.set_xlabel("Degree")
        ax_h.set_ylabel("Count")
        ax_h.set_title("Degree histogram")
        st.pyplot(fig_h, clear_figure=True)
        plt.close(fig_h)
    else:
        st.write("No degree data.")

st.caption(
    "Tip: large ``steps`` grow the matrix by 2 rows/cols each generation and slow down quickly. "
    "For full kernel sweeps use the CLI tools (`graphnet-automata-search`, etc.)."
)
