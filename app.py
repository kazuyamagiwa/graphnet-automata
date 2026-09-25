"""Streamlit demo for graphnet-automata.

Evolve a seed graph with a chosen 3x3 binary kernel and inspect the result.
Keep ``steps`` modest for interactive use — full 512-kernel sweeps belong in
the CLI tools, not this UI.

Layout notes
------------
Primary controls live on the main page (mobile-friendly). Results use tabs so
phone and desktop both get a readable single-pane view instead of cramped
side-by-side columns. Advanced options sit in an expander.
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
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Tighten padding and improve tap targets on narrow screens without
# breaking the centered desktop reading width.
st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 52rem;
      }
      div.stButton > button {
        min-height: 2.75rem;
        font-weight: 600;
      }
      @media (max-width: 640px) {
        .block-container {
          padding-left: 0.85rem;
          padding-right: 0.85rem;
        }
        h1 { font-size: 1.6rem !important; }
        /* Stack Streamlit columns on phones so plots aren't squeezed. */
        div[data-testid="stHorizontalBlock"] {
          flex-wrap: wrap;
        }
        div[data-testid="stHorizontalBlock"] > div {
          min-width: min(100%, 18rem) !important;
          flex: 1 1 100% !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS: dict[str, dict[str, float | int | bool]] = {
    "Quick demo": {
        "nodes": 8,
        "prob": 0.15,
        "kernel_index": 21,
        "steps": 25,
        "graph_seed": 1,
        "directed": True,
        "show_communities": True,
    },
    "Classic seed (13 / 0.05)": {
        "nodes": 13,
        "prob": 0.05,
        "kernel_index": 21,
        "steps": 50,
        "graph_seed": 1,
        "directed": True,
        "show_communities": True,
    },
    "Ordered-looking kernel": {
        "nodes": 13,
        "prob": 0.05,
        "kernel_index": 448,
        "steps": 60,
        "graph_seed": 1,
        "directed": True,
        "show_communities": True,
    },
}

st.title("graphnet-automata")
st.caption(
    "Evolve a seed graph with cellular-automaton-like adjacency updates. "
    "Pick a preset or tweak the sliders, then tap Evolve."
)

# --- Primary controls (always visible; no sidebar required) ---------------
preset_name = st.selectbox("Preset", list(PRESETS), index=1)
if (
    "active_preset" not in st.session_state
    or st.session_state["active_preset"] != preset_name
):
    st.session_state["active_preset"] = preset_name
    for key, value in PRESETS[preset_name].items():
        st.session_state[key] = value

nodes = st.slider("Nodes", min_value=2, max_value=40, key="nodes")
kernel_index = st.slider("Kernel index (0–511)", min_value=0, max_value=511, key="kernel_index")
steps = st.slider("Steps", min_value=1, max_value=150, key="steps")

run = st.button("Evolve", type="primary", use_container_width=True)

with st.expander("Advanced options", expanded=False):
    prob = st.slider(
        "Edge probability",
        min_value=0.001,
        max_value=1.0,
        format="%.3f",
        key="prob",
    )
    graph_seed = st.number_input(
        "RNG seed", min_value=0, max_value=10_000, step=1, key="graph_seed"
    )
    directed = st.checkbox("Directed seed graph", key="directed")
    show_communities = st.checkbox("Color Louvain communities", key="show_communities")

with st.expander("How it works", expanded=False):
    kcol, tcol = st.columns((1, 2))
    kernel = kernel_from_index(int(kernel_index))
    with kcol:
        fig_k, ax_k = plt.subplots(figsize=(2.2, 2.2))
        ax_k.imshow(kernel, cmap="gray_r", vmin=0, vmax=1)
        ax_k.set_xticks(range(3))
        ax_k.set_yticks(range(3))
        ax_k.set_title(f"{kernel_index} ({kernel_index:09b})", fontsize=9)
        st.pyplot(fig_k, clear_figure=True, use_container_width=True)
        plt.close(fig_k)
    with tcol:
        st.markdown(
            r"""
1. Build an Erdős–Rényi **seed** graph → adjacency matrix.
2. Each step **pads** the matrix (new nodes appear around the border).
3. A 3×3 **kernel** counts local structure; birth/survival rules update cells.
4. Convert the final matrix back to a graph for drawing and degree stats.

There are \(2^9 = 512\) binary kernels. This demo runs **one** kernel at a time.
"""
        )


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


def draw_graph_figure(
    graph: nx.Graph,
    partition: dict[int, int] | None,
    graph_seed: int,
) -> None:
    fig_g, ax_g = plt.subplots(figsize=(4.5, 4.5))
    pos = nx.spring_layout(graph, seed=int(graph_seed))
    node_color: list | str = (
        [partition[n] for n in graph.nodes()] if partition is not None else "#4C78A8"
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax_g,
        node_size=16,
        node_color=node_color,
        cmap=plt.cm.RdYlBu if partition is not None else None,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax_g, alpha=0.35, width=0.55)
    ax_g.set_axis_off()
    fig_g.tight_layout(pad=0.1)
    st.pyplot(fig_g, clear_figure=True, use_container_width=True)
    plt.close(fig_g)


def draw_histogram_figure(degree_count: collections.Counter) -> None:
    deg, cnt = zip(*sorted(degree_count.items()))
    fig_h, ax_h = plt.subplots(figsize=(4.5, 3.6))
    ax_h.bar(deg, cnt, width=0.8, color="#4C78A8")
    ax_h.set_xlabel("Degree")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Degree histogram")
    fig_h.tight_layout()
    st.pyplot(fig_h, clear_figure=True, use_container_width=True)
    plt.close(fig_h)


if run:
    st.session_state["run_request"] = {
        "nodes": int(nodes),
        "prob": float(st.session_state["prob"]),
        "kernel_index": int(kernel_index),
        "steps": int(steps),
        "graph_seed": int(st.session_state["graph_seed"]),
        "directed": bool(st.session_state["directed"]),
        "show_communities": bool(st.session_state["show_communities"]),
    }

if "run_request" not in st.session_state:
    st.info("Choose a preset or adjust the sliders, then tap **Evolve**.")
    st.stop()

params = st.session_state["run_request"]

with st.spinner("Evolving graph…"):
    adjacency, edges, degrees = evolve_graph(
        params["nodes"],
        params["prob"],
        params["kernel_index"],
        params["steps"],
        params["graph_seed"],
        params["directed"],
    )

graph = nx.Graph()
graph.add_nodes_from(range(adjacency.shape[0]))
graph.add_edges_from(edges)

n_nodes = graph.number_of_nodes()
n_edges = graph.number_of_edges()
degree_sequence = sorted(degrees.values(), reverse=True)
degree_count = collections.Counter(degree_sequence)
avg_degree = float(np.mean(degree_sequence)) if degree_sequence else 0.0

# Two metrics stay readable on a phone; extras go in an expander.
m1, m2 = st.columns(2)
m1.metric("Nodes", n_nodes)
m2.metric("Edges", n_edges)
with st.expander("More stats"):
    s1, s2 = st.columns(2)
    s1.metric("Avg degree", f"{avg_degree:.2f}")
    s2.metric("Matrix size", f"{adjacency.shape[0]}×{adjacency.shape[1]}")
    st.caption(
        f"kernel={params['kernel_index']} · steps={params['steps']} · "
        f"seed nodes={params['nodes']} · p={params['prob']:.3f}"
    )

partition = None
if params["show_communities"] and n_edges > 0:
    try:
        import community as community_louvain

        partition = community_louvain.best_partition(graph)
    except Exception:
        partition = None

# Tabs keep one clear visual at a time on phones; desktop stays comfortable too.
tab_graph, tab_hist = st.tabs(["Evolved graph", "Degree histogram"])

with tab_graph:
    draw_graph_figure(graph, partition, params["graph_seed"])
    if partition is None and params["show_communities"]:
        st.caption("Community coloring unavailable for this graph.")

with tab_hist:
    if degree_count:
        draw_histogram_figure(degree_count)
    else:
        st.write("No degree data.")

st.caption(
    "Tip: large steps grow the matrix by 2 rows/cols each generation and slow down quickly. "
    "For full kernel sweeps use the CLI tools (`graphnet-automata-search`, etc.)."
)
