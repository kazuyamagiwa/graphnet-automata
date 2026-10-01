"""Streamlit demo for graphnet-automata.

Evolve a seed graph with a chosen 3x3 binary kernel and inspect the result.
Keep ``steps`` modest for interactive use — full 512-kernel sweeps belong in
the CLI tools, not this UI.

Layout notes
------------
Primary controls live on the main page (mobile-friendly). Results use tabs so
phone and desktop both get a readable single-pane view instead of cramped
side-by-side columns. Advanced options sit in an expander.

After Evolve, a generation scrubber (and optional Play) walks stored
``next_state`` snapshots; a metric strip plots growth and order scores vs step.
An Advanced option can opt into a seed→mid→final triptych and short
auto-narrative (off by default).
"""

from __future__ import annotations

import collections
import time
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

from graphnet_automata import kernel_from_index
from graphnet_automata.history import (
    ENTROPY_SENTINEL,
    collect_evolution_history,
    evolution_narrative,
    graph_from_adjacency,
    lock_in_step,
    positions_for_step,
    triptych_indices,
)

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
    st.slider(
        "Edge probability",
        min_value=0.001,
        max_value=1.0,
        format="%.3f",
        key="prob",
    )
    st.number_input(
        "RNG seed", min_value=0, max_value=10_000, step=1, key="graph_seed"
    )
    st.checkbox("Directed seed graph", key="directed")
    st.checkbox("Color Louvain communities", key="show_communities")
    st.checkbox(
        "Show seed → mid → final overview",
        key="show_triptych",
        value=False,
        help="Optional three-frame summary plus a short metrics caption. Off by default.",
    )

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
4. Scrub generations to watch growth; optionally enable the seed→mid→final overview under Advanced.

There are \(2^9 = 512\) binary kernels. This demo runs **one** kernel at a time.
"""
        )


def _partition_for_graph(graph: nx.Graph) -> dict[int, int] | None:
    if graph.number_of_edges() == 0:
        return None
    try:
        import community as community_louvain

        return community_louvain.best_partition(graph)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def evolve_history(
    nodes: int,
    prob: float,
    kernel_index: int,
    steps: int,
    graph_seed: int,
    directed: bool,
    compute_communities: bool,
) -> dict[str, Any]:
    """Cached wrapper around :func:`collect_evolution_history`."""
    return collect_evolution_history(
        nodes=nodes,
        prob=prob,
        kernel_index=kernel_index,
        steps=steps,
        graph_seed=graph_seed,
        directed=directed,
        compute_communities=compute_communities,
    )


def draw_graph_figure(
    graph: nx.Graph,
    partition: dict[int, int] | None,
    pos: dict[int, tuple[float, float]],
    *,
    figsize: tuple[float, float] = (4.5, 4.5),
    node_size: int = 16,
    title: str | None = None,
) -> None:
    fig_g, ax_g = plt.subplots(figsize=figsize)
    node_color: list | str = (
        [partition[n] for n in graph.nodes()] if partition is not None else "#4C78A8"
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax_g,
        node_size=node_size,
        node_color=node_color,
        cmap=plt.cm.RdYlBu if partition is not None else None,
    )
    nx.draw_networkx_edges(graph, pos, ax=ax_g, alpha=0.35, width=0.55)
    ax_g.set_axis_off()
    if title:
        ax_g.set_title(title, fontsize=10)
    fig_g.tight_layout(pad=0.1)
    st.pyplot(fig_g, clear_figure=True, use_container_width=True)
    plt.close(fig_g)


def draw_triptych_frame(
    history: dict[str, Any],
    t: int,
    max_t: int,
    *,
    show_communities: bool,
    graph_seed: int,
    label: str,
) -> None:
    """Draw one seed/mid/final panel with the same layout rules as the scrubber."""
    adjacency = history["adjacencies"][t]
    graph = graph_from_adjacency(adjacency)
    n_nodes = history["n_nodes"][t]
    partition = _partition_for_graph(graph) if show_communities else None
    pos = positions_for_step(history["final_pos"], n_nodes, max_t, t)
    if len(pos) < n_nodes:
        pos = nx.spring_layout(graph, seed=int(graph_seed))
    draw_graph_figure(
        graph,
        partition,
        pos,
        figsize=(3.2, 3.2),
        node_size=12,
        title=f"{label} (t={t})",
    )


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


def draw_metrics_figure(
    history: dict[str, Any],
    current_t: int,
) -> None:
    steps_axis = list(range(len(history["n_nodes"])))
    lock_t = lock_in_step(history["degree_entropy"])
    ent_plot = [
        e if e < ENTROPY_SENTINEL else np.nan for e in history["degree_entropy"]
    ]

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.8), sharex=True)

    ax0 = axes[0]
    ax0.plot(steps_axis, history["n_nodes"], color="#4C78A8", label="|V|")
    ax0.plot(steps_axis, history["n_edges"], color="#F58518", label="|E|")
    ax0.set_ylabel("Count")
    ax0.legend(loc="upper left", fontsize=8, frameon=False)
    ax0.set_title("Growth")

    ax1 = axes[1]
    ax1.plot(steps_axis, ent_plot, color="#54A24B", label="Degree entropy")
    ax1.plot(
        steps_axis,
        history["avg_hist_count"],
        color="#B279A2",
        label="Mean hist. count",
    )
    communities = history["community_count"]
    if any(c is not None for c in communities):
        ax1.plot(
            steps_axis,
            [c if c is not None else np.nan for c in communities],
            color="#E45756",
            label="Communities",
            linestyle="--",
        )
    ax1.set_xlabel("Generation t")
    ax1.set_ylabel("Score")
    ax1.legend(loc="best", fontsize=8, frameon=False)
    ax1.set_title("Order")

    for ax in axes:
        ax.axvline(current_t, color="#333333", alpha=0.35, linewidth=1.2)
        if lock_t is not None:
            ax.axvline(
                lock_t,
                color="#54A24B",
                alpha=0.55,
                linewidth=1.4,
                linestyle=":",
            )

    if lock_t is not None:
        y = ent_plot[lock_t]
        axes[1].annotate(
            "structure lock-in",
            xy=(lock_t, 0.0 if np.isnan(y) else y),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            color="#54A24B",
        )

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


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
    st.session_state["gen_t"] = int(steps)
    st.session_state["playing"] = False

if "run_request" not in st.session_state:
    st.info("Choose a preset or adjust the sliders, then tap **Evolve**.")
    st.stop()

params = st.session_state["run_request"]

with st.spinner("Evolving graph…"):
    history = evolve_history(
        params["nodes"],
        params["prob"],
        params["kernel_index"],
        params["steps"],
        params["graph_seed"],
        params["directed"],
        bool(params["show_communities"]),
    )

max_t = int(history["steps"])
if "gen_t" not in st.session_state:
    st.session_state["gen_t"] = max_t

# --- Optional seed → mid → final triptych + narrative (P0.3, Advanced) ----
if st.session_state.get("show_triptych", False):
    seed_t, mid_t, final_t = triptych_indices(max_t)
    st.subheader("Evolution at a glance")
    caption = evolution_narrative(history, seed_nodes=int(params["nodes"]))
    st.write(caption)

    trip_cols = st.columns(3)
    frame_specs = (
        (seed_t, "Seed"),
        (mid_t, "Mid"),
        (final_t, "Final"),
    )
    for col, (frame_t, label) in zip(trip_cols, frame_specs):
        with col:
            draw_triptych_frame(
                history,
                frame_t,
                max_t,
                show_communities=bool(params["show_communities"]),
                graph_seed=int(params["graph_seed"]),
                label=label,
            )

    st.caption(
        "Same spring layout and Louvain coloring as the scrubber below "
        "(final embedding subset for earlier frames)."
    )

# Apply deferred playback mutations before the slider binds to gen_t.
if st.session_state.pop("_reset_gen_t", False):
    st.session_state["gen_t"] = 0
elif st.session_state.pop("_advance_gen", False):
    st.session_state["gen_t"] = min(int(st.session_state["gen_t"]) + 1, max_t)

st.session_state["gen_t"] = min(int(st.session_state["gen_t"]), max_t)

# --- Generation scrubber / playback (P0.1) ---------------------------------
scrub_l, scrub_r = st.columns((4, 1))
with scrub_l:
    st.slider(
        "Generation t",
        min_value=0,
        max_value=max_t,
        key="gen_t",
        help="t=0 is the seed; each step pads by +2 nodes then applies the CA rule.",
    )
with scrub_r:
    st.write("")  # vertical align with slider
    play_label = "Pause" if st.session_state.get("playing") else "Play"
    if st.button(play_label, use_container_width=True):
        starting = not st.session_state.get("playing", False)
        st.session_state["playing"] = starting
        if starting and int(st.session_state["gen_t"]) >= max_t:
            st.session_state["_reset_gen_t"] = True
        st.rerun()

t = int(st.session_state["gen_t"])
adjacency = history["adjacencies"][t]
graph = graph_from_adjacency(adjacency)
degrees = dict(graph.degree())
n_nodes = history["n_nodes"][t]
n_edges = history["n_edges"][t]
degree_sequence = sorted(degrees.values(), reverse=True)
degree_count = collections.Counter(degree_sequence)
avg_degree = history["avg_degree"][t]
ent_t = history["degree_entropy"][t]
avg_cnt_t = history["avg_hist_count"][t]

# Two metrics stay readable on a phone; extras go in an expander.
m1, m2 = st.columns(2)
m1.metric("Nodes", n_nodes)
m2.metric("Edges", n_edges)
with st.expander("More stats"):
    s1, s2 = st.columns(2)
    s1.metric("Avg degree", f"{avg_degree:.2f}")
    s2.metric("Matrix size", f"{adjacency.shape[0]}×{adjacency.shape[1]}")
    s3, s4 = st.columns(2)
    ent_display = "—" if ent_t >= ENTROPY_SENTINEL else f"{ent_t:.3f}"
    s3.metric("Degree entropy", ent_display)
    s4.metric("Mean hist. count", f"{avg_cnt_t:.2f}")
    communities_t = history["community_count"][t]
    if communities_t is not None:
        st.metric("Communities", communities_t)
    st.caption(
        f"t={t}/{max_t} · kernel={params['kernel_index']} · "
        f"seed nodes={params['nodes']} · p={params['prob']:.3f}"
    )
    st.caption(
        "Low degree entropy / high mean histogram count ⇒ ordered "
        "(same criteria as the search / degree CLI tools). "
        "Each step grows |V| by 2 via matrix padding."
    )

partition = None
if params["show_communities"]:
    partition = _partition_for_graph(graph)

pos = positions_for_step(history["final_pos"], n_nodes, max_t, t)
# Fallback if a node somehow lacks a mapped position.
if len(pos) < n_nodes:
    pos = nx.spring_layout(graph, seed=int(params["graph_seed"]))

# Tabs keep one clear visual at a time on phones; desktop stays comfortable too.
tab_graph, tab_hist, tab_growth = st.tabs(
    ["Evolved graph", "Degree histogram", "Growth & order"]
)

with tab_graph:
    draw_graph_figure(graph, partition, pos)
    if partition is None and params["show_communities"]:
        st.caption("Community coloring unavailable for this graph.")
    st.caption(f"Graph at generation t = {t} (seed at t = 0).")

with tab_hist:
    if degree_count:
        draw_histogram_figure(degree_count)
    else:
        st.write("No degree data.")

with tab_growth:
    draw_metrics_figure(history, current_t=t)
    st.caption(
        "Padding always grows |V|; structure shows up when degree entropy drops "
        "and histogram mass concentrates (mean hist. count rises). "
        "Dotted green line marks the largest entropy drop when present."
    )

st.caption(
    "Tip: large steps grow the matrix by 2 rows/cols each generation and slow down quickly. "
    "For full kernel sweeps use the CLI tools (`graphnet-automata-search`, etc.)."
)

# Schedule the next playback frame after the current view has rendered.
if st.session_state.get("playing"):
    if t < max_t:
        time.sleep(0.12)
        st.session_state["_advance_gen"] = True
        st.rerun()
    else:
        st.session_state["playing"] = False
