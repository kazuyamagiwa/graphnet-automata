"""Optimize seed-graph parameters with Optuna.

Instead of fixing ``nodes`` and ``prob``, this tool asks Optuna to minimize
the *average* low-entropy score across kernels. For each trial:

1. Sample ``node_num`` and ``prob_num``.
2. Evolve all 512 kernels with those seed settings.
3. Collect entropy values below ``2`` and average them.
4. If nothing qualifies, return a large penalty (``100``).

The study writes best parameters to a text file and a contour plot of the
sampled search surface.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import networkx as nx
import optuna

from graphnet_automata.generator import GeneratorState, kernel_from_index
from graphnet_automata.search import degree_entropy


def function_ga(node_num: int, prob_num: float, steps: int = 300) -> float:
    """Objective helper: mean low-entropy score over the kernel sweep.

    Parameters
    ----------
    node_num:
        Seed graph size.
    prob_num:
        Seed graph edge probability.
    steps:
        Evolution steps per kernel.

    Returns
    -------
    float
        Mean entropy among kernels that scored ``< 2``, or ``100.0`` if none
        did.
    """
    ent_list: list[float] = []
    for i in range(512):
        kernel = kernel_from_index(i)
        gen = GeneratorState(
            nodes=node_num, prob=prob_num, kernel=kernel, steps=steps
        )
        gen_g1 = nx.from_numpy_array(gen.run())
        ent, _, _ = degree_entropy(gen_g1)
        if ent < 2:
            ent_list.append(ent)

    return mean(ent_list) if ent_list else 100.0


def main(
    n_trials: int = 100,
    results_file: str | Path = "optimized.txt",
    contour_file: str | Path = "contour.png",
) -> None:
    """Run the Optuna study and write results plus a contour figure.

    Parameters
    ----------
    n_trials:
        Number of Optuna trials. Each trial itself sweeps 512 kernels, so this
        can be slow.
    results_file:
        Text file receiving ``best_params`` and ``best_value``.
    contour_file:
        PNG path for a tripcolor / contour view of the sampled surface.
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: minimize average low degree-entropy."""
        node_num = trial.suggest_int("node_num", 2, 100)
        # Log-uniform prior matches the original suggest_loguniform usage.
        prob_num = trial.suggest_float("prob_num", 0.001, 1.0, log=True)
        return function_ga(node_num, prob_num)

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    results_file = Path(results_file)
    with results_file.open("w", encoding="utf-8") as handle:
        print(study.best_params, file=handle)
        print(study.best_value, file=handle)

    # Visualize the sampled (node_num, prob_num) landscape.
    x = [trial.params["node_num"] for trial in study.trials]
    y = [trial.params["prob_num"] for trial in study.trials]
    z = [trial.value for trial in study.trials]

    _, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].tripcolor(x, y, z)
    ax[1].tricontourf(x, y, z, 20)
    ax[0].plot(x, y, "ko ")
    ax[1].plot(x, y, "ko ")
    plt.savefig(contour_file)
    plt.close()


if __name__ == "__main__":
    main()
