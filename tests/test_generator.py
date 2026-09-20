"""Smoke tests for the shared generator."""

import numpy as np

from graphnet_automata.generator import GeneratorState, kernel_from_index


def test_kernel_from_index_shape_and_bits() -> None:
    kernel = kernel_from_index(0b101010101)
    assert kernel.shape == (3, 3)
    assert kernel.dtype == np.uint8
    assert int("".join(str(int(x)) for x in kernel.ravel()), 2) == 0b101010101


def test_generator_run_grows_matrix() -> None:
    kernel = kernel_from_index(1)
    gen = GeneratorState(nodes=5, prob=0.2, kernel=kernel, steps=3, graph_seed=0)
    initial_shape = gen.seed.shape
    result = gen.run()
    assert result.ndim == 2
    assert result.shape[0] == result.shape[1]
    # Each step pads by 1 on each side, so size grows by 2 per step.
    assert result.shape[0] == initial_shape[0] + 2 * 3
