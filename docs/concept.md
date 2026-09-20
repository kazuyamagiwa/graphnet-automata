# Concept

## Premise

Start with an undirected (or directed) seed graph of a few nodes. Convert it
to an adjacency matrix, then repeatedly:

1. **Pad** the matrix by one layer on every side (two new nodes appear).
2. **Convolve** with a 3×3 binary kernel to count local structure.
3. **Update** each cell with birth / survival thresholds similar to a
   cellular automaton.
4. Convert the final matrix back into a graph for visualization and scoring.

Community detection and degree histograms are used to characterize the
structures that emerge.

## Kernels

There are \(2^9 = 512\) possible binary 3×3 kernels. Most tools in this
package sweep that full range. Helper
{func}`graphnet_automata.generator.kernel_from_index` maps an integer
`0..511` onto a kernel matrix.

## Update rule

After padding and convolution, each cell is set to `1` when:

* it was `0` and the neighbor count is in `1..4` (birth), or
* it was `1` and the neighbor count is greater than `4` (survival).

Otherwise the cell becomes `0`.

## Scoring ideas used in the tools

| Tool | Idea |
| --- | --- |
| Degree search | High average degree-histogram bin count |
| Entropy search | Low Shannon entropy of non-zero degrees |
| Generate | Store every kernel's adjacency matrix + average count |
| Optimize | Optuna over seed `nodes` / `prob` to minimize mean low entropy |
