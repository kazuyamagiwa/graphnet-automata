# Benchmark results — Bayesian CA vs gCastle

Final **N=100** figures on **Colab Tesla T4** (tuned GOLEM budget). Reproducible via:

```bash
pip install -U "jax[cuda13]" gcastle torch
python -m graphnet_automata.benchmark --n 100
```

## Setup

| Item | Value |
| --- | --- |
| Nodes / seed / samples | N=100, seed=0, 20 000 samples |
| True edges | 125 (SEM target band 100–150) |
| PC | `variant=stable`, `alpha=0.05`, `ci_test=fisherz` |
| GOLEM | see kwargs below (`num_iter=20000`) |
| Bayesian CA | JAX `@jit` `step`, 100 ticks, `k=3` |
| Device | PyTorch CUDA + JAX GPU (Tesla T4) |

## Device checks (Colab T4)

Both accelerators on GPU for this table (exact Colab printout may vary by JAX/Torch build):

- `torch.cuda.is_available(): True` → GOLEM `device_type='gpu'`
- JAX default backend / devices → GPU (T4)

## GOLEM kwargs (tuned)

```text
lambda_1=0.02 lambda_2=5.0 equal_variances=True learning_rate=0.001
num_iter=20000 checkpoint_iter=5000 seed=1 graph_thres=0.3
device_type=gpu early_stop=absent(gcastle)
```

gCastle’s GOLEM has **no** early-stop / patience / tol — only a fixed Adam `num_iter` plus `checkpoint_iter` logging.

## Results (final)

| Algorithm | Skeleton F1 | Directed F1 | Wall Time (s) |
| --- | ---: | ---: | ---: |
| PC (gcastle) | 0.968 | 0.813 | 528.631 |
| GOLEM (gcastle) | 0.577 | 0.403 | 251.182 |
| Bayesian CA (JAX) | 0.812 | 0.483 | 0.323 |

### Headline

- **Speed:** Bayesian CA is ~**778×** faster than GOLEM and ~**1636×** faster than PC (0.323 s vs 251.182 s / 528.631 s).
- **vs GOLEM accuracy:** skeleton F1 **0.812 vs 0.577**; directed F1 **0.483 vs 0.403**.
- **Recovery:** CA skeleton TP **121 / 125** true edges.

PC remains the strongest directed/skeleton F1 baseline; CA dominates the speed/accuracy tradeoff against the GPU GOLEM run under this honest budget.

### Footnote — prior untuned GOLEM (100k iters)

Same SEM / T4 host, `num_iter=100000` (before the 20k retune): GOLEM wall time **~1251 s**, skeleton F1 ~0.569, directed F1 ~0.395. Loss had already plateaued around ~15k iterations, so the 20k budget is the defensible GPU race setting.

## Related PRs

| PR | Topic | Status |
| --- | --- | --- |
| [#17](https://github.com/kazuyamagiwa/graphnet-automata/pull/17) | Phase 1–6 housekeeping | merged |
| [#18](https://github.com/kazuyamagiwa/graphnet-automata/pull/18) | Initial gCastle PC / NOTEARS bench | merged |
| [#19](https://github.com/kazuyamagiwa/graphnet-automata/pull/19) | N=100 + device checks | merged |
| [#20](https://github.com/kazuyamagiwa/graphnet-automata/pull/20) | NOTEARS → GOLEM | merged |
| [#21](https://github.com/kazuyamagiwa/graphnet-automata/pull/21) | GOLEM `num_iter=20000` | merged |
