# Held-out mutant-to-mutant edge audit

This analysis reuses the five-seed OOF predictions from the 92 S1129
single-site state panels. No model is retrained. The original training protocol
contains WT-to-single-mutant examples, whereas the evaluated native edge is
between two mutant endpoint states, $B\to C$.

The evaluation target is derived from the two assayed endpoints,

\[
y_{B\to C}^{\rm derived}=y_{WT\to C}-y_{WT\to B}.
\]

It is not an independently measured mutant-to-mutant thermodynamic step. The
experiment therefore tests compositional generalization under the benchmark
labels, not experimental validation of a physical cycle.

## Five-seed results

Metrics are calculated per outer fold, averaged across folds within each seed,
and then summarized as mean +/- sample SD across seeds 42--46. There are 9,904
ordered unseen edges per seed.

| Native edge model | PCC | Spearman | RMSE | MAE |
|:--|--:|--:|--:|--:|
| Pairwise | 0.3541 +/- 0.0290 | 0.3388 +/- 0.0182 | 2.1279 +/- 0.1019 | 1.5566 +/- 0.0678 |
| Soft consistency | 0.4424 +/- 0.0168 | 0.4020 +/- 0.0119 | 1.9098 +/- 0.0609 | 1.4042 +/- 0.0315 |
| Hard antisymmetry | 0.5022 +/- 0.0333 | 0.4544 +/- 0.0300 | 1.7829 +/- 0.0517 | 1.3337 +/- 0.0265 |
| Canonical state potential | **0.5469 +/- 0.0166** | **0.4887 +/- 0.0111** | **1.6110 +/- 0.0103** | **1.2204 +/- 0.0219** |

For the canonical state-potential model, the native $B\to C$ prediction is
the endpoint composition by construction. For the other models, replacing the
native edge with the post-hoc endpoint difference increases PCC to 0.5469,
0.5458, and 0.5265 for pairwise, soft, and hard antisymmetry, respectively.

On five-seed ensembles, a 10,000-replicate fold-stratified complex bootstrap
gives state-potential minus pairwise PCC +0.2024 (95% CI 0.1142 to 0.2957) and
RMSE -0.5081 (-1.4298 to -0.2497). Against soft consistency, the differences
are PCC +0.1080 (0.0271 to 0.2502) and RMSE -0.2815 (-0.8782 to -0.1184).
Against hard antisymmetry, the PCC interval crosses zero (+0.0486; -0.0406 to
0.1751), while the RMSE difference remains favorable (-0.1668; -0.4656 to
-0.0159).

## Reproduction

- `summarize_unseen_edges.py` joins the existing OOF paths to the endpoint
  measurements and writes per-fold, per-seed, and summary tables.
- `bootstrap_unseen_edges.py` performs the paired complex bootstrap.
- `outputs/unseen_edges/unseen_edge_summary.csv` is the primary summary.
- `outputs/unseen_edges/unseen_edge_complex_bootstrap.json` contains the
  ensemble comparisons.
