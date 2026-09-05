# Bioinformatics experiment completion report

Completed on 2026-09-02. Compact results and reproduction scripts are included
under `diffcormer_abbind_minexp/` in this repository.

## 1. Independent AbBind external test

- Test set: 395 mutations from 16 antibody--antigen complexes.
- Decontamination: remove an entire AbBind complex for exact PDB overlap with
  S1129 or when any chain clusters with an S1129 chain at 30% identity and 80%
  bidirectional coverage.
- AbBind labels used for training or model selection: zero.
- Primary models: five random seeds, each refitted to all 1,129 S1129 records
  after locking the architecture and hyperparameters.
- Primary seed mean ± SD: PCC 0.1197 ± 0.0232; Spearman 0.2380 ± 0.0234;
  RMSE 2.2743 ± 0.0298 kcal/mol; MAE 1.3378 ± 0.0344 kcal/mol; AUROC
  0.6391 ± 0.0180.
- Five-seed ensemble: PCC 0.1263; Spearman 0.2518; RMSE 2.2509 kcal/mol;
  MAE 1.3085 kcal/mol; AUROC 0.6431.
- Exactness outside S1129: ensemble reverse MAE 5.62e-8 kcal/mol; maximum
  absolute reverse residual 3.25e-7 kcal/mol.
- Matched-data backbone comparison: continuous-metric complex-bootstrap
  intervals all cross zero. AUROC difference is +0.0803 (95% CI
  0.0017--0.1941).
- Official-fold ensemble sensitivity: PCC 0.1310 ± 0.0236; Spearman
  0.2563 ± 0.0298; RMSE 2.2458 ± 0.0192 kcal/mol; MAE 1.2957 ± 0.0107
  kcal/mol; AUROC 0.6445 ± 0.0109.

Primary results:

- `outputs/abbind_state_external/refit_summary/external_refit_summary.json`
- `outputs/abbind_state_external/refit_summary/external_refit_seed_metrics.csv`
- `outputs/abbind_state_external/refit_summary/external_refit_ensemble_predictions.csv`

Sensitivity results and preprocessing audit:

- `outputs/abbind_state_external/evaluation/abbind_state_external_summary.json`
- `outputs/abbind_state_external/evaluation/abbind_state_seed_metrics.csv`
- `outputs/abbind_state_external/abbind_state_canonical_audit.json`
- `outputs/abbind_state_external/combined_cache_summary.json`

Large regenerable caches are kept locally under
`outputs/abbind_state_external/` and should not be included in a compact source
archive.

## 2. Retrained current-model input ablations

All ablations use official S1129 folds, seeds 42--46, and full retraining.

- Full: PCC 0.7574 ± 0.0094; RMSE 1.5415 ± 0.0230 kcal/mol.
- No state descriptors: PCC 0.7497 ± 0.0044; RMSE 1.5883 ± 0.0245.
  Complex-bootstrap performance loss: PCC 0.0145 (95% CI
  0.0053--0.0232); RMSE 0.0539 (0.0246--0.0769).
- ESM-2 + state descriptors, no ProteinMPNN: PCC 0.7515 ± 0.0072; RMSE
  1.5636 ± 0.0214. Performance loss: PCC 0.0090
  (0.0015--0.0172); RMSE 0.0329 (0.0075--0.0564).
- Persistent topology and directional atomic-delta removals are negative
  implementation controls: those historical pairwise branches are not called
  by the state-potential forward pass, and no continuous metric changed with
  statistical support.

Results:

- `outputs/state_feature_ablation_summary/state_feature_ablation_table.csv`
- `outputs/state_feature_ablation_summary/state_feature_ablation_per_seed.csv`
- `outputs/state_feature_ablation_summary/state_feature_ablation_complex_bootstrap.json`

## 3. Five-seed path and cycle audit

- Panels: 92.
- Comparisons per model and seed: 5,653 reverse edges; 9,904 ordered paths;
  31,922 triangle cycles; 133,736 four-state cycles.
- Hard antisymmetry: reverse error (1.63 ± 0.15)e-7 kcal/mol, but path error
  0.441 ± 0.038 and four-state circulation 0.348 ± 0.030 kcal/mol.
- Canonical state potential: reverse 2.20e-7, path 4.78e-7, triangle 3.08e-7,
  and four-state circulation 3.68e-7 kcal/mol (seed means).
- Every state-potential residual in every seed is below 1e-3 kcal/mol.

Results:

- `outputs/path_audit_five_seed/real_mutation_path_summary.json`
- `outputs/path_audit_five_seed/real_mutation_integrability_table.csv`
- `outputs/path_audit_five_seed/real_mutation_reverse_predictions.csv`
- `outputs/path_audit_five_seed/real_mutation_path_predictions.csv`
- `outputs/path_audit_five_seed/real_mutation_triangle_cycles.csv`
- `outputs/path_audit_five_seed/real_mutation_four_state_cycles.csv`

## 4. Reproduction scripts

- `prepare_abbind_subset.py`
- `augment_abbind_four_state.py`
- `canonicalize_external_states.py`
- `build_combined_external_state_cache.py`
- `evaluate_abbind_state_ensemble.py`
- `run_external_refit.sh`
- `summarize_external_refit.py`
- `run_missing_gpu_experiments.sh`
- `summarize_state_feature_ablations.py`

The generalized five-seed path evaluator is:

`diffcormer_abbind_minexp/evaluate_real_mutation_paths.py`

## 5. Updated paper files

- Main source: `elsarticle/kbs_antisymmetry_manuscript.tex`
- Main PDF: `elsarticle/kbs_antisymmetry_manuscript.pdf`
- Supplement source: `elsarticle/kbs_supplementary_material.tex`
- Supplement PDF: `elsarticle/kbs_supplementary_material.pdf`
- Updated five-seed Figure 2: `elsarticle/figures/fig2_integrability_audit.pdf`

## Remaining submission actions that are not experiments

The experimental gaps identified for a Bioinformatics submission are now
filled. Two publication-administration tasks remain:

1. Convert and shorten the current Elsevier/KBS manuscript to the current
   Bioinformatics article format and page/word limits.
2. Upload code and processed results to a permanent public repository and
   replace the provisional data-availability wording with stable URLs/DOIs.

No repository URL or DOI is fabricated in the manuscript.
