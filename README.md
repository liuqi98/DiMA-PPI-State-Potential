# DiMA-PPI State-Potential Reproducibility Bundle

Compact research artifact for a shared four-state potential model of
protein--protein interaction mutation effects. The formulation predicts
thermodynamic differences from one scalar state function, enforcing reverse
mutation antisymmetry and path additivity by construction.

This repository packages the manuscript and supplement, model and analysis
code, final compact result tables, out-of-fold predictions, strict-partition
bootstrap analyses, and the September 2026 Bioinformatics completion audit.
Large pretrained weights, raw structures, third-party datasets, graph caches,
and regenerated checkpoints are intentionally excluded.

## October 2026 manuscript update

- [Current editable Word manuscript](iscas2026_dima/dima_iscas2026_full.docx)
- [Current IEEE conference PDF](iscas2026_dima/dima_iscas2026_full.pdf)
- [Paper sources, figures, dataset and build instructions](iscas2026_dima/README.md)
- [Download the complete manuscript source archive](iscas2026_dima/dima_iscas2026_full_source.zip)

This update adds the conference manuscript, the user-supplied model figure in
Word, the PPI-Path92 label tables, archived evidence, and additional canonical-reuse
and unseen-edge analysis scripts. Existing model-training code and journal
artifacts are retained. Word and the IEEE PDF use different Figure 1 artwork,
as documented in the conference directory.

## Repository layout

- `iscas2026_dima/` — current conference manuscript, Word, figure code and PPI-Path92
- `elsarticle/` — manuscript, supplement, bibliography, figures, and figure code
- `diffcormer_abbind_minexp/` — training, evaluation, and audit entry points
- `diffcormer_abbind_minexp/outputs/` — compact reported results
- `bioinformatics_completion/` — final external-test and ablation report

## Read the artifact

- [Main manuscript](elsarticle/kbs_antisymmetry_manuscript.pdf)
- [Supplementary material](elsarticle/kbs_supplementary_material.pdf)
- [Reproducibility notes](elsarticle/reproducibility_README.md)
- [Experiment completion report](bioinformatics_completion/EXPERIMENT_COMPLETION_REPORT.md)

## Environment

The analysis stack requires Python 3.10+, PyTorch, PyTorch Geometric,
`torch-scatter`, NumPy, pandas, SciPy, and scikit-learn. Install versions that
match your CUDA/PyTorch platform; the model code also expects a local ESM-2
checkpoint.

The provided shell runners no longer contain machine-specific paths. Configure
the external resources at runtime:

```bash
export PYTHON_BIN=/path/to/python
export ESM_MODEL_PATH=/path/to/esm2_t33_650M_UR50D.pt
```

Prepared CSV files and graph caches default to paths under
`diffcormer_abbind_minexp/outputs/`. They are not included in this compact
artifact and must be regenerated from the public datasets described in the
manuscript or supplied explicitly with the runner environment variables.

## Included results

The bundle contains the five-seed S1129 state-potential comparisons, strict
sequence/interface split analyses, path/cycle audits, a 395-mutation
sequence-decontaminated AbBind external test, and state-feature ablations.
Reported metrics and their interpretation are frozen in the included PDF and
completion report.

## Reproducibility boundary

Bundled tables and figures can be inspected without model weights. Full model
retraining requires the excluded public-source datasets, ESM-2 weights, and
large regenerable graph caches. This repository is an archival research
artifact, not a packaged inference service.

## Citation

Please cite the manuscript included in this repository. The citation metadata
will be updated after publication.
