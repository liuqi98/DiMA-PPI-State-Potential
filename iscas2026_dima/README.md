# DiMA-PPI conference manuscript and reproducible figures

This directory contains the current conference manuscript, its editable Word
version, figure sources, and the compact evidence used to generate the results.
The original journal manuscript remains under `../elsarticle/`.

## Manuscript

- [Editable Word document](dima_iscas2026_full.docx): single-column editing layout;
  Figure 1 uses the supplied [model figure.jpg](model%20figure.jpg).
- [IEEE conference PDF](dima_iscas2026_full.pdf): four pages of technical content,
  with references continuing on page five. This version uses the programmatically
  generated vector model figure, rather than the JPG embedded in Word.
- [Complete source archive](dima_iscas2026_full_source.zip).
- [Bibliography](dima_full_references.bib): all 12 entries are cited.

The `iscas2026` directory name records the requested template; these files are a
manuscript draft and do not indicate acceptance or submission to the conference.
The current direct-accuracy table compares DSSA-PPI and DiMA-PPI and includes
matched controls and an ablation. StaB-ddG is retained in the fixed-weight
canonical-reuse experiment, but not in that accuracy table.

## Build the PDF and Word document

Use a TeX installation providing `pdflatex` and `bibtex`; IEEEtran class and
bibliography style are bundled. From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error dima_iscas2026_full.tex
bibtex dima_iscas2026_full
pdflatex -interaction=nonstopmode -halt-on-error dima_iscas2026_full.tex
pdflatex -interaction=nonstopmode -halt-on-error dima_iscas2026_full.tex
```

To regenerate the Word document, install `pypandoc_binary` and `python-docx`,
then run `python build_word.py` after compiling LaTeX. The converter reads the
`.aux` and `.bbl` files to preserve equation, table, figure and citation numbers.
Equations and tables are editable Word objects. The supplied JPG is embedded
unchanged; `draw_model.py` regenerates the separate vector schematic.

To regenerate tables and figures, install `numpy` and `matplotlib` and run:

```bash
python build_results.py
python draw_model.py
```

`build_results.py` reads archived measurements in `evidence/`; it does not train
models or generate new experimental results. Recompile LaTeX after rebuilding.

## Model and analysis code

The existing [model implementation](../diffcormer_abbind_minexp/train_residue_graph.py)
and training runners remain the training entry points. This update also adds:

- [Canonical state-reuse benchmark](../diffcormer_abbind_minexp/bioinformatics_completion_scripts/benchmark_dima_canonical_reuse.py).
- [Unseen-edge metric analysis](../diffcormer_abbind_minexp/bioinformatics_completion_scripts/summarize_unseen_edges.py).
- [Paired complex bootstrap](../diffcormer_abbind_minexp/bioinformatics_completion_scripts/bootstrap_unseen_edges.py).
- [Unseen-edge protocol and results](../diffcormer_abbind_minexp/bioinformatics_completion_scripts/UNSEEN_EDGE_README.md).

Each Python analysis script accepts paths via command-line arguments; consult
`--help`. The reuse benchmark additionally requires trained checkpoints and
prepared graph caches. The repository-level README documents model dependencies.
No retraining is performed as part of this manuscript release.

## Data and provenance

[PPI-Path92](ppi_path92/dataset_card.json) contains 92 site panels from 27
complexes and 9,904 directed mutant-to-mutant comparisons. Their targets are
differences of measured WT endpoint effects, not independently measured
intermediate mutations. The exported CSVs contain labels and provenance without
model prediction columns.

`evidence/manifest.json` records upstream source paths and hashes;
`SHA256SUMS` checks the files in this directory, excluding itself. Large weights,
raw PDB structures and graph caches remain outside this compact repository.
