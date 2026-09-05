#!/usr/bin/env python
"""Create the paper-ready five-seed S1129 formulation comparison."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "kbs_supplement"
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("pearson", "spearman", "rmse", "mae")
MODELS = {
    "Unconstrained pairwise": "skempi_s1129_dssa5_core_base_w020_seed",
    "Reverse-label supervision": "skempi_s1129_dssa5_core_rev_w020_seed",
    "Soft consistency": "skempi_s1129_dssa5_core_anti_w020_seed",
    "Hard antisymmetry": "skempi_s1129_dssa5_hard_antisym_seed",
    "Shared potential, no state descriptors": "skempi_s1129_dssa5_state_potential_nostate_seed",
    "Canonical four-state shared potential": (
        "skempi_s1129_dssa5_four_state_potential_canonical_seed"
    ),
}


def main():
    rows = []
    for model, stem in MODELS.items():
        values = {metric: [] for metric in METRICS}
        for seed in SEEDS:
            path = OUT / f"{stem}{seed}" / "summary.json"
            summary = json.loads(path.read_text())
            for metric in METRICS:
                values[metric].append(summary["mean_by_split"][metric]["fold_dssa_5"])
        row = {"model": model}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(np.mean(values[metric]))
            row[f"{metric}_sd"] = float(np.std(values[metric], ddof=1))
        rows.append(row)
    DEST.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "state_potential_model_comparison.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
