#!/usr/bin/env python
"""Evaluate native unseen mutant-to-mutant edges against derived endpoint labels.

For a held-out single-site panel with measured WT->B and WT->C effects, the
derived B->C target is y(WT->C) - y(WT->B).  It is not an independently
measured thermodynamic step; the purpose is to test compositional prediction
on edges absent from the single-mutant training protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


METRICS = ("pcc", "spearman", "rmse", "mae")
KEY = ["official_complex", "chain_id", "pdb_pos", "wt_aa"]


def metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return {
        "pcc": float(pearsonr(prediction, target).statistic),
        "spearman": float(spearmanr(prediction, target).statistic),
        "rmse": float(np.mean((prediction - target) ** 2) ** 0.5),
        "mae": float(np.mean(np.abs(prediction - target))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--path-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data_csv)
    representatives = (
        data.sort_values("source_row")
        .drop_duplicates(KEY + ["mut_aa"])
        [KEY + ["mut_aa", "ddG", "fold_dssa_5", "source_row"]]
    )
    first = representatives.rename(
        columns={
            "mut_aa": "intermediate_aa",
            "ddG": "target_step_one",
            "source_row": "intermediate_source_row",
        }
    )
    second = representatives.rename(
        columns={
            "mut_aa": "final_aa",
            "ddG": "target_direct",
            "source_row": "final_source_row",
        }
    )
    paths = pd.read_csv(args.path_predictions)
    rows = paths.merge(
        first,
        on=KEY + ["intermediate_aa"],
        validate="many_to_one",
    ).merge(
        second,
        on=KEY + ["final_aa"],
        suffixes=("_intermediate", "_final"),
        validate="many_to_one",
    )
    np.testing.assert_array_equal(rows["fold"], rows["fold_dssa_5_intermediate"])
    np.testing.assert_array_equal(rows["fold"], rows["fold_dssa_5_final"])
    if len(rows) != len(paths):
        raise RuntimeError(f"join lost path rows: {len(paths)} -> {len(rows)}")

    rows["target_unseen_edge"] = rows["target_direct"] - rows["target_step_one"]
    rows["native_unseen_prediction"] = rows["step_two_prediction"]
    rows["endpoint_composed_prediction"] = (
        rows["direct_prediction"] - rows["step_one_prediction"]
    )
    np.testing.assert_allclose(
        np.abs(
            rows["native_unseen_prediction"]
            - rows["endpoint_composed_prediction"]
        ),
        rows["path_error"],
        rtol=0,
        atol=2e-12,
    )

    fold_rows = []
    seed_rows = []
    prediction_types = {
        "native_unseen": "native_unseen_prediction",
        "endpoint_composed": "endpoint_composed_prediction",
    }
    for (seed, variant), group in rows.groupby(["seed", "variant"], sort=True):
        by_type = {}
        for prediction_type, prediction_column in prediction_types.items():
            per_fold = []
            for fold, fold_group in group.groupby("fold", sort=True):
                values = metrics(
                    fold_group[prediction_column], fold_group["target_unseen_edge"]
                )
                fold_rows.append(
                    {
                        "seed": int(seed),
                        "variant": variant,
                        "prediction_type": prediction_type,
                        "fold": int(fold),
                        "edges": len(fold_group),
                        **values,
                    }
                )
                per_fold.append([values[name] for name in METRICS])
            by_type[prediction_type] = np.mean(per_fold, axis=0)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "variant": variant,
                    "prediction_type": prediction_type,
                    "edges": len(group),
                    **dict(zip(METRICS, by_type[prediction_type])),
                }
            )

    fold_frame = pd.DataFrame(fold_rows)
    seed_frame = pd.DataFrame(seed_rows)
    summary_rows = []
    for (variant, prediction_type), group in seed_frame.groupby(
        ["variant", "prediction_type"], sort=True
    ):
        row = {
            "variant": variant,
            "prediction_type": prediction_type,
            "seeds": len(group),
            "edges_per_seed": int(group["edges"].iloc[0]),
        }
        for name in METRICS:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_sd"] = float(group[name].std(ddof=1))
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)

    audit_columns = [
        "seed",
        "variant",
        "fold",
        *KEY,
        "intermediate_aa",
        "final_aa",
        "intermediate_source_row",
        "final_source_row",
        "target_step_one",
        "target_direct",
        "target_unseen_edge",
        "native_unseen_prediction",
        "endpoint_composed_prediction",
        "path_error",
    ]
    rows[audit_columns].to_csv(args.output_dir / "unseen_edge_predictions.csv", index=False)
    fold_frame.to_csv(args.output_dir / "unseen_edge_per_fold.csv", index=False)
    seed_frame.to_csv(args.output_dir / "unseen_edge_per_seed.csv", index=False)
    summary_frame.to_csv(args.output_dir / "unseen_edge_summary.csv", index=False)
    metadata = {
        "target_definition": "y(WT->C) - y(WT->B)",
        "target_status": "derived from two assayed endpoints, not independently measured B->C",
        "training_scope": "original single-mutant protocol contains WT->mutant edges, not mutant->mutant edges",
        "aggregation": "metric per outer fold, mean folds within seed, then mean and sample SD across five seeds",
        "rows": len(rows),
        "variants": sorted(rows["variant"].unique()),
        "seeds": sorted(map(int, rows["seed"].unique())),
    }
    (args.output_dir / "unseen_edge_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
