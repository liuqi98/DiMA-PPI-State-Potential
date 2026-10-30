#!/usr/bin/env python
"""Fold-stratified complex bootstrap for unseen-edge ensemble comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("pairwise", "soft_consistency", "hard_antisymmetry", "four_state_potential")
PROPOSED = "four_state_potential"
METRICS = ("pcc", "rmse", "mae")
IDENTITY = [
    "fold",
    "official_complex",
    "chain_id",
    "pdb_pos",
    "wt_aa",
    "intermediate_aa",
    "final_aa",
    "target_unseen_edge",
]


def metrics(target, prediction):
    return {
        "pcc": float(np.corrcoef(target, prediction)[0, 1]),
        "rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
        "mae": float(np.mean(np.abs(target - prediction))),
    }


def fold_mean(frame, prediction):
    values = [
        metrics(group.target_unseen_edge.to_numpy(), group[prediction].to_numpy())
        for _, group in frame.groupby("fold", sort=True)
    ]
    return {name: float(np.mean([row[name] for row in values])) for name in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    data = pd.read_csv(args.predictions)
    ensemble = (
        data.groupby(IDENTITY + ["variant"], as_index=False)["native_unseen_prediction"]
        .mean()
        .pivot(index=IDENTITY, columns="variant", values="native_unseen_prediction")
        .reset_index()
    )
    if len(ensemble) != 9904 or set(VARIANTS) - set(ensemble.columns):
        raise RuntimeError("unexpected unseen-edge ensemble shape")
    ensemble = ensemble.rename(columns={name: f"prediction_{name}" for name in VARIANTS})
    observed = {
        variant: fold_mean(ensemble, f"prediction_{variant}") for variant in VARIANTS
    }

    fold_complexes = {
        fold: group.official_complex.drop_duplicates().to_numpy()
        for fold, group in ensemble.groupby("fold", sort=True)
    }
    groups = {
        (fold, complex_id): group.index.to_numpy()
        for (fold, complex_id), group in ensemble.groupby(
            ["fold", "official_complex"], sort=True
        )
    }
    target = ensemble.target_unseen_edge.to_numpy()
    predictions = {
        variant: ensemble[f"prediction_{variant}"].to_numpy() for variant in VARIANTS
    }
    rng = np.random.default_rng(args.seed)
    comparisons = [variant for variant in VARIANTS if variant != PROPOSED]
    boot = {
        (variant, metric): np.empty(args.repeats, dtype=np.float64)
        for variant in comparisons
        for metric in METRICS
    }
    for replicate in range(args.repeats):
        fold_values = {variant: [] for variant in VARIANTS}
        for fold, complexes in fold_complexes.items():
            sampled = rng.choice(complexes, size=len(complexes), replace=True)
            index = np.concatenate([groups[(fold, complex_id)] for complex_id in sampled])
            for variant in VARIANTS:
                fold_values[variant].append(metrics(target[index], predictions[variant][index]))
        for control in comparisons:
            for metric in METRICS:
                proposed_value = np.mean([row[metric] for row in fold_values[PROPOSED]])
                control_value = np.mean([row[metric] for row in fold_values[control]])
                boot[(control, metric)][replicate] = proposed_value - control_value

    rows = []
    for control in comparisons:
        for metric in METRICS:
            values = boot[(control, metric)]
            rows.append({
                "comparison": f"{PROPOSED}_minus_{control}",
                "metric": metric,
                "control": observed[control][metric],
                "proposed": observed[PROPOSED][metric],
                "difference": observed[PROPOSED][metric] - observed[control][metric],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_proposed_better": float(
                    np.mean(values > 0) if metric == "pcc" else np.mean(values < 0)
                ),
            })
    payload = {
        "estimand": "mean of five outer-fold metrics on five-seed ensemble predictions",
        "resampling_unit": "official complex independently within each outer fold",
        "target_status": "derived endpoint difference, not an independently measured B->C step",
        "edges": len(ensemble),
        "complexes": int(ensemble.official_complex.nunique()),
        "bootstrap_repeats": args.repeats,
        "bootstrap_seed": args.seed,
        "observed": observed,
        "comparisons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
