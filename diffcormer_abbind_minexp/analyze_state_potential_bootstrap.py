#!/usr/bin/env python
"""Fold-stratified complex bootstrap for four-state versus no-state potential."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "kbs_supplement"
SEEDS = (42, 43, 44, 45, 46)
REPS = 10_000
METRICS = ("PCC", "RMSE", "MAE")


def load_ensemble(stem):
    frames = []
    for seed in SEEDS:
        path = OUT / f"{stem}{seed}" / "oof_predictions.csv"
        frame = pd.read_csv(path)[
            ["dataset_index", "source_row", "target", "fold", "prediction"]
        ].rename(columns={"prediction": f"prediction_{seed}"})
        frames.append(frame)
    merged = frames[0]
    keys = ["dataset_index", "source_row", "target", "fold"]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, validate="one_to_one")
    merged["prediction"] = merged[
        [f"prediction_{seed}" for seed in SEEDS]
    ].mean(axis=1)
    return merged[keys + ["prediction"]]


def fast_metrics(target, prediction):
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    return {
        "PCC": float(np.corrcoef(target, prediction)[0, 1]),
        "RMSE": float(np.sqrt(np.mean(np.square(target - prediction)))),
        "MAE": float(np.mean(np.abs(target - prediction))),
    }


def fold_mean(frame, column):
    rows = [
        fast_metrics(group.target.to_numpy(), group[column].to_numpy())
        for _, group in frame.groupby("fold")
    ]
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-state-stem",
        default="skempi_s1129_dssa5_state_potential_nostate_seed",
    )
    parser.add_argument(
        "--four-state-stem",
        default="skempi_s1129_dssa5_four_state_potential_canonical_seed",
    )
    parser.add_argument("--output-stem", default="state_potential_canonical_paired_bootstrap")
    args = parser.parse_args()
    data = pd.read_csv(OUT / "data_skempi1131_dssa5" / "skempi_s1129_dssa5.csv")
    keys = ["dataset_index", "source_row", "target", "fold"]
    no_state = load_ensemble(args.no_state_stem).rename(columns={"prediction": "no_state"})
    four_state = load_ensemble(args.four_state_stem).rename(columns={"prediction": "four_state"})
    frame = no_state.merge(four_state, on=keys, validate="one_to_one")
    if frame.dataset_index.duplicated().any() or set(frame.dataset_index) != set(range(len(data))):
        raise RuntimeError("OOF dataset indices do not match the S1129 data table")
    complex_by_index = data["official_complex"].to_dict()
    frame["complex"] = frame.dataset_index.map(complex_by_index)

    no_state_metric = fold_mean(frame, "no_state")
    four_state_metric = fold_mean(frame, "four_state")
    observed = {
        metric: four_state_metric[metric] - no_state_metric[metric]
        for metric in METRICS
    }

    fold_complexes = {
        fold: group.complex.drop_duplicates().to_numpy()
        for fold, group in frame.groupby("fold")
    }
    groups = {
        (fold, complex_id): group.index.to_numpy()
        for (fold, complex_id), group in frame.groupby(["fold", "complex"])
    }
    target = frame.target.to_numpy()
    no_state_prediction = frame.no_state.to_numpy()
    four_state_prediction = frame.four_state.to_numpy()
    rng = np.random.default_rng(20260822)
    boot = {metric: np.empty(REPS, dtype=np.float64) for metric in METRICS}
    for replicate in range(REPS):
        no_state_rows, four_state_rows = [], []
        for fold, complexes in fold_complexes.items():
            sampled = rng.choice(complexes, size=len(complexes), replace=True)
            index = np.concatenate([groups[(fold, complex_id)] for complex_id in sampled])
            no_state_rows.append(fast_metrics(target[index], no_state_prediction[index]))
            four_state_rows.append(fast_metrics(target[index], four_state_prediction[index]))
        for metric in METRICS:
            boot[metric][replicate] = np.mean(
                [row[metric] for row in four_state_rows]
            ) - np.mean([row[metric] for row in no_state_rows])

    rows = []
    for metric in METRICS:
        values = boot[metric]
        improvement = values > 0 if metric == "PCC" else values < 0
        rows.append({
            "metric": metric,
            "no_state": no_state_metric[metric],
            "four_state": four_state_metric[metric],
            "difference_four_minus_no_state": observed[metric],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_probability_improvement": float(improvement.mean()),
        })

    DEST.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(DEST / f"{args.output_stem}.csv", index=False)
    payload = {
        "comparison": "five-seed ensemble four-state versus no-state shared potential",
        "estimand": "mean of the five outer-fold metrics",
        "resampling_unit": "official complex independently within each outer fold",
        "n_complexes": int(frame.complex.nunique()),
        "n_mutations": int(len(frame)),
        "bootstrap_replicates": REPS,
        "rows": rows,
    }
    (DEST / f"{args.output_stem}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
