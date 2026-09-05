#!/usr/bin/env python
"""Aggregate five train-all-S1129 state-potential external runs."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def metrics(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    def corr(left, right):
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
        return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")
    labels = target > 0
    positive, negative = int(labels.sum()), int((~labels).sum())
    ranks = rankdata(prediction)
    return {
        "pearson": corr(target, prediction),
        "spearman": corr(rankdata(target), ranks),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "auroc": float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative)),
    }


def load_ensemble(files, source_rows):
    values, reverse = [], []
    for path in files:
        frame = pd.read_csv(path).set_index("source_row").loc[source_rows]
        values.append(frame.prediction.to_numpy(float))
        reverse.append(frame.reverse_prediction.to_numpy(float))
    return np.asarray(values), np.asarray(reverse)


def bootstrap(data, proposed, baseline, repeats, seed):
    rng = np.random.default_rng(seed)
    complex_values = data.pdb_id.astype(str).to_numpy()
    complexes = np.unique(complex_values)
    observed_p = metrics(data.ddG, proposed)
    observed_b = metrics(data.ddG, baseline)
    samples = {name: [] for name in observed_p}
    for _ in range(repeats):
        chosen = rng.choice(complexes, len(complexes), replace=True)
        idx = np.concatenate([np.flatnonzero(complex_values == value) for value in chosen])
        left = metrics(data.ddG.to_numpy()[idx], proposed[idx])
        right = metrics(data.ddG.to_numpy()[idx], baseline[idx])
        for name in samples:
            samples[name].append(left[name] - right[name])
    result = {}
    for name, values in samples.items():
        values = np.asarray(values)
        result[name] = {
            "proposed_minus_backbone": float(observed_p[name] - observed_b[name]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_two_sided_p": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--refit-root", required=True)
    parser.add_argument("--legacy-pattern", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    args = parser.parse_args()

    data = pd.read_csv(args.data_csv)
    data = data.loc[data.external_source.eq("AbBind_test")].reset_index(drop=True)
    source_rows = data.source_row.astype(int).tolist()
    proposed_files = [Path(args.refit_root) / f"seed{seed}" / "oof_predictions.csv" for seed in (42, 43, 44, 45, 46)]
    legacy_files = [Path(path) for path in sorted(glob.glob(args.legacy_pattern))]
    if not all(path.is_file() for path in proposed_files) or len(legacy_files) != 5:
        raise RuntimeError("Expected five proposed and five legacy prediction files")
    proposed_seeds, proposed_reverse = load_ensemble(proposed_files, source_rows)
    legacy_seeds, _ = load_ensemble(legacy_files, source_rows)
    seed_rows = []
    for seed, prediction, reverse in zip((42, 43, 44, 45, 46), proposed_seeds, proposed_reverse):
        seed_rows.append({
            "seed": seed,
            **metrics(data.ddG, prediction),
            "reverse_mae": float(np.mean(np.abs(prediction + reverse))),
        })
    seed_frame = pd.DataFrame(seed_rows)
    proposed = proposed_seeds.mean(axis=0)
    reverse = proposed_reverse.mean(axis=0)
    legacy = legacy_seeds.mean(axis=0)
    summary = {
        "test_rows": len(data),
        "test_complexes": int(data.pdb_id.nunique()),
        "training_rows": 1129,
        "training_labels_from_abbind": 0,
        "selection_labels_from_abbind": 0,
        "seed_mean": seed_frame.drop(columns="seed").mean().to_dict(),
        "seed_std": seed_frame.drop(columns="seed").std(ddof=1).to_dict(),
        "five_seed_ensemble": metrics(data.ddG, proposed),
        "ensemble_reverse_mae": float(np.mean(np.abs(proposed + reverse))),
        "ensemble_reverse_max_abs": float(np.max(np.abs(proposed + reverse))),
        "legacy_backbone_ensemble": metrics(data.ddG, legacy),
        "proposed_minus_legacy_complex_bootstrap": bootstrap(
            data, proposed, legacy, args.bootstrap_repeats, args.bootstrap_seed
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(output / "external_refit_seed_metrics.csv", index=False)
    predictions = data[["source_row", "pdb_id", "ddG"]].copy()
    predictions["state_potential_prediction"] = proposed
    predictions["state_potential_reverse_prediction"] = reverse
    predictions["legacy_backbone_prediction"] = legacy
    predictions.to_csv(output / "external_refit_ensemble_predictions.csv", index=False)
    (output / "external_refit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
