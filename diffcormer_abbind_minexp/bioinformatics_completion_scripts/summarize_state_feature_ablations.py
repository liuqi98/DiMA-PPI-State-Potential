#!/usr/bin/env python
"""Summarize retrained five-seed ablations and complex-cluster uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


SEEDS = (42, 43, 44, 45, 46)
METRICS = ("pearson", "spearman", "rmse", "mae")


def metrics(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    def correlation(left, right):
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
        return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")
    return {
        "pearson": correlation(target, prediction),
        "spearman": correlation(rankdata(target), rankdata(prediction)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mae": float(np.mean(np.abs(prediction - target))),
    }


def load_seed_summary(directory):
    summary = json.loads((directory / "summary.json").read_text())
    return {
        metric: float(summary["mean_by_split"][metric]["fold_dssa_5"])
        for metric in METRICS
    }


def averaged_oof(directories):
    frames = []
    for directory in directories:
        frame = pd.read_csv(directory / "oof_predictions.csv")
        frames.append(frame[["source_row", "target", "prediction"]].set_index("source_row"))
    base = frames[0][["target"]].copy()
    for index, frame in enumerate(frames):
        if not base.index.equals(frame.index) or not np.allclose(base.target, frame.target):
            raise RuntimeError("OOF prediction tables do not align")
        base[f"seed_{index}"] = frame.prediction
    base["prediction"] = base.filter(like="seed_").mean(axis=1)
    return base[["target", "prediction"]]


def cluster_bootstrap(data, full, ablated, repeats, seed):
    rng = np.random.default_rng(seed)
    complexes = data.official_complex.astype(str).unique()
    observed_full = metrics(full.target, full.prediction)
    observed_ablated = metrics(ablated.target, ablated.prediction)
    values = {metric: [] for metric in METRICS}
    complex_values = data.official_complex.astype(str).to_numpy()
    for _ in range(repeats):
        chosen = rng.choice(complexes, size=len(complexes), replace=True)
        idx = np.concatenate([np.flatnonzero(complex_values == value) for value in chosen])
        full_metric = metrics(full.target.to_numpy()[idx], full.prediction.to_numpy()[idx])
        ablated_metric = metrics(ablated.target.to_numpy()[idx], ablated.prediction.to_numpy()[idx])
        values["pearson"].append(full_metric["pearson"] - ablated_metric["pearson"])
        values["spearman"].append(full_metric["spearman"] - ablated_metric["spearman"])
        values["rmse"].append(ablated_metric["rmse"] - full_metric["rmse"])
        values["mae"].append(ablated_metric["mae"] - full_metric["mae"])
    output = {}
    for metric in METRICS:
        samples = np.asarray(values[metric])
        observed = (
            observed_full[metric] - observed_ablated[metric]
            if metric in ("pearson", "spearman")
            else observed_ablated[metric] - observed_full[metric]
        )
        output[metric] = {
            "performance_loss_when_removed": float(observed),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_two_sided_p": float(min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()

    reference_root = Path(args.reference_root)
    ablation_root = Path(args.ablation_root)
    models = {
        "Full canonical four-state": [
            reference_root / f"skempi_s1129_dssa5_four_state_potential_canonical_seed{seed}"
            for seed in SEEDS
        ],
        "No state descriptors": [
            reference_root / f"skempi_s1129_dssa5_state_potential_nostate_seed{seed}"
            for seed in SEEDS
        ],
        "No ProteinMPNN": [
            ablation_root / f"skempi_s1129_state_potential_no_mpnn_seed{seed}"
            for seed in SEEDS
        ],
        "No topology": [
            ablation_root / f"skempi_s1129_state_potential_no_topology_seed{seed}"
            for seed in SEEDS
        ],
        "No atomic delta": [
            ablation_root / f"skempi_s1129_state_potential_no_atomic_seed{seed}"
            for seed in SEEDS
        ],
        "ESM + state descriptors only": [
            ablation_root / f"skempi_s1129_state_potential_esm_only_seed{seed}"
            for seed in SEEDS
        ],
    }
    rows = []
    seed_rows = []
    ensembles = {}
    for model, directories in models.items():
        values = {metric: [] for metric in METRICS}
        for seed, directory in zip(SEEDS, directories):
            current = load_seed_summary(directory)
            seed_rows.append({"model": model, "seed": seed, **current})
            for metric in METRICS:
                values[metric].append(current[metric])
        row = {"model": model}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(np.mean(values[metric]))
            row[f"{metric}_sd"] = float(np.std(values[metric], ddof=1))
        rows.append(row)
        ensembles[model] = averaged_oof(directories)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "state_feature_ablation_table.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "state_feature_ablation_per_seed.csv", index=False)
    data = pd.read_csv(args.data_csv).set_index("source_row").loc[ensembles["Full canonical four-state"].index]
    bootstrap = {}
    for model, ensemble in ensembles.items():
        if model == "Full canonical four-state":
            continue
        bootstrap[model] = cluster_bootstrap(
            data,
            ensembles["Full canonical four-state"],
            ensemble,
            args.bootstrap_repeats,
            args.bootstrap_seed,
        )
    (output / "state_feature_ablation_complex_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
