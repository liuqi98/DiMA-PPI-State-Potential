#!/usr/bin/env python
"""Summarize canonical state-potential results on strict S1129 partitions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "kbs_supplement"
SEEDS = (42, 43, 44, 45, 46)


def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    return {
        "PCC": float(pearsonr(target, prediction).statistic),
        "Spearman": float(spearmanr(target, prediction).statistic),
        "RMSE": float(mean_squared_error(target, prediction) ** 0.5),
        "MAE": float(mean_absolute_error(target, prediction)),
        "AUROC": float(roc_auc_score(target > 0, prediction)),
    }


def load_ensemble(pattern, split=None):
    frames, seed_rows = [], []
    for seed in SEEDS:
        frame = pd.read_csv(OUT / pattern.format(seed=seed) / "oof_predictions.csv")
        if split is not None:
            frame = frame[frame.split == split].copy()
        frame = frame[["dataset_index", "source_row", "target", "prediction"]]
        seed_rows.append({"seed": seed, **metrics(frame.target, frame.prediction)})
        frame = frame.rename(columns={"prediction": f"prediction_{seed}"})
        frames.append(frame)
    merged = frames[0]
    keys = ["dataset_index", "source_row", "target"]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, validate="one_to_one")
    merged["prediction"] = merged[[f"prediction_{seed}" for seed in SEEDS]].mean(axis=1)
    return merged, seed_rows


def paired_complex_bootstrap(reference, candidate, groups, repeats=10_000, seed=20260822):
    frame = reference[["dataset_index", "target", "prediction"]].rename(
        columns={"prediction": "reference"}
    ).merge(
        candidate[["dataset_index", "prediction"]].rename(columns={"prediction": "candidate"}),
        on="dataset_index", validate="one_to_one",
    )
    frame["complex"] = frame.dataset_index.map(groups)
    observed_reference = metrics(frame.target, frame.reference)
    observed_candidate = metrics(frame.target, frame.candidate)
    complexes = frame.complex.drop_duplicates().to_numpy()
    indices = {key: group.index.to_numpy() for key, group in frame.groupby("complex")}
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in observed_reference}
    target = frame.target.to_numpy()
    reference_prediction = frame.reference.to_numpy()
    candidate_prediction = frame.candidate.to_numpy()
    for _ in range(repeats):
        sampled = rng.choice(complexes, len(complexes), replace=True)
        index = np.concatenate([indices[key] for key in sampled])
        if np.unique(target[index] > 0).size < 2:
            continue
        reference_metrics = metrics(target[index], reference_prediction[index])
        candidate_metrics = metrics(target[index], candidate_prediction[index])
        for metric in values:
            values[metric].append(candidate_metrics[metric] - reference_metrics[metric])
    rows = []
    for metric, samples in values.items():
        samples = np.asarray(samples)
        rows.append({
            "metric": metric,
            "reference": observed_reference[metric],
            "candidate": observed_candidate[metric],
            "difference": observed_candidate[metric] - observed_reference[metric],
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "probability_improvement": float(
                np.mean(samples < 0) if metric in ("RMSE", "MAE") else np.mean(samples > 0)
            ),
        })
    return rows


def main():
    data = pd.read_csv(OUT / "data_skempi1131_dssa5" / "skempi_s1129_all_strict.csv")
    groups = data.official_complex.astype(str).to_dict()
    split_names = {
        "fold_sequence_cluster_5": "sequence_cluster",
        "fold_interface_structure_5": "interface_structure",
    }
    result_rows, bootstrap = [], {}
    for split, label in split_names.items():
        canonical, seed_rows = load_ensemble(
            "skempi_s1129_canonical_state_potential_strict_seed{seed}", split
        )
        base, _ = load_ensemble(
            f"skempi_s1129_{'sequence5' if label == 'sequence_cluster' else 'interface5'}_base_w020_seed{{seed}}"
        )
        soft, _ = load_ensemble(
            f"skempi_s1129_{'sequence5' if label == 'sequence_cluster' else 'interface5'}_anti_w020_seed{{seed}}"
        )
        hard, _ = load_ensemble(
            "skempi_s1129_hard_strict_seed{seed}", split
        )
        for model, frame, rows in (
            ("Unconstrained backbone", base, None),
            ("Soft consistency", soft, None),
            ("Hard antisymmetry", hard, None),
            ("Canonical state potential", canonical, seed_rows),
        ):
            values = metrics(frame.target, frame.prediction)
            entry = {"split": label, "model": model, **values}
            if rows is not None:
                for metric in values:
                    entry[f"seed_mean_{metric}"] = float(np.mean([row[metric] for row in rows]))
                    entry[f"seed_sd_{metric}"] = float(np.std([row[metric] for row in rows], ddof=1))
            result_rows.append(entry)
        bootstrap[label] = {
            "canonical_minus_backbone": paired_complex_bootstrap(base, canonical, groups),
            "canonical_minus_soft": paired_complex_bootstrap(soft, canonical, groups),
            "canonical_minus_hard": paired_complex_bootstrap(hard, canonical, groups),
        }
    DEST.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(result_rows)
    table.to_csv(DEST / "canonical_state_potential_strict_table.csv", index=False)
    (DEST / "canonical_state_potential_strict_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
