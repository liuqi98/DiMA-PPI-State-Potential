#!/usr/bin/env python
"""Evaluate the canonical state-potential checkpoints on external AbBind.

For each random seed, the five official S1129 fold models are averaged.  The
primary external estimate is summarized across the five independently trained
seed ensembles; an additional 25-model ensemble is retained for deployment.
No AbBind label is used for fitting, model selection, or ensemble weighting.
"""
import argparse
import glob
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch_geometric.loader import DataLoader


def metric_row(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    def correlation(left, right):
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
        return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")
    row = {
        "pearson": correlation(target, prediction),
        "spearman": correlation(rankdata(target), rankdata(prediction)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mae": float(np.mean(np.abs(prediction - target))),
    }
    labels = target > 0
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive and negative:
        prediction_ranks = rankdata(prediction)
        row["auroc"] = float(
            (prediction_ranks[labels].sum() - positive * (positive + 1) / 2)
            / (positive * negative)
        )
    else:
        row["auroc"] = None
    return row


def load_model(module, checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu")
    sequence_dim = int(saved["embedding_dim"]) * 3
    structure_dim = int(saved["input_dim"]) - sequence_dim - 4
    model = module.ResidueGraphModel(
        int(saved["input_dim"]),
        int(saved["hidden_dim"]),
        int(saved["layers"]),
        float(saved["dropout"]),
        input_fusion=saved["input_fusion"],
        sequence_dim=sequence_dim,
        structure_dim=structure_dim,
        readout=saved["readout"],
        pma_seeds=int(saved["pma_seeds"]),
        topology_dim=int(saved["topology_dim"]),
        atomic_delta_dim=int(saved["atomic_delta_dim"]),
        state_descriptor_dim=int(saved["state_descriptor_dim"]),
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, saved


def predict_checkpoint(module, graphs, checkpoint, batch_size, device):
    model, saved = load_model(module, checkpoint, device)
    predictions, reverse_predictions = [], []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reverse = module.reverse_mutation_batch(batch, int(saved["embedding_dim"]))
            pred = model(batch, cycle_mode=saved["cycle_mode"])
            reverse_pred = model(reverse, cycle_mode=saved["cycle_mode"])
            predictions.extend(
                (pred.cpu() * float(saved["target_std"]) + float(saved["target_mean"])).tolist()
            )
            reverse_predictions.extend(
                (reverse_pred.cpu() * float(saved["target_std"]) + float(saved["target_mean"])).tolist()
            )
    return np.asarray(predictions), np.asarray(reverse_predictions)


def legacy_ensemble(pattern, source_rows):
    files = [Path(value) for value in sorted(glob.glob(pattern))]
    if not files:
        return None, []
    predictions = []
    for path in files:
        frame = pd.read_csv(path).set_index("source_row")
        predictions.append(frame.loc[source_rows, "prediction"].to_numpy(float))
    return np.mean(predictions, axis=0), [str(path) for path in files]


def clustered_bootstrap(data, proposed, baseline, repeats, seed):
    rng = np.random.default_rng(seed)
    complexes = data.pdb_id.astype(str).unique()
    metric_names = ["pearson", "spearman", "rmse", "mae", "auroc"]
    samples = {name: [] for name in metric_names}
    for _ in range(repeats):
        chosen = rng.choice(complexes, size=len(complexes), replace=True)
        indices = np.concatenate([
            np.flatnonzero(data.pdb_id.astype(str).to_numpy() == complex_id)
            for complex_id in chosen
        ])
        proposed_metrics = metric_row(data.ddG.to_numpy()[indices], proposed[indices])
        baseline_metrics = metric_row(data.ddG.to_numpy()[indices], baseline[indices])
        for name in metric_names:
            samples[name].append(proposed_metrics[name] - baseline_metrics[name])
    output = {}
    for name, values in samples.items():
        values = np.asarray(values, dtype=float)
        output[name] = {
            "difference": metric_row(data.ddG, proposed)[name] - metric_row(data.ddG, baseline)[name],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_two_sided_p": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--legacy-pattern")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument(
        "--reuse-member-predictions",
        action="store_true",
        help="Reuse a completed member-prediction CSV and run only aggregation/statistics.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.code_dir).resolve()))
    module = importlib.import_module("train_residue_graph")
    module.pooled.patch_broken_dynamo()
    device = torch.device(args.device)
    data = pd.read_csv(args.data_csv)
    cache = torch.load(args.cache_path, map_location="cpu")
    if cache.get("row_ids") != [int(value) for value in data.source_row]:
        raise RuntimeError("Canonical cache does not align with the AbBind table")
    graphs = cache["graphs"]
    if int(cache.get("state_descriptor_valid_count", len(graphs))) != len(graphs):
        raise RuntimeError("Not all AbBind graphs have valid four-state descriptors")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    member_path = output / "abbind_state_member_predictions.csv"
    members = []
    seed_predictions = {}
    seed_reverse_predictions = {}
    if args.reuse_member_predictions and member_path.is_file():
        member_frame = pd.read_csv(member_path)
        for seed in args.seeds:
            selected = member_frame.loc[member_frame.seed.eq(seed)]
            grouped = selected.groupby("source_row")[["prediction", "reverse_prediction"]].mean()
            grouped = grouped.loc[data.source_row.astype(int)]
            seed_predictions[seed] = grouped.prediction.to_numpy()
            seed_reverse_predictions[seed] = grouped.reverse_prediction.to_numpy()
    else:
        for seed in args.seeds:
            fold_predictions, fold_reverse = [], []
            for fold in args.folds:
                checkpoint = (
                    Path(args.checkpoint_root)
                    / f"skempi_s1129_dssa5_four_state_potential_canonical_seed{seed}"
                    / f"checkpoint_fold_dssa_5_fold{fold}.pt"
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                prediction, reverse_prediction = predict_checkpoint(
                    module, graphs, checkpoint, args.batch_size, device
                )
                fold_predictions.append(prediction)
                fold_reverse.append(reverse_prediction)
                for index, row in data.iterrows():
                    members.append({
                        "source_row": int(row.source_row),
                        "pdb_id": row.pdb_id,
                        "target": float(row.ddG),
                        "seed": seed,
                        "fold": fold,
                        "prediction": float(prediction[index]),
                        "reverse_prediction": float(reverse_prediction[index]),
                        "reverse_residual": float(prediction[index] + reverse_prediction[index]),
                    })
            seed_predictions[seed] = np.mean(fold_predictions, axis=0)
            seed_reverse_predictions[seed] = np.mean(fold_reverse, axis=0)
        member_frame = pd.DataFrame(members)
        member_frame.to_csv(member_path, index=False)
    seed_rows = []
    for seed in args.seeds:
        row = {"seed": seed, **metric_row(data.ddG, seed_predictions[seed])}
        row["reverse_mae"] = float(np.mean(np.abs(seed_predictions[seed] + seed_reverse_predictions[seed])))
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(output / "abbind_state_seed_metrics.csv", index=False)

    proposed = np.mean([seed_predictions[seed] for seed in args.seeds], axis=0)
    reverse = np.mean([seed_reverse_predictions[seed] for seed in args.seeds], axis=0)
    prediction_frame = data[["source_row", "pdb_id", "chain_id", "pdb_pos", "wt_aa", "mut_aa", "ddG"]].copy()
    prediction_frame["prediction"] = proposed
    prediction_frame["reverse_prediction"] = reverse
    prediction_frame["reverse_residual"] = proposed + reverse

    summary = {
        "test_rows": len(data),
        "test_complexes": int(data.pdb_id.nunique()),
        "training_labels_from_abbind": 0,
        "selection_labels_from_abbind": 0,
        "fold_models_per_seed": len(args.folds),
        "random_seeds": args.seeds,
        "seed_mean": seed_frame.drop(columns="seed").mean().to_dict(),
        "seed_std": seed_frame.drop(columns="seed").std(ddof=1).to_dict(),
        "all_model_ensemble": metric_row(data.ddG, proposed),
        "all_model_ensemble_reverse_mae": float(np.mean(np.abs(proposed + reverse))),
        "all_model_ensemble_reverse_max_abs": float(np.max(np.abs(proposed + reverse))),
    }
    if args.legacy_pattern:
        baseline, files = legacy_ensemble(args.legacy_pattern, data.source_row.astype(int).tolist())
        if baseline is not None:
            prediction_frame["legacy_backbone_ensemble"] = baseline
            summary["legacy_backbone_ensemble"] = metric_row(data.ddG, baseline)
            summary["legacy_prediction_files"] = files
            summary["proposed_minus_legacy_complex_bootstrap"] = clustered_bootstrap(
                data, proposed, baseline, args.bootstrap_repeats, args.bootstrap_seed
            )
    prediction_frame.to_csv(output / "abbind_state_ensemble_predictions.csv", index=False)
    (output / "abbind_state_external_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
