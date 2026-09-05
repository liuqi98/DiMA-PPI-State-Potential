#!/usr/bin/env python
"""OOF path-integrability audit on real single-site mutation panels.

Each panel contains a crystallographic WT state A and at least two measured
mutants B and C at the same interface site.  Models are evaluated on the
synthetic-but-physically-defined edges A->B, B->C, and A->C using identical
cached state representations.  The B->C graph uses no unobserved directional
atomic delta; matched pairwise controls are therefore trained without atomic
delta and topology features.
"""
import argparse
import json
import sys
from itertools import combinations, permutations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

# The compact manuscript archive omits the multi-gigabyte feature-generation
# dependencies.  When run inside the full project, prefer its complete code
# directory so checkpoint classes and helper modules resolve consistently.
FULL_CODE_DIR = Path(__file__).resolve().parents[3] / "diffcormer_abbind_minexp"
if FULL_CODE_DIR.is_dir():
    sys.path.insert(0, str(FULL_CODE_DIR))

from train_residue_graph import ResidueGraphModel, reverse_mutation_batch


def compatible_graphs(left, right, embedding_dim):
    if left.x.shape != right.x.shape:
        return False
    if not torch.equal(left.edge_index, right.edge_index):
        return False
    if not torch.equal(left.edge_attr, right.edge_attr):
        return False
    if not torch.equal(left.mutation_mask, right.mutation_mask):
        return False
    if not torch.equal(left.other_chain_mask, right.other_chain_mask):
        return False
    return torch.equal(left.x[:, 3 * embedding_dim:], right.x[:, 3 * embedding_dim:])


def state_components(graph, mutant, embedding_dim):
    start = embedding_dim if mutant else 0
    sequence = graph.x[:, start:start + embedding_dim]
    descriptor_index = 1 if mutant else 0
    bound = graph.state_descriptors[:, descriptor_index, :]
    free = graph.state_descriptors[:, descriptor_index + 2, :]
    return sequence, bound, free


def make_edge(template, left_graph, left_mutant, right_graph, right_mutant, embedding_dim):
    if not compatible_graphs(template, left_graph, embedding_dim):
        raise RuntimeError("Left-state graph is not aligned with the panel template")
    if not compatible_graphs(template, right_graph, embedding_dim):
        raise RuntimeError("Right-state graph is not aligned with the panel template")
    left, left_bound, left_free = state_components(left_graph, left_mutant, embedding_dim)
    right, right_bound, right_free = state_components(right_graph, right_mutant, embedding_dim)
    graph = template.clone()
    graph.x = torch.cat([
        left, right, right - left, template.x[:, 3 * embedding_dim:]
    ], dim=1)
    graph.state_descriptors = torch.stack([
        left_bound.squeeze(0), right_bound.squeeze(0),
        left_free.squeeze(0), right_free.squeeze(0),
    ], dim=0)[None]
    return graph


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ResidueGraphModel(
        checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["layers"],
        checkpoint["dropout"], input_fusion=checkpoint["input_fusion"],
        sequence_dim=3 * checkpoint["embedding_dim"],
        structure_dim=checkpoint["input_dim"] - 3 * checkpoint["embedding_dim"] - 4,
        readout=checkpoint["readout"], pma_seeds=checkpoint["pma_seeds"],
        topology_dim=checkpoint["topology_dim"],
        atomic_delta_dim=checkpoint["atomic_delta_dim"],
        state_descriptor_dim=checkpoint["state_descriptor_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def predict(model, checkpoint, graphs, batch_size, device):
    values = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            raw = model(batch, cycle_mode=checkpoint["cycle_mode"])
            if checkpoint["hard_antisymmetry"]:
                reverse = reverse_mutation_batch(batch, checkpoint["embedding_dim"])
                raw = 0.5 * (
                    raw - model(reverse, cycle_mode=checkpoint["cycle_mode"])
                )
            calibrated = raw * checkpoint["target_std"] + checkpoint["target_mean"]
            values.extend(calibrated.detach().cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def summarize(values, thresholds):
    values = np.asarray(values, dtype=np.float64)
    result = {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }
    for threshold in thresholds:
        result[f"fraction_below_{threshold:g}"] = float(np.mean(values < threshold))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--canonical-checkpoint-root",
        help="Directory containing canonical seed directories (defaults to local outputs)",
    )
    parser.add_argument(
        "--control-checkpoint-root",
        help="Directory containing newly completed pairwise/soft/hard seed directories",
    )
    parser.add_argument(
        "--fallback-control-root",
        help="Fallback directory, used for pre-existing control seeds such as seed 43",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--strict-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    canonical_root = Path(args.canonical_checkpoint_root) if args.canonical_checkpoint_root else root / "outputs"
    control_root = Path(args.control_checkpoint_root) if args.control_checkpoint_root else root / "outputs"
    fallback_control_root = Path(args.fallback_control_root) if args.fallback_control_root else control_root
    data = pd.read_csv(args.data_csv)
    cache = torch.load(args.cache_path, map_location="cpu")
    if cache.get("row_ids") != [int(value) for value in data.source_row]:
        raise RuntimeError("Cache and data table do not align")
    graphs = cache["graphs"]
    embedding_dim = int(cache["embedding_dim"])
    key_columns = ["official_complex", "chain_id", "pdb_pos", "wt_aa", "fold_dssa_5"]
    panels = []
    for key, group in data.groupby(key_columns, sort=True):
        representatives = group.sort_values("source_row").drop_duplicates("mut_aa")
        if len(representatives) < 2:
            continue
        members = [(str(row.mut_aa), int(index)) for index, row in representatives.iterrows()]
        panels.append((key, members))
    if not panels:
        raise RuntimeError("No multi-mutant single-site panels were found")

    variants = {
        "pairwise": "skempi_s1129_path_pairwise_seed{seed}",
        "soft_consistency": "skempi_s1129_path_soft_seed{seed}",
        "hard_antisymmetry": "skempi_s1129_path_hard_seed{seed}",
        "four_state_potential": "skempi_s1129_dssa5_four_state_potential_canonical_seed{seed}",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    raw_reverse_rows = []
    raw_triangle_rows = []
    raw_four_cycle_rows = []
    summaries = []
    thresholds = (args.threshold, args.strict_threshold)

    for seed in args.seeds:
        for variant, pattern in variants.items():
            reverse_errors, path_errors = [], []
            triangle_cycle_errors, four_state_cycle_errors = [], []
            for fold in sorted(data.fold_dssa_5.unique()):
                model_root = canonical_root if variant == "four_state_potential" else control_root
                checkpoint_path = model_root / pattern.format(seed=seed) / (
                    f"checkpoint_fold_dssa_5_fold{int(fold)}.pt"
                )
                if variant != "four_state_potential" and not checkpoint_path.exists():
                    checkpoint_path = fallback_control_root / pattern.format(seed=seed) / (
                        f"checkpoint_fold_dssa_5_fold{int(fold)}.pt"
                    )
                if not checkpoint_path.exists():
                    raise FileNotFoundError(checkpoint_path)
                model, checkpoint = load_model(checkpoint_path, torch.device(args.device))
                edge_graphs, edge_keys = [], []
                for key, members in panels:
                    if int(key[-1]) != int(fold):
                        continue
                    template = graphs[members[0][1]]
                    states = [(str(key[3]), members[0][1], False)] + [
                        (amino_acid, index, True) for amino_acid, index in members
                    ]
                    for left_index, right_index in permutations(range(len(states)), 2):
                        left_aa, left_row, left_mutant = states[left_index]
                        right_aa, right_row, right_mutant = states[right_index]
                        edge_graphs.append(make_edge(
                            template, graphs[left_row], left_mutant,
                            graphs[right_row], right_mutant, embedding_dim,
                        ))
                        edge_keys.append((key, left_aa, right_aa))
                edge_predictions = predict(
                    model, checkpoint, edge_graphs, args.batch_size, torch.device(args.device)
                )
                prediction_map = {
                    (tuple(key), left, right): float(value)
                    for (key, left, right), value in zip(edge_keys, edge_predictions)
                }
                for key, members in panels:
                    if int(key[-1]) != int(fold):
                        continue
                    amino_acids = [str(key[3])] + [amino_acid for amino_acid, _ in members]
                    for i in range(len(amino_acids)):
                        for j in range(i + 1, len(amino_acids)):
                            forward = prediction_map[(tuple(key), amino_acids[i], amino_acids[j])]
                            backward = prediction_map[(tuple(key), amino_acids[j], amino_acids[i])]
                            reverse_error = abs(forward + backward)
                            reverse_errors.append(reverse_error)
                            raw_reverse_rows.append({
                                "seed": seed, "variant": variant, "fold": int(fold),
                                "official_complex": key[0], "chain_id": key[1],
                                "pdb_pos": key[2], "left_aa": amino_acids[i],
                                "right_aa": amino_acids[j],
                                "forward_prediction": forward,
                                "backward_prediction": backward,
                                "reverse_error": reverse_error,
                            })
                    wt = amino_acids[0]
                    mutants = amino_acids[1:]
                    for first, second in permutations(mutants, 2):
                        direct = prediction_map[(tuple(key), wt, second)]
                        step_one = prediction_map[(tuple(key), wt, first)]
                        step_two = prediction_map[(tuple(key), first, second)]
                        error = abs(direct - step_one - step_two)
                        path_errors.append(error)
                        raw_rows.append({
                            "seed": seed, "variant": variant, "fold": int(fold),
                            "official_complex": key[0], "chain_id": key[1],
                            "pdb_pos": key[2], "wt_aa": wt,
                            "intermediate_aa": first, "final_aa": second,
                            "direct_prediction": direct,
                            "step_one_prediction": step_one,
                            "step_two_prediction": step_two,
                            "path_error": error,
                        })
                    # Closed-loop audits use one deterministic orientation for
                    # every state subset.  The state order is WT followed by
                    # mutant amino acids in canonical source-row order.  This
                    # prevents post-hoc selection of a favorable circulation.
                    for first, second, third in combinations(amino_acids, 3):
                        edge_one = prediction_map[(tuple(key), first, second)]
                        edge_two = prediction_map[(tuple(key), second, third)]
                        edge_three = prediction_map[(tuple(key), third, first)]
                        circulation = edge_one + edge_two + edge_three
                        error = abs(circulation)
                        triangle_cycle_errors.append(error)
                        raw_triangle_rows.append({
                            "seed": seed, "variant": variant, "fold": int(fold),
                            "official_complex": key[0], "chain_id": key[1],
                            "pdb_pos": key[2], "state_a": first,
                            "state_b": second, "state_c": third,
                            "edge_ab": edge_one, "edge_bc": edge_two,
                            "edge_ca": edge_three, "cycle_circulation": circulation,
                            "cycle_error": error,
                        })
                    for first, second, third, fourth in combinations(amino_acids, 4):
                        edge_one = prediction_map[(tuple(key), first, second)]
                        edge_two = prediction_map[(tuple(key), second, third)]
                        edge_three = prediction_map[(tuple(key), third, fourth)]
                        edge_four = prediction_map[(tuple(key), fourth, first)]
                        circulation = edge_one + edge_two + edge_three + edge_four
                        error = abs(circulation)
                        four_state_cycle_errors.append(error)
                        raw_four_cycle_rows.append({
                            "seed": seed, "variant": variant, "fold": int(fold),
                            "official_complex": key[0], "chain_id": key[1],
                            "pdb_pos": key[2], "state_a": first,
                            "state_b": second, "state_c": third,
                            "state_d": fourth, "edge_ab": edge_one,
                            "edge_bc": edge_two, "edge_cd": edge_three,
                            "edge_da": edge_four, "cycle_circulation": circulation,
                            "cycle_error": error,
                        })
                del model
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            summaries.append({
                "seed": seed, "variant": variant,
                "reverse_error": summarize(reverse_errors, thresholds),
                "path_error": summarize(path_errors, thresholds),
                "triangle_cycle_error": summarize(triangle_cycle_errors, thresholds),
                "four_state_cycle_error": summarize(four_state_cycle_errors, thresholds),
            })
            print(json.dumps(summaries[-1]), flush=True)

    pd.DataFrame(raw_rows).to_csv(output / "real_mutation_path_predictions.csv", index=False)
    pd.DataFrame(raw_reverse_rows).to_csv(
        output / "real_mutation_reverse_predictions.csv", index=False
    )
    pd.DataFrame(raw_triangle_rows).to_csv(
        output / "real_mutation_triangle_cycles.csv", index=False
    )
    pd.DataFrame(raw_four_cycle_rows).to_csv(
        output / "real_mutation_four_state_cycles.csv", index=False
    )
    mechanism_rows = []
    for row in summaries:
        flattened = {"seed": row["seed"], "variant": row["variant"]}
        for audit_name in (
                "reverse_error", "path_error", "triangle_cycle_error",
                "four_state_cycle_error"):
            for statistic, value in row[audit_name].items():
                flattened[f"{audit_name}_{statistic}"] = value
        mechanism_rows.append(flattened)
    pd.DataFrame(mechanism_rows).to_csv(
        output / "real_mutation_integrability_table.csv", index=False
    )
    payload = {
        "definition": "|y(A->C)-y(A->B)-y(B->C)| on ordered mutant pairs",
        "triangle_cycle_definition": "|y(A->B)+y(B->C)+y(C->A)|",
        "four_state_cycle_definition": "|y(A->B)+y(B->C)+y(C->D)+y(D->A)|",
        "cycle_enumeration": "one deterministic canonical orientation per state subset",
        "canonical_panels": len(panels),
        "ordered_paths_per_model_seed": int(sum(
            len(members) * (len(members) - 1) for _, members in panels
        )),
        "triangle_cycles_per_model_seed": int(sum(
            comb(len(members) + 1, 3) for _, members in panels
        )),
        "four_state_cycles_per_model_seed": int(sum(
            comb(len(members) + 1, 4) for _, members in panels
        )),
        "thresholds_kcal_per_mol": list(thresholds),
        "matched_control_features": "ESM + ProteinMPNN residue graph; no directional atomic delta or topology",
        "summaries": summaries,
    }
    (output / "real_mutation_path_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "summaries"}, indent=2))


if __name__ == "__main__":
    main()
