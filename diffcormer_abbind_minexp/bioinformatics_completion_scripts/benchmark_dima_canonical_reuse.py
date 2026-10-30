#!/usr/bin/env python
"""Benchmark native edge evaluation against explicit DiMA state-potential reuse.

The benchmark uses the held-out multi-endpoint S1129 panels.  Checkpoint and
cache loading, graph assembly, and disk I/O are excluded from timed regions;
the reported time is model inference only.  In the canonical arm, every
(complex, chain, site, amino-acid, bound/free) state is evaluated once under
its outer-fold checkpoint and its scalar value is reused on every edge.
"""
import argparse
import json
import sys
import time
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def evaluate_states(model, checkpoint, records, batch_size, device):
    """Return one calibrated binding potential per canonical amino-acid state."""
    potentials = {}
    target_mean = float(checkpoint["target_mean"])
    if abs(target_mean) > 1e-8:
        raise RuntimeError("State-potential checkpoint must use zero target mean")
    scale = float(checkpoint["target_std"])
    with torch.no_grad():
        for mutant_flag in (False, True):
            subset = [(key, graph) for key, graph, mutant in records if mutant == mutant_flag]
            for offset in range(0, len(subset), batch_size):
                chunk = subset[offset:offset + batch_size]
                batch = next(iter(DataLoader(
                    [graph for _, graph in chunk], batch_size=len(chunk), shuffle=False
                ))).to(device)
                bound = model.state_potential(batch, mutant=mutant_flag, bound=True)
                free = model.state_potential(batch, mutant=mutant_flag, bound=False)
                values = ((bound - free) * scale).detach().cpu().numpy()
                for (key, _), value in zip(chunk, values):
                    potentials[key] = float(value)
    return potentials


def error_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else 0.0,
        "maximum": float(values.max()) if values.size else 0.0,
    }


def consistency(edge_map, panels):
    reverse, path, triangles, four_cycles = [], [], [], []
    for panel_key, members in panels:
        amino_acids = [str(panel_key[3])] + [amino_acid for amino_acid, _ in members]
        for left, right in combinations(amino_acids, 2):
            reverse.append(abs(edge_map[(panel_key, left, right)] +
                               edge_map[(panel_key, right, left)]))
        wt, mutants = amino_acids[0], amino_acids[1:]
        for first, second in permutations(mutants, 2):
            path.append(abs(edge_map[(panel_key, wt, second)] -
                            edge_map[(panel_key, wt, first)] -
                            edge_map[(panel_key, first, second)]))
        for a, b, c in combinations(amino_acids, 3):
            triangles.append(abs(edge_map[(panel_key, a, b)] +
                                 edge_map[(panel_key, b, c)] +
                                 edge_map[(panel_key, c, a)]))
        for a, b, c, d in combinations(amino_acids, 4):
            four_cycles.append(abs(edge_map[(panel_key, a, b)] +
                                   edge_map[(panel_key, b, c)] +
                                   edge_map[(panel_key, c, d)] +
                                   edge_map[(panel_key, d, a)]))
    return {
        "reverse": error_summary(reverse),
        "path": error_summary(path),
        "triangle": error_summary(triangles),
        "four_state_cycle": error_summary(four_cycles),
    }


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.project_root))
    from evaluate_real_mutation_paths import load_model, make_edge, predict

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
        members = [(str(row.mut_aa), int(index))
                   for index, row in representatives.iterrows()]
        panels.append((tuple(key), members))

    device = torch.device(args.device)
    independent_edges, canonical_edges = {}, {}
    independent_seconds = 0.0
    canonical_seconds = 0.0
    unique_states = 0
    directed_edges = 0
    for fold in sorted(data.fold_dssa_5.unique()):
        checkpoint_path = (args.project_root / "outputs" /
                           f"skempi_s1129_dssa5_four_state_potential_canonical_seed{args.seed}" /
                           f"checkpoint_fold_dssa_5_fold{int(fold)}.pt")
        model, checkpoint = load_model(checkpoint_path, device)
        fold_panels = [(key, members) for key, members in panels if int(key[-1]) == int(fold)]

        edge_graphs, edge_keys = [], []
        state_records = []
        for key, members in fold_panels:
            template = graphs[members[0][1]]
            states = [(str(key[3]), members[0][1], False)] + [
                (amino_acid, index, True) for amino_acid, index in members
            ]
            for amino_acid, row_index, mutant in states:
                state_records.append(((key, amino_acid), graphs[row_index], mutant))
            for left_index, right_index in permutations(range(len(states)), 2):
                left_aa, left_row, left_mutant = states[left_index]
                right_aa, right_row, right_mutant = states[right_index]
                edge_graphs.append(make_edge(
                    template, graphs[left_row], left_mutant,
                    graphs[right_row], right_mutant, embedding_dim,
                ))
                edge_keys.append((key, left_aa, right_aa))

        started = time.perf_counter()
        independent_values = predict(model, checkpoint, edge_graphs, args.batch_size, device)
        independent_seconds += time.perf_counter() - started
        independent_edges.update(zip(edge_keys, map(float, independent_values)))

        started = time.perf_counter()
        potentials = evaluate_states(
            model, checkpoint, state_records, args.batch_size, device
        )
        canonical_seconds += time.perf_counter() - started
        for key, left, right in edge_keys:
            canonical_edges[(key, left, right)] = (
                potentials[(key, right)] - potentials[(key, left)]
            )
        unique_states += len(state_records)
        directed_edges += len(edge_keys)
        del model

    all_keys = sorted(independent_edges)
    absolute_differences = np.asarray([
        abs(independent_edges[key] - canonical_edges[key]) for key in all_keys
    ])
    result = {
        "seed": args.seed,
        "device": str(device),
        "torch_threads": args.threads,
        "panels": len(panels),
        "amino_acid_states": unique_states,
        "bound_free_state_contexts": 2 * unique_states,
        "directed_edges": directed_edges,
        "independent_state_context_requests": 4 * directed_edges,
        "request_reduction_factor": (4 * directed_edges) / (2 * unique_states),
        "inference_seconds": {
            "independent_edge_evaluation": independent_seconds,
            "canonical_state_reuse": canonical_seconds,
            "speedup": independent_seconds / canonical_seconds,
        },
        "independent_consistency": consistency(independent_edges, panels),
        "canonical_consistency": consistency(canonical_edges, panels),
        "prediction_difference": {
            "mean_absolute": float(absolute_differences.mean()),
            "maximum_absolute": float(absolute_differences.max()),
        },
        "timing_scope": "model inference only; excludes checkpoint/cache I/O and graph assembly",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
