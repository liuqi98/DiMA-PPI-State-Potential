#!/usr/bin/env python
"""Canonicalize repeated AbBind state descriptors by physical state."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def state_key(row, amino_acid, phase):
    return (
        str(row["pdb_id"]),
        str(row["chain_id"]),
        str(row["pdb_pos"]),
        str(amino_acid),
        str(phase),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--audit-json", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data_csv)
    cache = torch.load(args.input_cache, map_location="cpu")
    expected = [int(value) for value in data.source_row]
    if cache.get("row_ids") != expected or len(cache["graphs"]) != len(data):
        raise RuntimeError("Input cache does not align with the AbBind table")

    candidates = defaultdict(list)
    for index, (_, row) in enumerate(data.iterrows()):
        values = cache["graphs"][index].state_descriptors.squeeze(0).numpy()
        for amino_acid, phase, state_index in (
            (row.wt_aa, "bound", 0),
            (row.mut_aa, "bound", 1),
            (row.wt_aa, "free", 2),
            (row.mut_aa, "free", 3),
        ):
            candidates[state_key(row, amino_acid, phase)].append(
                (int(row.source_row), values[state_index].copy())
            )

    canonical = {}
    duplicates = []
    for key, entries in candidates.items():
        entries.sort(key=lambda item: item[0])
        stacked = np.stack([value for _, value in entries])
        channel_range = float(np.max(np.ptp(stacked, axis=0))) if len(entries) > 1 else 0.0
        canonical[key] = entries[0][1]
        if len(entries) > 1:
            duplicates.append({
                "state_key": list(key),
                "occurrences": len(entries),
                "canonical_source_row": entries[0][0],
                "max_channel_range_before_canonicalization": channel_range,
            })

    graphs = []
    for index, (_, row) in enumerate(data.iterrows()):
        graph = cache["graphs"][index].clone()
        values = np.stack([
            canonical[state_key(row, row.wt_aa, "bound")],
            canonical[state_key(row, row.mut_aa, "bound")],
            canonical[state_key(row, row.wt_aa, "free")],
            canonical[state_key(row, row.mut_aa, "free")],
        ]).astype(np.float32)
        graph.state_descriptors = torch.from_numpy(values)[None]
        graph.state_descriptor_valid = torch.tensor([True])
        graphs.append(graph)

    output = Path(args.output_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **{key: value for key, value in cache.items() if key != "graphs"},
        "graphs": graphs,
        "state_canonical_key": ["pdb_id", "chain_id", "pdb_pos", "amino_acid", "phase"],
        "state_canonical_selection": "lowest source_row descriptor",
        "state_canonical_unique_keys": len(canonical),
        "state_canonical_duplicate_keys": len(duplicates),
    }, output)

    audit = {
        "rows": len(data),
        "unique_state_phase_keys": len(canonical),
        "duplicate_state_phase_keys": len(duplicates),
        "nonidentical_duplicate_keys_before_canonicalization": sum(
            row["max_channel_range_before_canonicalization"] > 0 for row in duplicates
        ),
        "maximum_channel_range_before_canonicalization": max(
            (row["max_channel_range_before_canonicalization"] for row in duplicates),
            default=0.0,
        ),
        "selection_rule": "reuse the lowest-source-row descriptor for each physical state",
        "duplicate_states": sorted(
            duplicates,
            key=lambda row: row["max_channel_range_before_canonicalization"],
            reverse=True,
        ),
    }
    audit_path = Path(args.audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "duplicate_states"}, indent=2))


if __name__ == "__main__":
    main()
