#!/usr/bin/env python
"""Merge canonical S1129 and AbBind graphs for train-all/external-test runs."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--skempi-csv", required=True)
    parser.add_argument("--skempi-cache", required=True)
    parser.add_argument("--abbind-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.combined_csv)
    skempi_data = pd.read_csv(args.skempi_csv)
    skempi = torch.load(args.skempi_cache, map_location="cpu")
    abbind = torch.load(args.abbind_cache, map_location="cpu")
    train_rows = data.external_source.eq("S1129_train")
    test_rows = data.external_source.eq("AbBind_test")
    train_ids = data.loc[train_rows, "source_row"].astype(int).tolist()
    test_ids = data.loc[test_rows, "source_row"].astype(int).tolist()
    if skempi.get("row_ids") != skempi_data.source_row.astype(int).tolist():
        raise RuntimeError("Canonical S1129 cache does not align with its source table")
    combined_train = data.loc[train_rows].reset_index(drop=True)
    source_train = skempi_data.reset_index(drop=True)
    for column in ("pdb_id", "official_complex", "official_mutation", "ddG"):
        left = combined_train[column].fillna("NA").astype(str).tolist()
        right = source_train[column].fillna("NA").astype(str).tolist()
        if left != right:
            raise RuntimeError(f"S1129 row order differs in column {column}")
    if abbind.get("row_ids") != test_ids:
        raise RuntimeError("Canonical AbBind cache does not align with the combined table")
    train_graphs = []
    for graph, row_id in zip(skempi["graphs"], train_ids):
        graph = graph.clone()
        graph.row_id = torch.tensor([row_id], dtype=torch.long)
        train_graphs.append(graph)
    graphs = train_graphs + list(abbind["graphs"])
    row_ids = train_ids + test_ids
    if row_ids != data.source_row.astype(int).tolist():
        raise RuntimeError("Combined graph order differs from the external-test table")

    output = Path(args.output_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **{key: value for key, value in skempi.items() if key != "graphs"},
        "graphs": graphs,
        "row_ids": row_ids,
        "state_descriptor_valid_count": len(graphs),
        "split_source": "canonical S1129 training graphs plus canonical decontaminated AbBind test graphs",
    }, output)
    summary = {
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "total_rows": len(graphs),
        "state_descriptor_dim": int(graphs[0].state_descriptors.shape[-1]),
        "output_cache": str(output.resolve()),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
