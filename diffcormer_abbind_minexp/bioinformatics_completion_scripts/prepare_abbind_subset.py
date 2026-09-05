#!/usr/bin/env python
"""Extract the decontaminated AbBind test partition and its cached graphs.

The historical external cache contains S1129 followed by AbBind.  The current
state-potential checkpoints are already trained on S1129, so only the AbBind
partition is needed to construct four-state descriptors for external testing.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--combined-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.combined_csv)
    selected = data.external_source.eq("AbBind_test").to_numpy()
    external = data.loc[selected].copy().reset_index(drop=True)
    cache = torch.load(args.combined_cache, map_location="cpu")
    if len(cache["graphs"]) != len(data):
        raise RuntimeError("The combined graph cache does not match the data table")

    graphs = [graph for graph, keep in zip(cache["graphs"], selected) if keep]
    expected_ids = [int(value) for value in external.source_row]
    observed_ids = [int(graph.row_id.view(-1)[0]) for graph in graphs]
    if observed_ids != expected_ids:
        raise RuntimeError("Selected graph row IDs do not match the AbBind table")

    external.to_csv(output / "abbind_decontaminated.csv", index=False)
    torch.save({
        **{key: value for key, value in cache.items() if key != "graphs"},
        "graphs": graphs,
        "row_ids": expected_ids,
        "split_source": "decontaminated AbBind external test only",
    }, output / "abbind_base_graphs.pt")
    summary = {
        "rows": len(external),
        "complexes": int(external.pdb_id.nunique()),
        "source_row_min": min(expected_ids),
        "source_row_max": max(expected_ids),
        "label": "DSSA convention after sign conversion from AbBind",
    }
    (output / "abbind_subset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
