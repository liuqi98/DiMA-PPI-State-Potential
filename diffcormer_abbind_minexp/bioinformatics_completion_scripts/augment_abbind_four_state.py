#!/usr/bin/env python
"""Generate WT/Mut x bound/free descriptors for single-site AbBind records.

AbBind writes mutations as ``chain:WTpositionMut`` whereas the original
SKEMPI helper expects ``WTchainpositionMut``.  This external-test adapter uses
the already-normalized chain, position, WT, and mutant columns directly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def repacked_bound(pyrosetta, row, scorefxn, pack_radius, pose_cache, helpers):
    from pyrosetta.toolbox.mutants import mutate_residue

    path = str(row["pdb_path"])
    if path not in pose_cache:
        pose_cache[path] = pyrosetta.pose_from_pdb(path)
    source = pose_cache[path]
    site = helpers.pose_index(source, row["chain_id"], row["pdb_pos"])
    observed = source.residue(site).name1()
    if observed != str(row["wt_aa"]):
        raise ValueError(
            f"WT mismatch for row {row['source_row']}: pose={observed}, csv={row['wt_aa']}"
        )
    wt_pose, mut_pose = source.clone(), source.clone()
    mutate_residue(wt_pose, site, str(row["wt_aa"]), pack_radius, scorefxn)
    if "disulfide" in mut_pose.residue(site).name().lower() and str(row["mut_aa"]) != "C":
        conformation = mut_pose.conformation()
        partner = int(pyrosetta.rosetta.core.conformation.get_disulf_partner(conformation, site))
        pyrosetta.rosetta.core.conformation.break_disulfide(conformation, site, partner)
    mutate_residue(mut_pose, site, str(row["mut_aa"]), pack_radius, scorefxn)
    return (
        wt_pose,
        mut_pose,
        helpers.pose_index(wt_pose, row["chain_id"], row["pdb_pos"]),
        helpers.pose_index(mut_pose, row["chain_id"], row["pdb_pos"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--pack-radius", type=float, default=8.0)
    parser.add_argument("--local-radius", type=float, default=12.0)
    parser.add_argument("--distance-bins", type=int, default=10)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.code_dir).resolve()))
    import augment_graph_cache_atomic_delta as atomic_helpers
    import augment_graph_cache_four_state as state_helpers
    import pyrosetta

    pyrosetta.init("-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false -ex1 -ex2aro")
    scorefxn = pyrosetta.get_fa_scorefxn()
    data = pd.read_csv(args.data_csv)
    base = torch.load(args.base_cache, map_location="cpu")
    if base.get("row_ids") != [int(value) for value in data.source_row]:
        raise RuntimeError("Base cache and AbBind table row IDs differ")

    bins = np.linspace(0.0, args.local_radius, args.distance_bins + 1)
    descriptor_dim = (
        2 * len(atomic_helpers.ELEMENTS) * args.distance_bins
        + len(atomic_helpers.ELEMENTS)
        + 2 * len(atomic_helpers.SCORE_TERMS)
    )
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    pose_cache, graphs = {}, []
    limit = len(data) if args.max_samples is None else min(args.max_samples, len(data))
    for index, (_, row) in enumerate(data.iterrows()):
        graph = base["graphs"][index].clone()
        state_path = state_dir / f"{int(row.source_row):04d}.npy"
        if state_path.exists():
            values = np.load(state_path)
            if values.shape != (4, descriptor_dim):
                raise RuntimeError(f"Invalid resumed descriptor: {state_path}")
            print(f"[{index + 1}/{limit}] row={int(row.source_row)} resumed", flush=True)
        elif index < limit:
            wb, mb, wb_site, mb_site = repacked_bound(
                pyrosetta, row, scorefxn, args.pack_radius, pose_cache, atomic_helpers
            )
            wu, mu, wu_site, mu_site = state_helpers.isolated_repacked_poses(
                pyrosetta, row, scorefxn, args.pack_radius, pose_cache
            )
            values = np.stack([
                state_helpers.state_descriptor(
                    pyrosetta, pose, site, scorefxn, bins, args.local_radius
                )
                for pose, site in ((wb, wb_site), (mb, mb_site), (wu, wu_site), (mu, mu_site))
            ]).astype(np.float32)
            np.save(state_path, values)
            print(f"[{index + 1}/{limit}] row={int(row.source_row)}", flush=True)
        else:
            values = np.zeros((4, descriptor_dim), dtype=np.float32)
        graph.state_descriptors = torch.from_numpy(values)[None]
        graph.state_descriptor_valid = torch.tensor([state_path.exists()])
        graphs.append(graph)

    valid_count = sum((state_dir / f"{int(value):04d}.npy").exists() for value in data.source_row)
    output = Path(args.output_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **{key: value for key, value in base.items() if key != "graphs"},
        "graphs": graphs,
        "state_descriptor_dim": descriptor_dim,
        "state_descriptor_order": ["wt_bound", "mut_bound", "wt_unbound", "mut_unbound"],
        "state_distance_bins": bins.tolist(),
        "state_score_terms": list(atomic_helpers.SCORE_TERMS),
        "state_pack_radius": args.pack_radius,
        "state_local_radius": args.local_radius,
        "state_descriptor_valid_count": valid_count,
        "state_descriptor_dir": str(state_dir),
    }, output)
    print(f"Wrote {len(graphs)} graphs ({valid_count} valid) to {output}")


if __name__ == "__main__":
    main()
