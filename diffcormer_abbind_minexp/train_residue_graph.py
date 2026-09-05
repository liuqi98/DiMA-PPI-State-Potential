#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

import train_esm_delta_mlp as pooled


def rbf_distance(distance, bins=16, max_distance=20.0):
    centers = torch.linspace(0.0, max_distance, bins, dtype=distance.dtype)
    width = max_distance / (bins - 1)
    return torch.exp(-((distance[:, None] - centers[None, :]) / width) ** 2)


def make_graph(row, wt, mut, coord_cache, node_radius, edge_radius):
    coords, chains = pooled.row_coordinates(row, coord_cache)
    if "mut_indices" in row.index and pd.notna(row["mut_indices"]):
        sites = [int(value) for value in json.loads(row["mut_indices"])]
    else:
        sites = [int(row["mut_index"])]
    if not sites:
        raise ValueError(f"Row {row['source_row']} has no mutation sites")
    site_distance = np.linalg.norm(
        coords[:, None, :] - coords[np.asarray(sites)][None, :, :], axis=2
    ).min(axis=1)
    selected = np.flatnonzero(site_distance <= node_radius)
    selected = np.unique(np.append(selected, sites))
    old_to_new = {int(old): i for i, old in enumerate(selected)}
    local_coords = coords[selected]
    local_chains = chains[selected]
    local_distance = np.linalg.norm(
        local_coords[:, None, :] - local_coords[None, :, :], axis=-1
    )
    src, dst = np.nonzero((local_distance <= edge_radius) & (local_distance > 0))
    edges = {(int(a), int(b)): float(local_distance[a, b]) for a, b in zip(src, dst)}
    # Retain covalent context even when a structure has an unusual coordinate gap.
    for old in selected:
        nxt = int(old) + 1
        if nxt in old_to_new and chains[int(old)] == chains[nxt]:
            a, b = old_to_new[int(old)], old_to_new[nxt]
            d = float(np.linalg.norm(coords[int(old)] - coords[nxt]))
            edges[(a, b)] = d
            edges[(b, a)] = d
    pairs = sorted(edges)
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    distances = torch.tensor([edges[pair] for pair in pairs], dtype=torch.float32)
    edge_attr = rbf_distance(distances)

    selected_t = torch.from_numpy(selected).long()
    wt_local = wt[selected_t]
    mut_local = mut[selected_t]
    delta = mut_local - wt_local
    mutation_mask = torch.zeros(len(selected), dtype=torch.bool)
    for site in sites:
        mutation_mask[old_to_new[site]] = True
    if "chain_ids" in row.index and pd.notna(row["chain_ids"]):
        mutation_chains = {str(value) for value in json.loads(row["chain_ids"])}
    else:
        mutation_chains = {str(row["chain_id"])}
    same_chain = torch.tensor(
        np.isin(local_chains, list(mutation_chains)), dtype=torch.float32
    )
    other_chain = 1.0 - same_chain
    distance_scalar = torch.tensor(site_distance[selected] / node_radius, dtype=torch.float32)
    scalar = torch.stack(
        [same_chain, other_chain, mutation_mask.float(), distance_scalar], dim=1
    )
    x = torch.cat([wt_local, mut_local, delta, scalar], dim=1).float()
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(row["ddG"])], dtype=torch.float32),
        mutation_mask=mutation_mask,
        other_chain_mask=other_chain.bool(),
        edge_same_chain=torch.tensor(
            local_chains[src] == local_chains[dst], dtype=torch.bool
        ),
        row_id=torch.tensor([int(row["source_row"])], dtype=torch.long),
    )


def build_graphs(df, args):
    cache_path = Path(args.cache_path)
    row_ids = [int(v) for v in df["source_row"]]
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("row_ids") == row_ids
            and cached.get("node_radius") == args.node_radius
            and cached.get("edge_radius") == args.edge_radius
        ):
            graphs = cached["graphs"]
            topology_dim = int(graphs[0].topology.shape[-1]) if hasattr(graphs[0], "topology") else 0
            atomic_delta_dim = int(graphs[0].atomic_delta.shape[-1]) if hasattr(graphs[0], "atomic_delta") else 0
            if atomic_delta_dim > 0:
                if args.atomic_delta_view == "summary":
                    for graph in graphs:
                        graph.atomic_delta = graph.atomic_delta[:, -30:]
                elif args.atomic_delta_view == "energy":
                    for graph in graphs:
                        graph.atomic_delta = graph.atomic_delta[:, -19:]
                # Remove descriptor channels that are constant for the entire
                # dataset. This is target-free and avoids needless parameters
                # for impossible element/bin combinations.
                atomic = torch.cat([graph.atomic_delta for graph in graphs], dim=0)
                keep = atomic.std(dim=0) > 1e-8
                if not bool(keep.all()):
                    for graph in graphs:
                        graph.atomic_delta = graph.atomic_delta[:, keep]
                    atomic_delta_dim = int(keep.sum())
            state_descriptor_dim = (int(graphs[0].state_descriptors.shape[-1])
                                    if hasattr(graphs[0], "state_descriptors") else 0)
            if (state_descriptor_dim > 0
                    and int(cached.get("state_descriptor_valid_count", len(graphs))) != len(graphs)):
                raise RuntimeError(
                    "Four-state cache is partial and may only be used for descriptor smoke tests"
                )
            if args.feature_ablation in ("no_mpnn", "esm_only"):
                # x = [WT ESM, Mut ESM, delta ESM, ProteinMPNN, four scalars].
                embedding_dim = int(cached["embedding_dim"])
                for graph in graphs:
                    graph.x = torch.cat([
                        graph.x[:, :3 * embedding_dim], graph.x[:, -4:]
                    ], dim=1)
            if args.feature_ablation in ("no_topology", "no_auxiliary", "esm_only"):
                for graph in graphs:
                    if hasattr(graph, "topology"):
                        del graph.topology
                topology_dim = 0
            if args.feature_ablation in ("no_atomic", "no_auxiliary", "esm_only"):
                for graph in graphs:
                    if hasattr(graph, "atomic_delta"):
                        del graph.atomic_delta
                atomic_delta_dim = 0
            return (graphs, int(cached["embedding_dim"]), int(graphs[0].x.shape[1]),
                    topology_dim, atomic_delta_dim, state_descriptor_dim)
        print(f"Ignoring stale graph cache: {cache_path}")

    model, alphabet = pooled.load_esm(args.model_path, torch.device(args.device))
    wt_sequences = list(dict.fromkeys(df["wt_sequence"].tolist()))
    wt_reps = pooled.embed_sequences(
        model, alphabet, wt_sequences, args.esm_batch_size, torch.device(args.device)
    )
    wt_map = dict(zip(wt_sequences, wt_reps))
    mut_reps = pooled.embed_sequences(
        model,
        alphabet,
        df["mutated_sequence"].tolist(),
        args.esm_batch_size,
        torch.device(args.device),
    )
    coord_cache = {}
    graphs = []
    for i, (_, row) in enumerate(df.iterrows()):
        graphs.append(
            make_graph(
                row,
                wt_map[row["wt_sequence"]],
                mut_reps[i],
                coord_cache,
                args.node_radius,
                args.edge_radius,
            )
        )
    embedding_dim = int(wt_reps[0].shape[1])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graphs": graphs,
            "row_ids": row_ids,
            "embedding_dim": embedding_dim,
            "node_radius": args.node_radius,
            "edge_radius": args.edge_radius,
        },
        cache_path,
    )
    return graphs, embedding_dim, int(graphs[0].x.shape[1]), 0, 0, 0


class ResidueGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, layers=3, dropout=0.15,
                 input_fusion="concat", sequence_dim=None, structure_dim=0,
                 readout="attention", pma_seeds=4, topology_dim=0,
                 atomic_delta_dim=0, state_descriptor_dim=0):
        super().__init__()
        self.input_fusion = input_fusion
        self.sequence_dim = sequence_dim
        self.structure_dim = structure_dim
        self.readout = readout
        self.pma_seeds = pma_seeds
        self.topology_dim = topology_dim
        self.atomic_delta_dim = atomic_delta_dim
        self.state_descriptor_dim = state_descriptor_dim
        if sequence_dim is None or sequence_dim % 3:
            raise ValueError("sequence_dim must contain WT, mutant, and delta blocks")
        self.embedding_dim = sequence_dim // 3
        if input_fusion == "gated":
            if sequence_dim is None or structure_dim < 0 or sequence_dim + structure_dim + 4 != input_dim:
                raise ValueError("Gated fusion requires aligned sequence, optional structure, and 4 scalar dimensions")
            self.sequence_input = nn.Sequential(
                nn.Linear(sequence_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
            )
            self.scalar_input = nn.Linear(4, hidden_dim)
            if structure_dim > 0:
                self.structure_input = nn.Sequential(
                    nn.Linear(structure_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
                )
                self.structure_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.Sigmoid()
                )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        else:
            self.input = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
            )
        self.layers = nn.ModuleList(
            [TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=16, dropout=dropout)
             for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        if readout == "pma":
            self.pool_seeds = nn.Parameter(torch.empty(pma_seeds, hidden_dim))
            nn.init.xavier_uniform_(self.pool_seeds)
            self.pool_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.pool_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.pool_norm = nn.LayerNorm(hidden_dim)
            self.pool_ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            pooled_dim = hidden_dim * pma_seeds
        else:
            self.pool_score = nn.Linear(hidden_dim, 1)
            pooled_dim = hidden_dim
        if topology_dim > 0:
            self.topology_input = nn.Sequential(
                nn.Linear(topology_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.topology_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
            )
            topology_head_dim = hidden_dim
        else:
            topology_head_dim = 0
        if atomic_delta_dim > 0:
            self.atomic_delta_input = nn.Sequential(
                nn.LayerNorm(atomic_delta_dim), nn.Linear(atomic_delta_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            )
            self.atomic_delta_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
            )
            atomic_delta_head_dim = hidden_dim
        else:
            atomic_delta_head_dim = 0
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + pooled_dim + topology_head_dim + atomic_delta_head_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        # A shared scalar state potential.  Four evaluations of this head form
        # an exact thermodynamic cycle, rather than asking a regression head to
        # learn antisymmetry from augmented examples.
        if state_descriptor_dim > 0:
            self.state_descriptor_input = nn.Sequential(
                nn.LayerNorm(state_descriptor_dim),
                nn.Linear(state_descriptor_dim, hidden_dim), nn.SiLU(),
            )
            state_extra_dim = hidden_dim
        else:
            state_extra_dim = 0
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + pooled_dim + state_extra_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def pma_pool(self, x, batch_index, mask):
        keys = self.pool_key(x)
        values = self.pool_value(x)
        outputs = []
        scale = x.shape[1] ** -0.5
        for seed in self.pool_seeds:
            score = (keys * seed[None]).sum(dim=1) * scale
            score = score.masked_fill(~mask, -1e4)
            weight = softmax(score, batch_index) * mask.float()
            denominator = scatter_sum(weight, batch_index, dim=0).clamp(min=1e-8)
            weight = weight / denominator[batch_index]
            pooled = scatter_sum(weight[:, None] * values, batch_index, dim=0)
            pooled = self.pool_norm(seed[None] + pooled)
            outputs.append(pooled + self.pool_ff(pooled))
        return torch.cat(outputs, dim=1)

    def orthogonality_loss(self):
        if self.readout != "pma" or self.pma_seeds <= 1:
            return self.pool_seeds.new_tensor(0.0) if self.readout == "pma" else next(self.parameters()).new_tensor(0.0)
        seeds = F.normalize(self.pool_seeds, dim=1)
        similarity = seeds @ seeds.t()
        return ((similarity - torch.eye(self.pma_seeds, device=seeds.device)) ** 2).mean()

    def encode_nodes(self, batch, x_input=None, same_chain_only=False):
        node_input = batch.x if x_input is None else x_input
        if self.input_fusion == "gated":
            sequence = self.sequence_input(node_input[:, :self.sequence_dim])
            scalars = node_input[:, -4:]
            x = sequence + self.scalar_input(scalars)
            if self.structure_dim > 0:
                structure_start = self.sequence_dim
                structure = self.structure_input(
                    node_input[:, structure_start:structure_start + self.structure_dim]
                )
                gate = self.structure_gate(torch.cat([sequence, structure, scalars], dim=1))
                x = x + gate * structure
            x = self.fusion_norm(x)
        else:
            x = self.input(node_input)
        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        if same_chain_only:
            if hasattr(batch, "edge_same_chain"):
                keep = batch.edge_same_chain
            else:
                # Cached graphs predate edge_same_chain. The first scalar is
                # membership in the mutated chain; for the two-chain S1131
                # complexes this exactly identifies inter-chain edges.
                src, dst = edge_index
                keep = batch.x[src, -4] == batch.x[dst, -4]
            edge_index, edge_attr = edge_index[:, keep], edge_attr[keep]
        for conv, norm in zip(self.layers, self.norms):
            x = norm(x + F.silu(conv(x, edge_index, edge_attr)))
        return x

    def pool_nodes(self, x, batch, pool_mask):
        if self.readout == "pma":
            return self.pma_pool(x, batch.batch, pool_mask)
        score = self.pool_score(x).squeeze(-1)
        score = score.masked_fill(~pool_mask, -1e4)
        weight = softmax(score, batch.batch) * pool_mask.float()
        denominator = scatter_sum(weight, batch.batch, dim=0).clamp(min=1e-8)
        weight = weight / denominator[batch.batch]
        return scatter_sum(weight[:, None] * x, batch.batch, dim=0)

    def state_features(self, batch, same_chain_only=False):
        x = self.encode_nodes(batch, same_chain_only=same_chain_only)
        # Pool all substituted residues so a joint multi-point mutant remains
        # one thermodynamic observation. For single mutants this is identical
        # to the historical direct indexing behavior.
        mutation = self.pool_nodes(x, batch, batch.mutation_mask)
        pooled_other = self.pool_nodes(x, batch, batch.other_chain_mask)
        return mutation, pooled_other

    def single_state_input(self, batch, mutant):
        """Replace pair features by one sequence state without direction cues."""
        start = self.embedding_dim if mutant else 0
        state = batch.x[:, start:start + self.embedding_dim]
        neutral_sequence = torch.cat([state, state, torch.zeros_like(state)], dim=1)
        return torch.cat([neutral_sequence, batch.x[:, self.sequence_dim:]], dim=1)

    def state_potential(self, batch, mutant, bound):
        node_input = self.single_state_input(batch, mutant)
        encoded = self.encode_nodes(batch, node_input, same_chain_only=not bound)
        mutation = self.pool_nodes(encoded, batch, batch.mutation_mask)
        if bound:
            context_mask = ~batch.mutation_mask
        else:
            # The isolated state contains only the mutated chain.  Partner
            # nodes are excluded both from message passing and readout.
            context_mask = (~batch.other_chain_mask) & (~batch.mutation_mask)
        context = self.pool_nodes(encoded, batch, context_mask)
        features = [mutation, context]
        if self.state_descriptor_dim > 0:
            state_index = (2 if not bound else 0) + (1 if mutant else 0)
            descriptor = batch.state_descriptors[:, state_index, :]
            features.append(self.state_descriptor_input(descriptor))
        return self.state_head(torch.cat(features, dim=1)).squeeze(-1)

    def thermodynamic_prediction(self, batch):
        wt_bound = self.state_potential(batch, mutant=False, bound=True)
        mut_bound = self.state_potential(batch, mutant=True, bound=True)
        wt_unbound = self.state_potential(batch, mutant=False, bound=False)
        mut_unbound = self.state_potential(batch, mutant=True, bound=False)
        return (mut_bound - wt_bound) - (mut_unbound - wt_unbound)

    def auxiliary_features(self, batch, mutation):
        features = []
        if self.topology_dim > 0:
            topology = self.topology_input(batch.topology)
            topology = self.topology_gate(torch.cat([mutation, topology], dim=1)) * topology
            features.append(topology)
        if self.atomic_delta_dim > 0:
            atomic_delta = self.atomic_delta_input(batch.atomic_delta)
            atomic_delta = self.atomic_delta_gate(
                torch.cat([mutation, atomic_delta], dim=1)
            ) * atomic_delta
            features.append(atomic_delta)
        return features

    def forward(self, batch, thermodynamic_cycle=False, cycle_mode=None):
        if cycle_mode is None:
            cycle_mode = "message_difference" if thermodynamic_cycle else "none"
        if cycle_mode == "state_potential":
            return self.thermodynamic_prediction(batch)
        bound_mutation, bound_partner = self.state_features(batch, same_chain_only=False)
        if cycle_mode == "message_difference":
            free_mutation, free_partner = self.state_features(batch, same_chain_only=True)
            features = [bound_mutation - free_mutation, bound_partner - free_partner]
        else:
            features = [bound_mutation, bound_partner]
        features.extend(self.auxiliary_features(batch, bound_mutation))
        return self.head(torch.cat(features, dim=1)).squeeze(-1)


def metrics(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    output = {
        "pearson": float(pearsonr(pred, target).statistic),
        "spearman": float(spearmanr(pred, target).correlation),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
        "mae": float(np.mean(np.abs(pred - target))),
    }
    classes = target > 0
    output["auroc"] = (float(roc_auc_score(classes, pred))
                       if np.unique(classes).size == 2 else float("nan"))
    return output


def correlation_loss(pred, target):
    pred = pred - pred.mean()
    target = target - target.mean()
    denominator = pred.square().sum().sqrt() * target.square().sum().sqrt()
    return 1.0 - (pred * target).sum() / denominator.clamp(min=1e-8)


def reverse_mutation_batch(batch, embedding_dim):
    """Construct the WT<->Mut direction without changing graph geometry."""
    reverse = batch.clone()
    wt = batch.x[:, :embedding_dim]
    mut = batch.x[:, embedding_dim:2 * embedding_dim]
    delta = batch.x[:, 2 * embedding_dim:3 * embedding_dim]
    reverse.x = batch.x.clone()
    reverse.x[:, :embedding_dim] = mut
    reverse.x[:, embedding_dim:2 * embedding_dim] = wt
    reverse.x[:, 2 * embedding_dim:3 * embedding_dim] = -delta
    if hasattr(batch, "atomic_delta"):
        reverse.atomic_delta = -batch.atomic_delta
    if hasattr(batch, "state_descriptors"):
        reverse.state_descriptors = batch.state_descriptors[:, [1, 0, 3, 2], :].clone()
    return reverse


def train_fold(graphs, train_idx, test_idx, args, input_dim, embedding_dim, topology_dim,
               atomic_delta_dim, state_descriptor_dim, pdb_ids=None, checkpoint_path=None):
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    if args.validation_grouping == "complex":
        # Hold out whole complexes so validation targets share no complex with the fit set.
        complexes = np.unique(pdb_ids[train_idx])
        order = torch.randperm(len(complexes), generator=generator).numpy()
        target = max(8, len(train_idx) // 8)
        val_mask = np.zeros(len(train_idx), dtype=bool)
        picked = 0
        for c in complexes[order]:
            member = pdb_ids[train_idx] == c
            val_mask |= member
            picked += int(member.sum())
            if picked >= target:
                break
        val_idx, fit_idx = train_idx[val_mask], train_idx[~val_mask]
    else:
        order = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
        n_val = max(8, len(order) // 8)
        val_idx, fit_idx = order[:n_val], order[n_val:]
    train_y = torch.tensor([graphs[i].y.item() for i in fit_idx])
    target_mean, target_std = train_y.mean(), train_y.std().clamp(min=1e-6)
    if args.cycle_mode == "state_potential" or args.hard_antisymmetry:
        # An additive target offset would break exact reverse antisymmetry when
        # predictions are transformed back to kcal/mol.  Scaling is safe.
        target_mean = train_y.new_tensor(0.0)

    fit_loader = DataLoader(Subset(graphs, fit_idx.tolist()), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(graphs, val_idx.tolist()), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(graphs, test_idx.tolist()), batch_size=args.batch_size)
    torch.manual_seed(args.seed)
    sequence_dim = embedding_dim * 3
    structure_dim = input_dim - sequence_dim - 4
    model = ResidueGraphModel(
        input_dim, args.hidden_dim, args.layers, args.dropout,
        input_fusion=args.input_fusion,
        sequence_dim=sequence_dim,
        structure_dim=structure_dim,
        readout=args.readout,
        pma_seeds=args.pma_seeds,
        topology_dim=topology_dim,
        atomic_delta_dim=atomic_delta_dim,
        state_descriptor_dim=state_descriptor_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_score, stale = None, -float("inf"), 0
    for _ in range(args.epochs):
        model.train()
        for batch in fit_loader:
            batch = batch.to(device)
            raw_pred = model(batch, cycle_mode=args.cycle_mode)
            reverse = None
            reverse_pred = None
            if args.hard_antisymmetry:
                reverse = reverse_mutation_batch(batch, embedding_dim)
                raw_reverse_pred = model(reverse, cycle_mode=args.cycle_mode)
                pred = 0.5 * (raw_pred - raw_reverse_pred)
                reverse_pred = -pred
            else:
                pred = raw_pred
            target = (batch.y.view(-1) - target_mean.to(device)) / target_std.to(device)
            loss = F.smooth_l1_loss(pred, target, beta=args.huber_beta)
            if ((args.reverse_supervision_weight > 0 or args.cycle_consistency_weight > 0)
                    and not args.hard_antisymmetry):
                reverse = reverse_mutation_batch(batch, embedding_dim)
                reverse_pred = model(reverse, cycle_mode=args.cycle_mode)
                scale = target_std.to(device)
                if args.reverse_supervision_weight > 0:
                    reverse_target = (-batch.y.view(-1) - target_mean.to(device)) / scale
                    reverse_loss = F.smooth_l1_loss(
                        reverse_pred, reverse_target, beta=args.huber_beta
                    )
                    loss = loss + args.reverse_supervision_weight * reverse_loss
                if args.cycle_consistency_weight > 0:
                    cycle_loss = ((pred * scale + reverse_pred * scale + 2 * target_mean.to(device)) ** 2).mean()
                    loss = loss + args.cycle_consistency_weight * cycle_loss / scale.square()
            if args.pearson_loss_weight > 0 and batch.num_graphs >= 3:
                loss = loss + args.pearson_loss_weight * correlation_loss(pred, target)
            if args.readout == "pma":
                loss = loss + args.pma_orth_weight * model.orthogonality_loss()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_predictions, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                raw_prediction = model(batch, cycle_mode=args.cycle_mode)
                if args.hard_antisymmetry:
                    reverse = reverse_mutation_batch(batch, embedding_dim)
                    raw_reverse = model(reverse, cycle_mode=args.cycle_mode)
                    raw_prediction = 0.5 * (raw_prediction - raw_reverse)
                prediction = raw_prediction.cpu() * target_std + target_mean
                val_predictions.extend(prediction.tolist())
                val_targets.extend(batch.y.view(-1).cpu().tolist())
        val_metric = metrics(val_predictions, val_targets)
        if args.early_stop_metric == "mse":
            score = -(val_metric["rmse"] ** 2)
        elif args.early_stop_metric == "pearson":
            score = val_metric["pearson"]
        else:
            target_scale = max(float(np.std(val_targets)), 1e-6)
            score = val_metric["pearson"] - args.early_stop_rmse_weight * val_metric["rmse"] / target_scale
        if score > best_score:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "input_dim": input_dim,
            "embedding_dim": embedding_dim,
            "topology_dim": topology_dim,
            "atomic_delta_dim": atomic_delta_dim,
            "state_descriptor_dim": state_descriptor_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "input_fusion": args.input_fusion,
            "readout": args.readout,
            "pma_seeds": args.pma_seeds,
            "cycle_mode": args.cycle_mode,
            "hard_antisymmetry": args.hard_antisymmetry,
        }, checkpoint_path)
    predictions, reverse_predictions, targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            reverse = reverse_mutation_batch(batch, embedding_dim)
            raw_prediction = model(batch, cycle_mode=args.cycle_mode)
            raw_reverse = model(reverse, cycle_mode=args.cycle_mode)
            if args.hard_antisymmetry:
                raw_prediction = 0.5 * (raw_prediction - raw_reverse)
                raw_reverse = -raw_prediction
            predictions.extend((raw_prediction.cpu() * target_std + target_mean).tolist())
            reverse_predictions.extend((raw_reverse.cpu() * target_std + target_mean).tolist())
            targets.extend(batch.y.view(-1).cpu().tolist())
    return metrics(predictions, targets), predictions, reverse_predictions, targets, float(best_score)


def run(args):
    pooled.patch_broken_dynamo()
    df = pd.read_csv(args.data_csv)
    if args.smoke:
        df = df.groupby("pdb_id", group_keys=False).head(args.smoke_per_complex).reset_index(drop=True)
    (graphs, embedding_dim, input_dim, topology_dim, atomic_delta_dim,
     state_descriptor_dim) = build_graphs(df, args)
    if args.cache_only:
        print(json.dumps({"cache_only": True, "rows": len(df), "embedding_dim": embedding_dim,
                          "node_input_dim": input_dim, "topology_dim": topology_dim,
                          "atomic_delta_dim": atomic_delta_dim}, indent=2), flush=True)
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.save_fold_checkpoints and args.nested_anti_weights:
        raise ValueError("Fold checkpoint export is not implemented for nested weight selection")
    results = []
    oof_rows = []
    for split in args.splits:
        folds = sorted(df[split].unique())
        if args.only_fold is not None:
            folds = [args.only_fold]
        if args.smoke:
            folds = folds[:2]
        values = df[split].to_numpy()
        pdb_ids = df["pdb_id"].to_numpy()
        for fold in folds:
            test_idx = np.flatnonzero(values == fold)
            train_idx = np.flatnonzero(values != fold)
            selected_weight = None
            if args.nested_anti_weights:
                candidates = []
                original_weight = args.cycle_consistency_weight
                for weight in args.nested_anti_weights:
                    args.cycle_consistency_weight = float(weight)
                    candidate = train_fold(
                        graphs, train_idx, test_idx, args, input_dim, embedding_dim,
                        topology_dim, atomic_delta_dim, state_descriptor_dim,
                        pdb_ids=pdb_ids,
                    )
                    candidates.append((candidate[-1], float(weight), candidate))
                    print({"nested_fold": int(fold), "candidate_weight": float(weight),
                           "inner_validation_score": float(candidate[-1])}, flush=True)
                args.cycle_consistency_weight = original_weight
                _, selected_weight, chosen = max(candidates, key=lambda item: item[0])
                print({"nested_fold": int(fold), "selected_anti_weight": selected_weight},
                      flush=True)
                result, predictions, reverse_predictions, targets, _ = chosen
            else:
                checkpoint_path = (
                    output / f"checkpoint_{split}_fold{int(fold)}.pt"
                    if args.save_fold_checkpoints else None
                )
                result, predictions, reverse_predictions, targets, _ = train_fold(
                    graphs, train_idx, test_idx, args, input_dim, embedding_dim,
                    topology_dim, atomic_delta_dim, state_descriptor_dim,
                    pdb_ids=pdb_ids, checkpoint_path=checkpoint_path,
                )
            result.update({"split": split, "fold": int(fold), "n_train": len(train_idx), "n_test": len(test_idx)})
            if selected_weight is not None:
                result["selected_anti_weight"] = selected_weight
            results.append(result)
            for dataset_index, prediction, reverse_prediction, target in zip(
                    test_idx, predictions, reverse_predictions, targets):
                oof_rows.append({
                    "split": split,
                    "fold": int(fold),
                    "dataset_index": int(dataset_index),
                    "source_row": int(df.iloc[int(dataset_index)]["source_row"]),
                    "target": float(target),
                    "prediction": float(prediction),
                    "reverse_prediction": float(reverse_prediction),
                    "selected_anti_weight": selected_weight,
                })
            print(result, flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / ("smoke_results.csv" if args.smoke else "results.csv"), index=False)
    pd.DataFrame(oof_rows).to_csv(
        output / ("smoke_oof_predictions.csv" if args.smoke else "oof_predictions.csv"),
        index=False,
    )
    grouped = frame.groupby("split")[["pearson", "spearman", "rmse", "mae", "auroc"]]
    summary = {
        "rows": len(df), "complexes": int(df.pdb_id.nunique()), "device": args.device,
        "embedding_dim": embedding_dim, "node_input_dim": input_dim,
        "topology_dim": topology_dim,
        "atomic_delta_dim": atomic_delta_dim,
        "state_descriptor_dim": state_descriptor_dim,
        "atomic_delta_view": args.atomic_delta_view,
        "node_radius": args.node_radius, "edge_radius": args.edge_radius,
        "seed": args.seed, "input_fusion": args.input_fusion,
        "readout": args.readout, "pma_seeds": args.pma_seeds,
        "pma_orth_weight": args.pma_orth_weight,
        "pearson_loss_weight": args.pearson_loss_weight,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_rmse_weight": args.early_stop_rmse_weight,
        "thermodynamic_cycle": args.thermodynamic_cycle,
        "cycle_mode": args.cycle_mode,
        "antisymmetry_weight": args.antisymmetry_weight,
        "reverse_supervision_weight": args.reverse_supervision_weight,
        "cycle_consistency_weight": args.cycle_consistency_weight,
        "hard_antisymmetry": args.hard_antisymmetry,
        "feature_ablation": args.feature_ablation,
        "validation_grouping": args.validation_grouping,
        "nested_anti_weights": args.nested_anti_weights,
        "mean_by_split": grouped.mean().to_dict(), "std_by_split": grouped.std().to_dict(),
    }
    path = output / ("smoke_summary.json" if args.smoke else "summary.json")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--esm-batch-size", type=int, default=2)
    parser.add_argument("--node-radius", type=float, default=20.0)
    parser.add_argument("--edge-radius", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--pearson-loss-weight", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", choices=["mse", "pearson", "hybrid"], default="mse")
    parser.add_argument("--early-stop-rmse-weight", type=float, default=0.05)
    parser.add_argument("--thermodynamic-cycle", action="store_true")
    parser.add_argument(
        "--cycle-mode", choices=["none", "message_difference", "state_potential"],
        default=None,
        help="Use the legacy message-difference approximation or an exact shared-potential cycle.",
    )
    parser.add_argument("--antisymmetry-weight", type=float, default=0.0)
    parser.add_argument("--reverse-supervision-weight", type=float)
    parser.add_argument("--cycle-consistency-weight", type=float)
    parser.add_argument(
        "--hard-antisymmetry", action="store_true",
        help="Use (f(WT,Mut)-f(Mut,WT))/2 as an exactly antisymmetric predictor.",
    )
    parser.add_argument(
        "--feature-ablation", choices=["full", "no_mpnn", "no_topology", "no_atomic", "no_auxiliary", "esm_only"],
        default="full", help="Mask knowledge-input blocks from a common graph cache.",
    )
    parser.add_argument(
        "--nested-anti-weights", type=float, nargs="+",
        help="Select the consistency weight independently inside each outer-fold training set.",
    )
    parser.add_argument(
        "--validation-grouping", choices=["mutation", "complex"], default="mutation",
        help="How the inner validation split is drawn from each outer-fold training set.",
    )
    parser.add_argument("--only-fold", type=int)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--input-fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument(
        "--splits", nargs="+",
        choices=["fold_random_10", "fold_position_10", "fold_complex_10",
                 "fold_dssa_5",
                 "fold_external_abbind",
                 "fold_sequence_cluster_5", "fold_sequence_cluster_10",
                 "fold_interface_structure_5", "fold_interface_structure_10"],
        default=["fold_random_10", "fold_position_10", "fold_complex_10"],
    )
    parser.add_argument(
        "--atomic-delta-view", choices=["all", "summary", "energy"], default="all",
        help="Use all atomic features, the 30 contact+energy summaries, or 19 energy terms.",
    )
    parser.add_argument("--readout", choices=["attention", "pma"], default="attention")
    parser.add_argument("--pma-seeds", type=int, default=4)
    parser.add_argument("--pma-orth-weight", type=float, default=0.01)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--save-fold-checkpoints", action="store_true",
        help="Save the selected model and target scaling for each outer fold.",
    )
    parser.add_argument("--smoke-per-complex", type=int, default=2)
    args = parser.parse_args()
    if args.cycle_mode is None:
        args.cycle_mode = "message_difference" if args.thermodynamic_cycle else "none"
    args.thermodynamic_cycle = args.cycle_mode != "none"
    if args.reverse_supervision_weight is None:
        args.reverse_supervision_weight = args.antisymmetry_weight
    if args.cycle_consistency_weight is None:
        args.cycle_consistency_weight = args.antisymmetry_weight
    run(args)


if __name__ == "__main__":
    main()
