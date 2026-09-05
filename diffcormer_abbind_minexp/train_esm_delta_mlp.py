#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

import prepare_abbind_minexp as data_prep


def patch_broken_dynamo():
    try:
        import torch._dynamo as dynamo
        getattr(dynamo, "disable")
        getattr(dynamo, "graph_break")
        return
    except Exception:
        pass

    import sys
    import types

    fake = types.ModuleType("torch._dynamo")

    def disable(fn=None, recursive=True):
        if fn is None:
            return lambda f: f
        return fn

    fake.disable = disable
    fake.graph_break = lambda: None
    sys.modules["torch._dynamo"] = fake
    torch._dynamo = fake


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
HYDROPATHY = dict(zip(AA_ORDER, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
VOLUME = dict(zip(AA_ORDER, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 193.6, 227.8]))
CHARGE = {aa: 0.0 for aa in AA_ORDER}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})


class DeltaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2, improved=False):
        super().__init__()
        if improved:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pairwise_rank_loss(pred, target):
    diff_y = target[:, None] - target[None, :]
    sign = torch.sign(diff_y)
    mask = sign != 0
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    diff_pred = pred[:, None] - pred[None, :]
    return F.softplus(-sign[mask] * diff_pred[mask]).mean()


def regression_loss(pred, target, args):
    if args.regression_loss == "mse":
        return F.mse_loss(pred, target)
    if args.regression_loss == "huber":
        return F.smooth_l1_loss(pred, target, beta=args.huber_beta)
    raise ValueError(f"Unknown regression_loss: {args.regression_loss}")


def metrics(pred, target):
    if len(pred) < 2 or np.std(pred) == 0 or np.std(target) == 0:
        return {"spearman": float("nan"), "pearson": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    return {
        "spearman": float(spearmanr(pred, target).correlation),
        "pearson": float(pearsonr(pred, target).statistic),
        "mae": float(np.mean(np.abs(pred - target))),
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
    }


def load_esm(model_path, device):
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    except (IndexError, KeyError, TypeError) as exc:
        # Some locally mirrored checkpoints contain an incompatible/corrupt
        # contact-regression sidecar. Sequence embeddings do not use that head,
        # so fall back to the main model weights only.
        print(f"Contact-regression load failed ({exc}); loading main ESM checkpoint only")
        model_data = torch.load(str(model_path), map_location="cpu")
        model_name = Path(model_path).stem
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data=None
        )
    model.eval().to(device)
    return model, alphabet


@torch.no_grad()
def embed_sequences(model, alphabet, sequences, batch_size, device):
    converter = alphabet.get_batch_converter()
    reps = []
    for start in range(0, len(sequences), batch_size):
        batch = [(str(i), seq) for i, seq in enumerate(sequences[start : start + batch_size])]
        _, _, toks = converter(batch)
        toks = toks.to(device)
        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        rep = out["representations"][model.num_layers].cpu()
        for i, seq in enumerate(sequences[start : start + batch_size]):
            reps.append(rep[i, 1 : len(seq) + 1].contiguous())
    return reps


def row_coordinates(row, coord_cache):
    """Reconstruct the exact coordinate order used by the prepared sequence."""
    path = str(row["pdb_path"])
    if path not in coord_cache:
        coord_cache[path] = data_prep.load_complex_records(Path(path))
    full_sequence, full_residues, index, _ = coord_cache[path]
    if "residue_keys" in row.index and pd.notna(row["residue_keys"]):
        residue_by_key = {
            (str(item["chain"]), data_prep.normalize_pdb_pos(item["pdb_pos"])): item
            for item in full_residues
        }
        keys = [(str(chain), data_prep.normalize_pdb_pos(position))
                for chain, position in json.loads(row["residue_keys"])]
        try:
            residues = [residue_by_key[key] for key in keys]
        except KeyError as exc:
            raise RuntimeError(f"Missing prepared residue key for {row['mut_position_key']}: {exc}") from exc
        coords = np.stack([r["coord"] for r in residues], axis=0).astype(np.float32)
        chains = np.asarray([str(r["chain"]) for r in residues], dtype=object)
        if len(coords) != int(row["seq_len"]):
            raise RuntimeError(
                f"Coordinate/sequence length mismatch for {row['mut_position_key']}: "
                f"{len(coords)} != {row['seq_len']}"
            )
        return coords, chains
    residue = index.get((str(row["chain_id"]), data_prep.normalize_pdb_pos(row["pdb_pos"])))
    if residue is None:
        raise RuntimeError(f"Missing coordinate for {row['mut_position_key']}")
    if len(full_sequence) > 600:
        _, residues, _, _ = data_prep.crop_complex(
            full_sequence,
            full_residues,
            residue,
            max_len=600,
        )
    else:
        residues = full_residues
    coords = np.stack([r["coord"] for r in residues], axis=0).astype(np.float32)
    chains = np.asarray([str(r["chain"]) for r in residues], dtype=object)
    if len(coords) != int(row["seq_len"]):
        raise RuntimeError(
            f"Coordinate/sequence length mismatch for {row['mut_position_key']}: "
            f"{len(coords)} != {row['seq_len']}"
        )
    return coords, chains


def graph_rich_feature(
    row,
    wt,
    mut,
    coord_cache,
):
    """Build mutation-centered radial message-passing descriptors."""
    delta = mut - wt
    site = int(row["mut_index"])
    coords, chains = row_coordinates(row, coord_cache)
    distances = np.linalg.norm(coords - coords[site][None, :], axis=1)
    same_chain = chains == str(row["chain_id"])
    same_chain[site] = False
    other_chain = ~same_chain
    other_chain[site] = False

    parts = [delta[site], wt[site], mut[site]]
    scalar_counts = []
    shells = [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0)]
    for low, high in shells:
        shell = (distances >= low) & (distances < high)
        masks = [shell & same_chain, shell & other_chain]
        for mask in masks:
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                parts.extend([torch.zeros_like(parts[0]) for _ in range(4)])
            else:
                idx_t = torch.from_numpy(idx).long()
                parts.extend(
                    [
                        delta[idx_t].mean(dim=0),
                        wt[idx_t].mean(dim=0),
                        mut[idx_t].mean(dim=0),
                        delta[idx_t].abs().mean(dim=0),
                    ]
                )
            scalar_counts.append(min(float(mask.sum()), 32.0) / 32.0)
    return torch.cat(parts + [torch.tensor(scalar_counts, dtype=parts[0].dtype)], dim=0)


def build_features(df, model_path, cache_path, batch_size, device, feature_set):
    cache_path = Path(cache_path)
    expected_row_ids = [int(x) for x in df["source_row"].tolist()]
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        cached_feature_set = cached.get("feature_set", "delta")
        if cached.get("row_ids") == expected_row_ids and cached_feature_set == feature_set:
            return cached["features"], cached["targets"], cached["row_ids"], cached["embedding_dim"]
        print(f"Ignoring stale cache with {len(cached.get('row_ids', []))} rows: {cache_path}")

    model, alphabet = load_esm(model_path, device)
    # Cropping is mutation-centered, so one complex can legitimately have
    # several different WT sequence windows. Key by the actual sequence rather
    # than pdb_id to keep residue indices aligned with each row.
    wt_sequences = list(dict.fromkeys(df["wt_sequence"].tolist()))
    wt_reps = embed_sequences(model, alphabet, wt_sequences, batch_size, device)
    wt_map = dict(zip(wt_sequences, wt_reps))

    mut_reps = embed_sequences(model, alphabet, df["mutated_sequence"].tolist(), batch_size, device)

    features = []
    targets = []
    row_ids = []
    for i, (_, row) in enumerate(df.iterrows()):
        wt = wt_map[row["wt_sequence"]]
        mut = mut_reps[i]
        delta = mut - wt
        mut_index = int(row["mut_index"])
        site_delta = delta[mut_index]
        wt_site = wt[mut_index]
        mut_site = mut[mut_index]

        interface_indices = json.loads(row["interface_indices"])
        if interface_indices:
            interface_delta = delta[interface_indices].mean(dim=0)
            interface_wt = wt[interface_indices].mean(dim=0)
            interface_mut = mut[interface_indices].mean(dim=0)
        else:
            interface_delta = torch.zeros_like(site_delta)
            interface_wt = torch.zeros_like(site_delta)
            interface_mut = torch.zeros_like(site_delta)

        chain_offsets = json.loads(row["chain_offsets"])
        chain_start, chain_end = chain_offsets[str(row["chain_id"])]
        chain_delta = delta[int(chain_start) : int(chain_end)].mean(dim=0)
        global_delta = delta.mean(dim=0)
        win_start = max(0, mut_index - 2)
        win_end = min(delta.shape[0], mut_index + 3)
        window_delta = delta[win_start:win_end].mean(dim=0)

        count = torch.tensor([min(float(row["interface_count"]), 32.0) / 32.0], dtype=site_delta.dtype)
        if feature_set == "delta":
            feat = torch.cat([site_delta, interface_delta, count], dim=0)
        elif feature_set == "delta_context":
            feat = torch.cat(
                [
                    site_delta,
                    interface_delta,
                    window_delta,
                    chain_delta,
                    global_delta,
                    count,
                ],
                dim=0,
            )
        elif feature_set == "rich":
            feat = torch.cat(
                [
                    site_delta,
                    wt_site,
                    mut_site,
                    interface_delta,
                    window_delta,
                    chain_delta,
                    global_delta,
                    count,
                ],
                dim=0,
            )
        elif feature_set == "rich_abs_interface":
            feat = torch.cat(
                [
                    site_delta,
                    wt_site,
                    mut_site,
                    interface_delta,
                    interface_wt,
                    interface_mut,
                    window_delta,
                    chain_delta,
                    global_delta,
                    count,
                ],
                dim=0,
            )
        elif feature_set == "graph_rich":
            if i == 0:
                coord_cache = {}
            feat = graph_rich_feature(row, wt, mut, coord_cache)
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        features.append(feat)
        targets.append(float(row["ddG"]))
        row_ids.append(int(row["source_row"]))

    features = torch.stack(features).float()
    targets = torch.tensor(targets, dtype=torch.float32)
    if feature_set == "graph_rich":
        embedding_dim = int(wt_reps[0].shape[-1])
    else:
        feature_blocks = {"delta": 2, "delta_context": 5, "rich": 7, "rich_abs_interface": 9}[feature_set]
        embedding_dim = int((features.shape[1] - 1) // feature_blocks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "targets": targets,
            "row_ids": row_ids,
            "feature_set": feature_set,
            "embedding_dim": embedding_dim,
        },
        cache_path,
    )
    return features, targets, row_ids, embedding_dim


def add_metadata_features(x, df):
    extra = []
    for _, row in df.iterrows():
        wt = str(row["wt_aa"])
        mut = str(row["mut_aa"])
        wt_one_hot = [1.0 if wt == aa else 0.0 for aa in AA_ORDER]
        mut_one_hot = [1.0 if mut == aa else 0.0 for aa in AA_ORDER]
        seq_den = max(1.0, float(row["seq_len"]) - 1.0)
        scalars = [
            HYDROPATHY.get(mut, 0.0) - HYDROPATHY.get(wt, 0.0),
            VOLUME.get(mut, 0.0) - VOLUME.get(wt, 0.0),
            CHARGE.get(mut, 0.0) - CHARGE.get(wt, 0.0),
            float(row["mut_index"]) / seq_den,
            min(float(row["interface_count"]), 32.0) / 32.0,
            float(row["seq_len"]) / 600.0,
        ]
        extra.append(wt_one_hot + mut_one_hot + scalars)
    return torch.cat([x, torch.tensor(extra, dtype=x.dtype)], dim=1)


def reverse_feature_rows(x, feature_set, embedding_dim, has_metadata):
    """Construct Mut->WT features from the already embedded endpoint sequences.

    Both endpoint sequences were passed through ESM, so swapping their contextual
    representations and reversing delta blocks is exact for these feature sets.
    """
    block_counts = {
        "delta": 2,
        "delta_context": 5,
        "rich": 7,
        "rich_abs_interface": 9,
        "graph_rich": 35,
    }
    n_blocks = block_counts[feature_set]
    scalar_dim = 8 if feature_set == "graph_rich" else 1
    core_dim = embedding_dim * n_blocks + scalar_dim
    core = x[:, :core_dim].clone()
    offset = 0

    def take():
        nonlocal offset
        value = core[:, offset : offset + embedding_dim]
        offset += embedding_dim
        return value

    if feature_set == "delta":
        delta_blocks = [take(), take()]
        core = torch.cat([-delta_blocks[0], -delta_blocks[1], core[:, offset:]], dim=1)
    elif feature_set == "delta_context":
        delta_blocks = [take() for _ in range(5)]
        core = torch.cat([-value for value in delta_blocks] + [core[:, offset:]], dim=1)
    elif feature_set == "rich":
        site_delta, wt_site, mut_site, interface_delta, window_delta, chain_delta, global_delta = [
            take() for _ in range(7)
        ]
        core = torch.cat(
            [
                -site_delta,
                mut_site,
                wt_site,
                -interface_delta,
                -window_delta,
                -chain_delta,
                -global_delta,
                core[:, offset:],
            ],
            dim=1,
        )
    elif feature_set == "rich_abs_interface":
        (
            site_delta,
            wt_site,
            mut_site,
            interface_delta,
            interface_wt,
            interface_mut,
            window_delta,
            chain_delta,
            global_delta,
        ) = [take() for _ in range(9)]
        core = torch.cat(
            [
                -site_delta,
                mut_site,
                wt_site,
                -interface_delta,
                interface_mut,
                interface_wt,
                -window_delta,
                -chain_delta,
                -global_delta,
                core[:, offset:],
            ],
            dim=1,
        )
    else:
        site_delta, site_wt, site_mut = take(), take(), take()
        reversed_blocks = [-site_delta, site_mut, site_wt]
        for _ in range(8):
            shell_delta, shell_wt, shell_mut, shell_abs_delta = (
                take(), take(), take(), take()
            )
            reversed_blocks.extend(
                [-shell_delta, shell_mut, shell_wt, shell_abs_delta]
            )
        core = torch.cat(reversed_blocks + [core[:, offset:]], dim=1)

    if not has_metadata:
        return core

    metadata = x[:, core_dim:].clone()
    wt_one_hot = metadata[:, :20]
    mut_one_hot = metadata[:, 20:40]
    scalars = metadata[:, 40:].clone()
    if scalars.shape[1] >= 3:
        scalars[:, :3] = -scalars[:, :3]
    metadata = torch.cat([mut_one_hot, wt_one_hot, scalars], dim=1)
    return torch.cat([core, metadata], dim=1)


def add_fold_target_encoding(x_train, x_test, df, y, train_idx, test_idx):
    """Add train-only position/complex target summaries.

    Position encoding is valid for random mutation splits where alternate
    substitutions at a known site may be observed during training. It is
    deliberately computed inside each outer fold and is zero for unseen groups.
    """
    y_np = y.detach().cpu().numpy()
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    global_mean = float(np.mean(y_np[train_idx]))
    global_std = max(float(np.std(y_np[train_idx])), 1e-6)

    train_pos = {}
    train_complex = {}
    for idx in train_idx:
        pos_key = str(df.iloc[idx]["mut_position_key"])
        complex_key = str(df.iloc[idx]["pdb_id"])
        train_pos.setdefault(pos_key, []).append(float(y_np[idx]))
        train_complex.setdefault(complex_key, []).append(float(y_np[idx]))

    def encode(frame, table, frame_indices=None):
        values = []
        counts = []
        for row_number, (_, row) in enumerate(frame.iterrows()):
            key = str(row["mut_position_key"]) if table is train_pos else str(row["pdb_id"])
            group = table.get(key)
            # Training encodings must not contain the row's own target.
            if group is not None and frame_indices is not None:
                own_y = float(y_np[frame_indices[row_number]])
                group = list(group)
                group.remove(own_y)
            if group is None:
                values.append(0.0)
                counts.append(0.0)
            elif len(group) == 0:
                values.append(0.0)
                counts.append(0.0)
            else:
                values.append((float(np.mean(group)) - global_mean) / global_std)
                counts.append(min(len(group), 32) / 32.0)
        return np.asarray(values, dtype=np.float32), np.asarray(counts, dtype=np.float32)

    pos_train, pos_count_train = encode(train_df, train_pos, train_idx)
    pos_test, pos_count_test = encode(test_df, train_pos)
    complex_train, complex_count_train = encode(train_df, train_complex, train_idx)
    complex_test, complex_count_test = encode(test_df, train_complex)

    train_extra = torch.from_numpy(
        np.stack([pos_train, pos_count_train, complex_train, complex_count_train], axis=1)
    )
    test_extra = torch.from_numpy(
        np.stack([pos_test, pos_count_test, complex_test, complex_count_test], axis=1)
    )
    return torch.cat([x_train, train_extra], dim=1), torch.cat([x_test, test_extra], dim=1)


def parse_seeds(value):
    return [int(seed.strip()) for seed in str(value).split(",") if seed.strip()]


def train_fold(x, y, train_idx, test_idx, args):
    device = torch.device(args.device)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    if args.target_encoding:
        x_train, x_test = add_fold_target_encoding(
            x_train, x_test, args.current_df, y, train_idx, test_idx
        )

    if args.reverse_augmentation:
        x_reverse = reverse_feature_rows(
            x_train,
            args.feature_set,
            args.embedding_dim,
            args.augment_metadata,
        )
        x_train = torch.cat([x_train, x_reverse], dim=0)
        y_train = torch.cat([y_train, -y_train], dim=0)

    if args.standardize_features:
        x_mean = x_train.mean(dim=0)
        x_std = x_train.std(dim=0).clamp(min=1e-6)
        x_train = (x_train - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std

    mean = y_train.mean()
    std = y_train.std().clamp(min=1e-6)
    y_train_z = (y_train - mean) / std

    ensemble_preds = []
    for seed in parse_seeds(args.ensemble_seeds):
        torch.manual_seed(seed)
        model = DeltaMLP(
            input_dim=x_train.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            improved=args.improved_mlp,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.early_stop:
            n_train = len(x_train)
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(n_train, generator=generator)
            n_val = max(8, n_train // 5)
            val_local = order[:n_val]
            fit_local = order[n_val:]
            fit_x, fit_y = x_train[fit_local], y_train_z[fit_local]
            val_x, val_y = x_train[val_local].to(device), y_train_z[val_local].to(device)
        else:
            fit_x, fit_y = x_train, y_train_z
            val_x = val_y = None

        loader = DataLoader(TensorDataset(fit_x, fit_y), batch_size=args.batch_size, shuffle=True)
        best_state = None
        best_loss = float("inf")
        stale_epochs = 0

        for _ in range(args.epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = args.mse_weight * regression_loss(pred, yb, args) + args.rank_weight * pairwise_rank_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if args.early_stop:
                model.eval()
                with torch.no_grad():
                    val_loss = F.mse_loss(model(val_x), val_y).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= args.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            pred_z = model(x_test.to(device)).cpu()
        ensemble_preds.append((pred_z * std + mean).numpy())

    pred = np.mean(ensemble_preds, axis=0)
    return metrics(pred, y_test.numpy())


def run_experiment(args):
    df = pd.read_csv(args.data_csv)
    if args.smoke:
        df = df.groupby("pdb_id", group_keys=False).head(max(1, args.smoke_per_complex)).reset_index(drop=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    args.device = device

    x, y, _, embedding_dim = build_features(
        df,
        args.model_path,
        args.cache_path,
        args.esm_batch_size,
        torch.device(device),
        args.feature_set,
    )
    if args.augment_metadata:
        x = add_metadata_features(x, df)
    args.embedding_dim = embedding_dim
    results = []
    split_cols = [
        f"fold_random_{args.n_folds}",
        f"fold_position_{args.n_folds}",
        f"fold_complex_{args.n_folds}",
    ]
    for split_col in split_cols:
        for fold in sorted(df[split_col].unique()):
            test_idx = np.flatnonzero(df[split_col].to_numpy() == fold)
            train_idx = np.flatnonzero(df[split_col].to_numpy() != fold)
            if len(test_idx) < 2 or len(train_idx) < 4:
                continue
            args.current_df = df
            fold_metrics = train_fold(x, y, train_idx, test_idx, args)
            fold_metrics.update({"split": split_col, "fold": int(fold), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
            results.append(fold_metrics)
            print(fold_metrics)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(results)
    result_path = out_dir / ("smoke_results.csv" if args.smoke else "results.csv")
    res.to_csv(result_path, index=False)

    summary = {
        "rows": int(len(df)),
        "complexes": int(df["pdb_id"].nunique()),
        "embedding_dim": embedding_dim,
        "device": device,
        "epochs": args.epochs,
        "features": int(x.shape[1]),
        "feature_set": args.feature_set,
        "augment_metadata": bool(args.augment_metadata),
        "target_encoding": bool(args.target_encoding),
        "ensemble_seeds": parse_seeds(args.ensemble_seeds),
        "early_stop": bool(args.early_stop),
        "regression_loss": args.regression_loss,
        "mean_by_split": res.groupby("split")[["spearman", "pearson", "mae", "rmse"]].mean().to_dict() if not res.empty else {},
        "std_by_split": res.groupby("split")[["spearman", "pearson", "mae", "rmse"]].std().to_dict() if not res.empty else {},
    }
    summary_path = out_dir / ("smoke_summary.json" if args.smoke else "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {result_path}")


def main():
    patch_broken_dynamo()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default="outputs/data/abbind_antibody_single_maxlen600.csv")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to an ESM-2 checkpoint (required when building a feature cache)",
    )
    parser.add_argument("--cache-path", default="outputs/cache/esm_delta_features.pt")
    parser.add_argument("--output-dir", default="outputs/esm_delta_mlp")
    parser.add_argument(
        "--feature-set",
        choices=["delta", "delta_context", "rich", "rich_abs_interface", "graph_rich"],
        default="delta",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--esm-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.8)
    parser.add_argument("--mse-weight", type=float, default=0.2)
    parser.add_argument("--regression-loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--augment-metadata", action="store_true")
    parser.add_argument("--standardize-features", action="store_true")
    parser.add_argument("--improved-mlp", action="store_true")
    parser.add_argument("--target-encoding", action="store_true")
    parser.add_argument("--reverse-augmentation", action="store_true")
    parser.add_argument("--ensemble-seeds", default="43")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--n-folds", type=int, choices=[5, 10], default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-complex", type=int, default=4)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
