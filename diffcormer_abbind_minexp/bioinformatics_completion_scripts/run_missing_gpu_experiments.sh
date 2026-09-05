#!/usr/bin/env bash
set -euo pipefail

# Complete five-seed path controls and retrained state-potential feature
# ablations. Each invocation runs serially on one GPU; launch two invocations
# with disjoint seed lists to use two GPUs without within-device contention.
gpu="${1:-1}"
shift || true
seeds=("$@")
if [[ "${#seeds[@]}" -eq 0 ]]; then
  seeds=(42 43 44 45 46)
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_csv="$root/outputs/data_skempi1131_dssa5/skempi_s1129_dssa5.csv"
cache_path="$root/outputs/cache/residue_graph_esm650m_abmpnn_topology_atomicdelta_atomicph_chemistry_s1129_dssa5_four_state_canonical.pt"
model_path="${ESM_MODEL_PATH:?Set ESM_MODEL_PATH to an ESM-2 checkpoint}"
output_root="${OUTPUT_ROOT:-$root/outputs/bioinformatics_completion_work}"
mkdir -p "$output_root/path_models" "$output_root/state_feature_ablations"

train_path_controls() {
  local seed="$1"
  local variant name output extra
  [[ "$seed" == 43 ]] && return 0
  for variant in pairwise soft hard; do
    name="skempi_s1129_path_${variant}_seed${seed}"
    output="$output_root/path_models/$name"
    [[ -s "$output/summary.json" && -s "$output/checkpoint_fold_dssa_5_fold4.pt" ]] && continue
    extra=(--reverse-supervision-weight 0 --cycle-consistency-weight 0)
    [[ "$variant" == soft ]] && extra=(--reverse-supervision-weight 0 --cycle-consistency-weight 0.20)
    [[ "$variant" == hard ]] && extra+=(--hard-antisymmetry)
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$root/train_residue_graph.py" \
      --data-csv "$data_csv" --model-path "$model_path" --cache-path "$cache_path" \
      --output-dir "$output" --device cuda --batch-size 8 --epochs 60 --patience 10 \
      --hidden-dim 96 --layers 3 --dropout 0.15 --lr 5e-4 --weight-decay 1e-4 \
      --huber-beta 0.5 --seed "$seed" --input-fusion gated --splits fold_dssa_5 \
      --feature-ablation no_auxiliary --validation-grouping mutation \
      --save-fold-checkpoints "${extra[@]}" > "$output_root/path_models/${name}.log" 2>&1
  done
}

train_state_ablations() {
  local seed="$1"
  local variant name output
  for variant in no_mpnn no_topology no_atomic esm_only; do
    name="skempi_s1129_state_potential_${variant}_seed${seed}"
    output="$output_root/state_feature_ablations/$name"
    [[ -s "$output/summary.json" ]] && continue
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$root/train_residue_graph.py" \
      --data-csv "$data_csv" --model-path "$model_path" --cache-path "$cache_path" \
      --output-dir "$output" --device cuda --batch-size 8 --epochs 60 --patience 10 \
      --hidden-dim 96 --layers 3 --dropout 0.15 --lr 5e-4 --weight-decay 1e-4 \
      --huber-beta 0.5 --seed "$seed" --input-fusion gated --splits fold_dssa_5 \
      --cycle-mode state_potential --reverse-supervision-weight 0 \
      --cycle-consistency-weight 0 --validation-grouping mutation \
      --feature-ablation "$variant" > "$output_root/state_feature_ablations/${name}.log" 2>&1
  done
}

for seed in "${seeds[@]}"; do
  train_path_controls "$seed"
done
for seed in "${seeds[@]}"; do
  train_state_ablations "$seed"
done
