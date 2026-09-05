#!/usr/bin/env bash
set -euo pipefail

# Matched control for the four-state model: retain the exact shared-potential
# architecture but omit the independently generated bound/free descriptors.
# Usage: run_s1129_state_potential_nostate.sh GPU SEED [SEED ...]

gpu="${1:?usage: run_s1129_state_potential_nostate.sh GPU SEED [SEED ...]}"
shift
seeds=("$@")
if [[ "${#seeds[@]}" -eq 0 ]]; then
  echo "at least one seed is required" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
data_csv="outputs/data_skempi1131_dssa5/skempi_s1129_dssa5.csv"
cache_path="outputs/cache/residue_graph_esm650m_abmpnn_topology_atomicdelta_atomicph_chemistry_s1129_dssa5.pt"
model_path="${ESM_MODEL_PATH:?Set ESM_MODEL_PATH to an ESM-2 checkpoint}"
python_bin="${PYTHON_BIN:-python}"

for seed in "${seeds[@]}"; do
  name="skempi_s1129_dssa5_state_potential_nostate_seed${seed}"
  if [[ -s "outputs/${name}/summary.json" && -s "outputs/${name}/oof_predictions.csv" ]]; then
    echo "skip completed: ${name}"
    continue
  fi

  echo "$(date '+%F %T') start ${name} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" train_residue_graph.py \
    --data-csv "$data_csv" \
    --model-path "$model_path" \
    --cache-path "$cache_path" \
    --output-dir "outputs/${name}" \
    --device cuda --batch-size 8 --epochs 60 --patience 10 \
    --hidden-dim 96 --layers 3 --dropout 0.15 --lr 5e-4 --weight-decay 1e-4 \
    --huber-beta 0.5 --seed "$seed" --input-fusion gated --splits fold_dssa_5 \
    --cycle-mode state_potential --reverse-supervision-weight 0 \
    --cycle-consistency-weight 0 --validation-grouping mutation \
    > "outputs/${name}.log" 2>&1
  echo "$(date '+%F %T') done ${name}"
done
