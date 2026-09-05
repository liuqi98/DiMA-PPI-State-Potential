#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
shift || true
seeds=("$@")
if [[ "${#seeds[@]}" -eq 0 ]]; then
  seeds=(42 43 44 45 46)
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
data_csv="${DATA_CSV:-$root/outputs/external_abbind/s1129_train_abbind_test.csv}"
cache_path="${CACHE_PATH:-$root/outputs/abbind_state_external/s1129_train_abbind_test_four_state_canonical.pt}"
model_path="${ESM_MODEL_PATH:?Set ESM_MODEL_PATH to an ESM-2 checkpoint}"
output_root="${OUTPUT_ROOT:-$root/outputs/abbind_state_external/refit_all_s1129}"
mkdir -p "$output_root"

for seed in "${seeds[@]}"; do
  output="$output_root/seed${seed}"
  [[ -s "$output/summary.json" ]] && continue
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$root/train_residue_graph.py" \
    --data-csv "$data_csv" --model-path "$model_path" --cache-path "$cache_path" \
    --output-dir "$output" --splits fold_external_abbind --only-fold 0 --seed "$seed" \
    --device cuda --batch-size 8 --epochs 60 --patience 10 --hidden-dim 96 --layers 3 \
    --dropout 0.15 --lr 5e-4 --weight-decay 1e-4 --huber-beta 0.5 \
    --input-fusion gated --cycle-mode state_potential --reverse-supervision-weight 0 \
    --cycle-consistency-weight 0 --validation-grouping mutation --save-fold-checkpoints \
    > "$output_root/seed${seed}.log" 2>&1
done
