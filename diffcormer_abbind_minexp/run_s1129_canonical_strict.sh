#!/usr/bin/env bash
set -euo pipefail

# Evaluate the canonical shared-potential model under parallel sequence- and
# interface-aware distribution shifts. Usage: WAIT_PID SEED [SEED ...]
wait_pid="${1:?usage: run_s1129_canonical_strict.sh WAIT_PID SEED [SEED ...]}"
shift
seeds=("$@")
if [[ "${#seeds[@]}" -eq 0 ]]; then
  echo "at least one seed is required" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
if [[ "$wait_pid" != "0" ]]; then
  while kill -0 "$wait_pid" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for mutation-path controls ${wait_pid}"
    sleep 30
  done
fi

python_bin="${PYTHON_BIN:-python}"
data_csv=outputs/data_skempi1131_dssa5/skempi_s1129_all_strict.csv
cache_path=outputs/cache/residue_graph_esm650m_abmpnn_topology_atomicdelta_atomicph_chemistry_s1129_dssa5_four_state_canonical.pt
model_path="${ESM_MODEL_PATH:?Set ESM_MODEL_PATH to an ESM-2 checkpoint}"

for seed in "${seeds[@]}"; do
  name="skempi_s1129_canonical_state_potential_strict_seed${seed}"
  if [[ -s "outputs/${name}/summary.json" && -s "outputs/${name}/oof_predictions.csv" ]]; then
    echo "skip completed: ${name}"
    continue
  fi
  echo "$(date '+%F %T') start ${name} on physical GPU0"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" train_residue_graph.py \
    --data-csv "$data_csv" --model-path "$model_path" \
    --cache-path "$cache_path" --output-dir "outputs/${name}" \
    --device cuda --batch-size 8 --epochs 60 --patience 10 \
    --hidden-dim 96 --layers 3 --dropout 0.15 --lr 5e-4 --weight-decay 1e-4 \
    --huber-beta 0.5 --seed "$seed" --input-fusion gated \
    --splits fold_sequence_cluster_5 fold_interface_structure_5 \
    --cycle-mode state_potential --reverse-supervision-weight 0 \
    --cycle-consistency-weight 0 --validation-grouping mutation \
    > "outputs/${name}.log" 2>&1
  echo "$(date '+%F %T') done ${name}"
done

"$python_bin" analyze_state_potential_bootstrap.py \
  > outputs/kbs_supplement/analyze_state_potential_canonical.log 2>&1
"$python_bin" summarize_state_potential_models.py \
  > outputs/kbs_supplement/summarize_state_potential_models.log 2>&1
"$python_bin" analyze_canonical_strict.py \
  > outputs/kbs_supplement/analyze_canonical_strict.log 2>&1
echo "$(date '+%F %T') canonical state-potential analyses complete"
