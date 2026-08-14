#!/usr/bin/env bash
# Fill the missing RAW CIL stateless CL-method runs so the CIL raw-vs-TD
# ablation matches the DIL one (naive + replay + EWC, both archs).
#
# Already have (raw, CIL, stateless regime):
#   hgrn naive (n=10), lstm naive (n=10)
# Missing (this script adds them, all RAW features, plastic settings):
#   {hgrn,lstm} × {replay_stateless, ewc_stateless} × subjects 1..10
#
# EWC uses λ=30 (plastic value, matches the TD/DIL ablation).
# Replay uses the same known-good params as the TD runs.
#
# Outputs → results_cil_td/  ... NO — raw goes to results/  (where raw CIL lives).
#
# Usage:
#   nohup ./fill_raw_ablation_cil.sh ./NinaProData "1 2 3 4 5 6 7 8 9 10" > logs/fill_raw_cil.out 2>&1 &

set -uo pipefail
export MPLBACKEND=Agg
DATA="${1:-./NinaProData}"
SUBJECTS="${2:-1 2 3 4 5 6 7 8 9 10}"
RES="results"          # raw CIL runs live alongside the other raw results
EP=5
mkdir -p logs "$RES"
[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

COMMON="--scenario cil --exercise 1 --data_path $DATA --d_model 128 --num_layers 4 \
        --batch_size 24 --epochs_per_task $EP --results_dir $RES"
run() { echo ">> $*"; python3 hgrn/train.py $COMMON "$@"; }

for SUBJ in $SUBJECTS; do
  for A in hgrn lstm; do
    # replay (raw)
    run --arch "$A" --subject "$SUBJ" --mode replay_stateless --lr 1e-3 \
        --replay_weight 0.5 --replay_capacity 10000 --noise_std 0.01 --replay_batch_size 16
    # EWC (raw, plastic λ=30)
    run --arch "$A" --subject "$SUBJ" --mode ewc_stateless --lr 3e-4 --ewc_lambda 30
  done
done

echo "DONE raw-CIL-ablation fill — $(date '+%H:%M:%S')"
