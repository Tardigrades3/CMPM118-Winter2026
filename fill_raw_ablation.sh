#!/usr/bin/env bash
# Fill the missing RAW DIL stateless CL-method runs so the raw-vs-TD ablation
# table is balanced (naive + replay + EWC, both archs) instead of naive-dominated.
#
# Already have (raw, DIL, stateless regime):
#   hgrn naive (n=4), hgrn replay (n=7), hgrn ewc (FROZEN λ=12032 — unusable)
#   lstm naive (n=1)
# This script adds, all RAW features, plastic settings, seeds 1-3:
#   lstm naive (seeds 2,3) ; lstm replay ; hgrn ewc λ=30 ; lstm ewc λ=30
#
# EWC uses λ=30 (the plastic value — matches the TD ablation so the comparison
# is method-matched; the raw HPO λ=12032 freezes the net and is excluded).
#
# Outputs → results_dil_recovered/  (where the other raw DIL runs live).

set -uo pipefail
export MPLBACKEND=Agg
DATA="${1:-./NinaProData}"
RES="results_dil_recovered"
EP=5
mkdir -p logs "$RES"
[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

COMMON="--scenario dil --exercise 1 --data_path $DATA --d_model 128 --num_layers 4 \
        --batch_size 24 --epochs_per_task $EP --results_dir $RES"

run() { echo ">> $*"; python3 hgrn/train.py $COMMON "$@"; }

# lstm naive raw (add seeds 2,3 to the existing seed-1 run)
for S in 2 3; do
  run --arch lstm --mode stateless --lr 1e-3 --seed "$S"
done

# lstm replay raw (seeds 1-3)
for S in 1 2 3; do
  run --arch lstm --mode replay_stateless --lr 1e-3 \
      --replay_weight 0.5 --replay_capacity 10000 --noise_std 0.01 --replay_batch_size 16 --seed "$S"
done

# ewc raw λ=30 plastic (both archs, seeds 1-3)
for A in hgrn lstm; do
  for S in 1 2 3; do
    run --arch "$A" --mode ewc_stateless --lr 3e-4 --ewc_lambda 30 --seed "$S"
  done
done

echo "DONE raw-ablation fill — $(date '+%H:%M:%S')"
