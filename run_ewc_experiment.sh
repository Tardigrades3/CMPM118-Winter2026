#!/usr/bin/env bash
# One-shot EWC comparison — run this, then step away.
#
# Goal: figure out WHY this branch's EWC (online accumulated Fisher) retains so
# much better than aran-HGRN's. We run two variants on this branch and compare
# them (and against aran-HGRN's known CIL-EWC HGRN = 0.278):
#
#   Variant A  — aran-HGRN's HPO params (d_model=256, layers=2, lr=4.8e-4, λ=1591/727)
#                → ONLY the EWC formulation differs from aran-HGRN.
#   Variant B  — this branch's original defaults (d_model=128, layers=4, lr=1e-4, λ=2000)
#                → the config you originally observed "working".
#
# Decode the result:
#   A >> 0.278            → the online-EWC FORMULATION is the cause
#   B >> A                → the HYPERPARAMETERS (lower lr / more depth / higher λ) add on top
#
# Usage:
#   ./run_ewc_experiment.sh                       # data at ./NinaProData
#   ./run_ewc_experiment.sh /path/to/NinaProData
#   nohup ./run_ewc_experiment.sh &               # detach + survive SSH logout
#
# Everything is logged to logs/ewc_experiment_<timestamp>.log

set -uo pipefail
export MPLBACKEND=Agg                 # no GUI windows; figures still save to results/plots/

DATA="${1:-./NinaProData}"
ARCHS="hgrn"                          # focus on HGRN (the architecture in question)
MODES="stateful ewc_stateful"        # naive baseline vs EWC, to read off forgetting
SUBJECTS="$(seq -s ' ' 1 10)"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/ewc_experiment_${TS}.log"
mkdir -p logs

# Activate the EC2 venv if present (harmless elsewhere).
[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

{
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  EWC comparison experiment — started $TS"
echo "║  data=$DATA  archs=$ARCHS  modes=$MODES"
echo "╚══════════════════════════════════════════════════════════════╝"

# Pull aran-HGRN's tuned params for Variant A (no-op if already present).
./sync_hpo_params.sh || echo "WARN: sync_hpo_params failed — Variant A may use defaults"

# ── Variant A: HPO params, online EWC (isolates the formulation) ──────────────
echo ""; echo "########## VARIANT A — HPO params (isolates EWC formulation) ##########"
./run_pipeline.sh --data "$DATA" --skip-hpo \
  --archs-sweep "$ARCHS" --modes-sweep "$MODES" --cil-subjects "$SUBJECTS" \
  --results-dir results_A_hpo
./run_pipeline.sh --data "$DATA" --skip-hpo --scenario dil \
  --archs-sweep "$ARCHS" --modes-sweep "$MODES" \
  --results-dir results_A_hpo

# ── Variant B: original different_experiments defaults, online EWC ────────────
echo ""; echo "########## VARIANT B — original defaults (lr=1e-4, layers=4, λ=2000) ##########"
./run_pipeline.sh --data "$DATA" --use-defaults \
  --archs-sweep "$ARCHS" --modes-sweep "$MODES" --cil-subjects "$SUBJECTS" \
  --results-dir results_B_defaults
./run_pipeline.sh --data "$DATA" --use-defaults --scenario dil \
  --archs-sweep "$ARCHS" --modes-sweep "$MODES" \
  --results-dir results_B_defaults

# ── Summaries ─────────────────────────────────────────────────────────────────
echo ""; echo "########## METRICS ##########"
echo "===== Variant A (HPO params, online EWC) ====="
python3 analysis/compute_metrics.py results_A_hpo      --out-dir results_A_hpo
echo ""
echo "===== Variant B (original defaults, online EWC) ====="
python3 analysis/compute_metrics.py results_B_defaults --out-dir results_B_defaults

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DONE — $(date '+%Y%m%d_%H%M%S')"
echo "║  Reference: aran-HGRN CIL-EWC HGRN final acc = 0.278 (single-task EWC)"
echo "║  Compare the 'ewc_stateful' rows above:"
echo "║    A >> 0.278  → online-EWC formulation is the driver"
echo "║    B >> A      → hyperparameters (lr/depth/λ) add further gains"
echo "║  Eval JSONs: results_A_hpo/ , results_B_defaults/"
echo "╚══════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log saved to: $LOG"
