#!/usr/bin/env bash
# Full pipeline: HPO → sweep   (different_experiments branch)
#
# Phase 1  Architecture HPO   (stateless proxy, representative subjects)
# Phase 2  CL-method HPO      (CIL + DIL proxy, one study per method)
# Phase 3  Full sweep         (all subjects/exercises × all modes × all archs)
#
# This branch's distinguishing feature is ONLINE EWC (Fisher accumulated across
# tasks). To compare it fairly against the aran-HGRN branch (single-task EWC),
# reuse that branch's tuned hyperparameters and only swap the EWC formulation:
#
#   ./run_pipeline.sh --data ./NinaProData --skip-hpo            # CIL sweep, reuse tuned params
#   ./run_pipeline.sh --data ./NinaProData --skip-hpo --scenario dil
#
# Eval JSONs are written to a SEPARATE dir (default results_diffexp/) so they do
# not mix with the other branch's results/. HPO param JSONs are read from
# --out-dir (default results/) where the existing tuned JSONs already live.
#
#   ./run_pipeline.sh --data ./NinaProData                       # full run, all phases
#   ./run_pipeline.sh --data ./NinaProData --skip-cl-hpo         # phase 1 + 3
#   ./run_pipeline.sh --data ./NinaProData --hpo-only            # phases 1+2, no sweep
#   ./run_pipeline.sh --data ./NinaProData --dry-run             # print commands only
#
# Compare afterwards:
#   python analysis/compute_metrics.py results_diffexp   # this branch (online EWC)
#   python analysis/compute_metrics.py results           # other branch (single-task EWC)

set -uo pipefail

# Headless matplotlib: never open a GUI window during the run (figures that are
# generated still save to disk via savefig). Critical on a headless EC2 box.
export MPLBACKEND=Agg

# ── defaults ──────────────────────────────────────────────────────────────────
DATA="./NinaProData"
ARCH="hgrn"                        # arch to HPO; arch params also applied to lstm in sweep
HPO_SUBJ_ARCH="27 23 4 3 24 17"   # 6 stratified subjects for arch HPO
HPO_SUBJ_CL="27 4 17"             # 3 subjects for CL-method HPO
SCENARIO="cil"                     # 'cil' or 'dil' for the Phase-3 sweep
CIL_SUBJECTS="$(seq -s ' ' 1 10)" # subjects for the full CIL sweep
DIL_EXERCISE=1                     # exercise number for DIL sweep
ARCHS_SWEEP="hgrn lstm"            # architectures to include in the sweep
MODES_SWEEP="stateless stateful ewc_stateful replay_stateful herding_stateful"
EPOCHS=5
N_TRIALS_ARCH=50
N_TRIALS_CL=30
OUT_DIR="results"                  # where HPO param JSONs are read/written
RESULTS_DIR="results_diffexp"      # where Phase-3 eval JSONs are written (kept separate)
SKIP_HPO=false
SKIP_ARCH_HPO=false
SKIP_CL_HPO=false
HPO_ONLY=false
USE_DEFAULTS=false                 # ignore HPO JSONs; use train.py argparse defaults
DRY_RUN=false

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)              DATA="$2";            shift 2 ;;
    --arch)              ARCH="$2";            shift 2 ;;
    --hpo-subjects-arch) HPO_SUBJ_ARCH="$2";  shift 2 ;;
    --hpo-subjects-cl)   HPO_SUBJ_CL="$2";    shift 2 ;;
    --scenario)          SCENARIO="$2";        shift 2 ;;
    --cil-subjects)      CIL_SUBJECTS="$2";    shift 2 ;;
    --dil-exercise)      DIL_EXERCISE="$2";    shift 2 ;;
    --archs-sweep)       ARCHS_SWEEP="$2";     shift 2 ;;
    --modes-sweep)       MODES_SWEEP="$2";     shift 2 ;;
    --epochs)            EPOCHS="$2";          shift 2 ;;
    --n-trials-arch)     N_TRIALS_ARCH="$2";   shift 2 ;;
    --n-trials-cl)       N_TRIALS_CL="$2";     shift 2 ;;
    --out-dir)           OUT_DIR="$2";         shift 2 ;;
    --results-dir)       RESULTS_DIR="$2";     shift 2 ;;
    --skip-hpo)          SKIP_HPO=true;        shift   ;;
    --skip-arch-hpo)     SKIP_ARCH_HPO=true;   shift   ;;
    --skip-cl-hpo)       SKIP_CL_HPO=true;     shift   ;;
    --hpo-only)          HPO_ONLY=true;        shift   ;;
    --use-defaults)      USE_DEFAULTS=true; SKIP_HPO=true; shift ;;
    --dry-run)           DRY_RUN=true;         shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
_jget() {
    # _jget <json_file> <key> <default>
    python3 -c "
import json, sys
try:
    v = json.load(open('$1'))['params'].get('$2')
    print(v if v is not None else $3)
except Exception:
    print($3)
"
}

_run() {
    echo "  \$ $*"
    $DRY_RUN || "$@"
}

mkdir -p "$OUT_DIR" "$RESULTS_DIR" "logs/$SCENARIO"

ARCH_JSON="$OUT_DIR/hpo_arch_${ARCH}_ex1_best.json"

# ── Phase 1: Architecture HPO ─────────────────────────────────────────────────
if ! $SKIP_HPO && ! $SKIP_ARCH_HPO; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "Phase 1 — Architecture HPO"
    echo "  arch     : $ARCH"
    echo "  subjects : $HPO_SUBJ_ARCH"
    echo "  trials   : $N_TRIALS_ARCH"
    echo "══════════════════════════════════════════════════"
    _run python3 hpo.py arch \
        --subjects $HPO_SUBJ_ARCH \
        --exercise 1 \
        --data_path "$DATA" \
        --arch "$ARCH" \
        --epochs_per_task "$EPOCHS" \
        --n_trials "$N_TRIALS_ARCH" \
        --out_dir "$OUT_DIR"
fi

# Read arch params (fall back to defaults if JSON absent)
D_MODEL=$(_jget "$ARCH_JSON" d_model 128)
NUM_LAYERS=$(_jget "$ARCH_JSON" num_layers 4)
LR=$(_jget "$ARCH_JSON" lr 0.0001)
WEIGHT_DECAY=$(_jget "$ARCH_JSON" weight_decay 0.01)
BATCH_SIZE=$(_jget "$ARCH_JSON" batch_size 32)

echo ""
echo "Arch params → d_model=$D_MODEL  num_layers=$NUM_LAYERS  lr=$LR  wd=$WEIGHT_DECAY  bs=$BATCH_SIZE"
[[ -f "$ARCH_JSON" ]] || echo "  (using defaults — $ARCH_JSON not found)"

# ── Phase 2: CL-Method HPO (runs for both cil and dil scenarios) ─────────────
if ! $SKIP_HPO && ! $SKIP_CL_HPO; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "Phase 2 — CL Method HPO"
    echo "  subjects : $HPO_SUBJ_CL"
    echo "  trials   : $N_TRIALS_CL each (× 2 scenarios × 3 methods)"
    echo "══════════════════════════════════════════════════"

    for HPO_SCENARIO in cil dil; do
        for CL_MODE in ewc_stateful replay_stateful herding_stateful; do
            echo ""
            echo "── $CL_MODE ($HPO_SCENARIO) ──"
            EX_ARG=""
            [[ "$HPO_SCENARIO" == "dil" ]] && EX_ARG="--exercise $DIL_EXERCISE"
            _run python3 hpo.py cl \
                --mode "$CL_MODE" \
                --scenario "$HPO_SCENARIO" \
                --subjects $HPO_SUBJ_CL \
                $EX_ARG \
                --data_path "$DATA" \
                --arch "$ARCH" \
                --arch_params "$ARCH_JSON" \
                --epochs_per_task "$EPOCHS" \
                --n_trials "$N_TRIALS_CL" \
                --out_dir "$OUT_DIR"
        done
    done
fi

echo ""
echo "CL params will be read from $OUT_DIR/hpo_cl_*_${ARCH}_{cil,dil}_best.json at sweep time."

$HPO_ONLY && { echo ""; echo "HPO complete (--hpo-only). Exiting."; exit 0; }

# ── Phase 3: Sweep ────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "Phase 3 — $(echo "$SCENARIO" | tr '[:lower:]' '[:upper:]') Sweep   (eval JSONs → $RESULTS_DIR)"
echo "  archs    : $ARCHS_SWEEP"
echo "  modes    : $MODES_SWEEP"
[[ "$SCENARIO" == "cil" ]] && echo "  subjects : $CIL_SUBJECTS"
[[ "$SCENARIO" == "dil" ]] && echo "  exercise : $DIL_EXERCISE"
echo "══════════════════════════════════════════════════"

mkdir -p "logs/$SCENARIO"

failed=()
total=0
completed=0

_run_one() {
  local label="$1"; shift
  local log="logs/$SCENARIO/${label}.log"
  total=$((total + 1))
  echo "[$(date '+%H:%M:%S')] ($total) $label"
  if $DRY_RUN; then echo "  $*"; return; fi
  "$@" 2>&1 | tee "$log"
  local ec=${PIPESTATUS[0]}
  if [[ $ec -ne 0 ]]; then
    echo "  FAILED — see $log"; failed+=("$label")
  else
    completed=$((completed + 1))
  fi
  echo ""
}

_extra_args() {
  # Reads CL params from the HPO JSON matching the current SCENARIO.
  case "$1" in
    ewc_stateful)
      local json="$OUT_DIR/hpo_cl_ewc_stateful_${ARCH}_${SCENARIO}_best.json"
      local lambda; lambda=$(_jget "$json" ewc_lambda 2000)
      echo "--ewc_lambda $lambda" ;;
    replay_stateful|replay_stateless)
      local json="$OUT_DIR/hpo_cl_replay_stateful_${ARCH}_${SCENARIO}_best.json"
      local w; w=$(_jget "$json" replay_weight 0.5)
      local c; c=$(_jget "$json" capacity 10000)
      local n; n=$(_jget "$json" noise_std 0.01)
      local b; b=$(_jget "$json" replay_batch_size 16)
      echo "--replay_weight $w --replay_capacity $c --noise_std $n --replay_batch_size $b" ;;
    herding_stateful)
      local json="$OUT_DIR/hpo_cl_herding_stateful_${ARCH}_${SCENARIO}_best.json"
      local cpp; cpp=$(_jget "$json" capacity_per_class 20)
      local b; b=$(_jget "$json" replay_batch_size 16)
      echo "--herding_capacity_per_class $cpp --replay_batch_size $b" ;;
    *)
      echo "" ;;
  esac
}

# Architecture args passed to train.py. Empty under --use-defaults so train.py
# falls back to its own argparse defaults = the ORIGINAL different_experiments
# config (d_model=128 num_layers=4 lr=1e-4 wd=0.01 bs=32 ewc_lambda=2000).
if $USE_DEFAULTS; then
  ARCH_ARGS=()
  echo "  config   : train.py defaults (d_model=128 num_layers=4 lr=1e-4 wd=0.01 bs=32 ewc_lambda=2000)"
else
  ARCH_ARGS=(--d_model "$D_MODEL" --num_layers "$NUM_LAYERS" --lr "$LR" --weight_decay "$WEIGHT_DECAY" --batch_size "$BATCH_SIZE")
  echo "  config   : HPO params (d_model=$D_MODEL num_layers=$NUM_LAYERS lr=$LR wd=$WEIGHT_DECAY bs=$BATCH_SIZE)"
fi

for SWEEP_ARCH in $ARCHS_SWEEP; do
  for MODE in $MODES_SWEEP; do
    if $USE_DEFAULTS; then EXTRA=""; else EXTRA=$(_extra_args "$MODE"); fi

    if [[ "$SCENARIO" == "dil" ]]; then
      _run_one "${SWEEP_ARCH}_${MODE}_ex${DIL_EXERCISE}" \
        python3 hgrn/train.py \
        --arch            "$SWEEP_ARCH" \
        --scenario        dil \
        --exercise        "$DIL_EXERCISE" \
        --mode            "$MODE" \
        --data_path       "$DATA" \
        ${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"} \
        --epochs_per_task "$EPOCHS" \
        --results_dir     "$RESULTS_DIR" \
        $EXTRA
    else
      for SUBJ in $CIL_SUBJECTS; do
        _run_one "${SWEEP_ARCH}_${MODE}_s${SUBJ}" \
          python3 hgrn/train.py \
          --arch            "$SWEEP_ARCH" \
          --scenario        cil \
          --subject         "$SUBJ" \
          --mode            "$MODE" \
          --data_path       "$DATA" \
          ${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"} \
          --epochs_per_task "$EPOCHS" \
          --results_dir     "$RESULTS_DIR" \
          $EXTRA
      done
    fi
  done
done

# ── summary ───────────────────────────────────────────────────────────────────
$DRY_RUN && exit 0

echo "════════════════════════════════════════"
echo "Sweep complete: $completed / $total runs succeeded."
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed (${#failed[@]}):"
  for r in "${failed[@]}"; do echo "  ✗ $r"; done
  exit 1
fi
echo ""
echo "Compare EWC formulations:"
echo "  python analysis/compute_metrics.py $RESULTS_DIR   # this branch (online EWC)"
echo "  python analysis/compute_metrics.py results        # other branch (single-task EWC)"
