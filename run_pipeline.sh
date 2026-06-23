#!/usr/bin/env bash
# Full pipeline: HPO → CIL sweep
#
# Phase 1  Architecture HPO   (stateless DIL proxy, 6 representative subjects)
# Phase 2  CL-method HPO      (CIL proxy, 3 subjects, one study per method)
# Phase 3  Full CIL sweep     (all subjects × all modes × all archs)
#
# Usage
# ─────
#   ./run_pipeline.sh --data ./NinaProData           # full run, all phases
#   ./run_pipeline.sh --data ./NinaProData --skip-hpo          # phase 3 only (reads existing JSON)
#   ./run_pipeline.sh --data ./NinaProData --skip-cl-hpo       # skip phase 2, run phase 1 + 3
#   ./run_pipeline.sh --data ./NinaProData --hpo-only          # phases 1+2, no sweep
#
# After HPO runs once the JSON files persist in results/.
# Re-running with --skip-hpo goes straight to the sweep using those saved params.

set -uo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
DATA="./NinaProData"
ARCH="hgrn"                        # arch to HPO; arch params also applied to lstm in sweep
HPO_SUBJ_ARCH="27 23 4 3 24 17"   # 6 stratified subjects for arch HPO
HPO_SUBJ_CL="27 4 17"             # 3 subjects for CL-method HPO
CIL_SUBJECTS="$(seq -s ' ' 1 10)" # subjects for the full CIL sweep
ARCHS_SWEEP="hgrn lstm"            # architectures to include in the sweep
MODES_SWEEP="stateless stateful ewc_stateful replay_stateful herding_stateful"
EPOCHS=5
N_TRIALS_ARCH=50
N_TRIALS_CL=30
OUT_DIR="results"
SKIP_HPO=false
SKIP_ARCH_HPO=false
SKIP_CL_HPO=false
HPO_ONLY=false
DRY_RUN=false

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)              DATA="$2";            shift 2 ;;
    --arch)              ARCH="$2";            shift 2 ;;
    --hpo-subjects-arch) HPO_SUBJ_ARCH="$2";  shift 2 ;;
    --hpo-subjects-cl)   HPO_SUBJ_CL="$2";    shift 2 ;;
    --cil-subjects)      CIL_SUBJECTS="$2";    shift 2 ;;
    --archs-sweep)       ARCHS_SWEEP="$2";     shift 2 ;;
    --modes-sweep)       MODES_SWEEP="$2";     shift 2 ;;
    --epochs)            EPOCHS="$2";          shift 2 ;;
    --n-trials-arch)     N_TRIALS_ARCH="$2";   shift 2 ;;
    --n-trials-cl)       N_TRIALS_CL="$2";     shift 2 ;;
    --out-dir)           OUT_DIR="$2";         shift 2 ;;
    --skip-hpo)          SKIP_HPO=true;        shift   ;;
    --skip-arch-hpo)     SKIP_ARCH_HPO=true;   shift   ;;
    --skip-cl-hpo)       SKIP_CL_HPO=true;     shift   ;;
    --hpo-only)          HPO_ONLY=true;        shift   ;;
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

mkdir -p "$OUT_DIR" logs/cil

ARCH_JSON="$OUT_DIR/hpo_arch_${ARCH}_ex1_best.json"
EWC_JSON="$OUT_DIR/hpo_cl_ewc_stateful_${ARCH}_best.json"
REPLAY_JSON="$OUT_DIR/hpo_cl_replay_stateful_${ARCH}_best.json"
HERDING_JSON="$OUT_DIR/hpo_cl_herding_stateful_${ARCH}_best.json"

# ── Phase 1: Architecture HPO ─────────────────────────────────────────────────
if ! $SKIP_HPO && ! $SKIP_ARCH_HPO; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "Phase 1 — Architecture HPO"
    echo "  arch     : $ARCH"
    echo "  subjects : $HPO_SUBJ_ARCH"
    echo "  trials   : $N_TRIALS_ARCH"
    echo "══════════════════════════════════════════════════"
    _run python hpo.py arch \
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

# ── Phase 2: CL-Method HPO ───────────────────────────────────────────────────
if ! $SKIP_HPO && ! $SKIP_CL_HPO; then
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "Phase 2 — CL Method HPO"
    echo "  subjects : $HPO_SUBJ_CL"
    echo "  trials   : $N_TRIALS_CL each"
    echo "══════════════════════════════════════════════════"

    for CL_MODE in ewc_stateful replay_stateful herding_stateful; do
        echo ""
        echo "── $CL_MODE ──"
        _run python hpo.py cl \
            --mode "$CL_MODE" \
            --subjects $HPO_SUBJ_CL \
            --data_path "$DATA" \
            --arch "$ARCH" \
            --arch_params "$ARCH_JSON" \
            --epochs_per_task "$EPOCHS" \
            --n_trials "$N_TRIALS_CL" \
            --out_dir "$OUT_DIR"
    done
fi

# Read CL-method params (fall back to defaults)
EWC_LAMBDA=$(_jget "$EWC_JSON" ewc_lambda 2000)

REPLAY_WEIGHT=$(_jget "$REPLAY_JSON" replay_weight 0.5)
REPLAY_CAPACITY=$(_jget "$REPLAY_JSON" capacity 10000)
NOISE_STD=$(_jget "$REPLAY_JSON" noise_std 0.01)
REPLAY_BATCH=$(_jget "$REPLAY_JSON" replay_batch_size 16)

HERDING_CPP=$(_jget "$HERDING_JSON" capacity_per_class 20)
HERDING_BATCH=$(_jget "$HERDING_JSON" replay_batch_size 16)

echo ""
echo "EWC    → ewc_lambda=$EWC_LAMBDA"
echo "Replay → weight=$REPLAY_WEIGHT  capacity=$REPLAY_CAPACITY  noise=$NOISE_STD  batch=$REPLAY_BATCH"
echo "Herding→ cpp=$HERDING_CPP  batch=$HERDING_BATCH"

$HPO_ONLY && { echo ""; echo "HPO complete (--hpo-only). Exiting."; exit 0; }

# ── Phase 3: Full CIL sweep ───────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "Phase 3 — Full CIL Sweep"
echo "  archs    : $ARCHS_SWEEP"
echo "  modes    : $MODES_SWEEP"
echo "  subjects : $CIL_SUBJECTS"
echo "══════════════════════════════════════════════════"

failed=()
total=0
completed=0

for SWEEP_ARCH in $ARCHS_SWEEP; do
  for MODE in $MODES_SWEEP; do
    for SUBJ in $CIL_SUBJECTS; do
      total=$((total + 1))
      label="${SWEEP_ARCH}_${MODE}_s${SUBJ}"
      log="logs/cil/${label}.log"
      echo "[$(date '+%H:%M:%S')] ($total) $SWEEP_ARCH | $MODE | subject $SUBJ"

      # Select the right CL params for this mode
      case "$MODE" in
        ewc_stateful)
          EXTRA_ARGS="--ewc_lambda $EWC_LAMBDA"
          ;;
        replay_stateful|replay_stateless)
          EXTRA_ARGS="--replay_weight $REPLAY_WEIGHT --replay_capacity $REPLAY_CAPACITY --noise_std $NOISE_STD --replay_batch_size $REPLAY_BATCH"
          ;;
        herding_stateful)
          EXTRA_ARGS="--herding_capacity_per_class $HERDING_CPP --replay_batch_size $HERDING_BATCH"
          ;;
        *)
          EXTRA_ARGS=""
          ;;
      esac

      CMD=(python hgrn/train.py
        --arch            "$SWEEP_ARCH"
        --scenario        cil
        --subject         "$SUBJ"
        --mode            "$MODE"
        --data_path       "$DATA"
        --d_model         "$D_MODEL"
        --num_layers      "$NUM_LAYERS"
        --lr              "$LR"
        --weight_decay    "$WEIGHT_DECAY"
        --batch_size      "$BATCH_SIZE"
        --epochs_per_task "$EPOCHS"
        $EXTRA_ARGS
      )

      if $DRY_RUN; then
        echo "  ${CMD[*]}"
        continue
      fi

      "${CMD[@]}" 2>&1 | tee "$log"
      exit_code=${PIPESTATUS[0]}

      if [[ $exit_code -ne 0 ]]; then
        echo "  FAILED — see $log"
        failed+=("$label")
      else
        completed=$((completed + 1))
      fi
      echo ""
    done
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
echo "Compute metrics with:"
echo "  python analysis/compute_metrics.py"
