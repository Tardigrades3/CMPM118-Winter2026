#!/usr/bin/env bash
# Run CIL training independently for each subject × mode × architecture.
#
# Usage:
#   ./run_cil_subjects.sh                          # default: subjects 1-10, all modes+archs
#   ./run_cil_subjects.sh --data ./NinaProData_DB1
#   ./run_cil_subjects.sh --subjects "1 2 3 4 5"
#   ./run_cil_subjects.sh --archs hgrn
#   ./run_cil_subjects.sh --modes "stateful ewc_stateful"
#   ./run_cil_subjects.sh --dry-run               # print commands without running
#
# Logs are written to logs/cil/<arch>_<mode>_s<subj>.log
# Summarize results with: python analysis/compute_metrics.py

set -uo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
DATA="./NinaProData"
SUBJECTS="$(seq -s ' ' 1 10)"
ARCHS="hgrn lstm"
MODES="stateless stateful replay_stateful ewc_stateful herding_stateful"
EPOCHS=5
BATCH=32
LR=1e-4
DRY_RUN=false

# ── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)      DATA="$2";     shift 2 ;;
    --subjects)  SUBJECTS="$2"; shift 2 ;;
    --archs)     ARCHS="$2";    shift 2 ;;
    --modes)     MODES="$2";    shift 2 ;;
    --epochs)    EPOCHS="$2";   shift 2 ;;
    --dry-run)   DRY_RUN=true;  shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── setup ────────────────────────────────────────────────────────────────────
LOG_DIR="logs/cil"
mkdir -p "$LOG_DIR"

n_archs=$(echo "$ARCHS"    | wc -w | tr -d ' ')
n_modes=$(echo "$MODES"    | wc -w | tr -d ' ')
n_subj=$( echo "$SUBJECTS" | wc -w | tr -d ' ')
total=$((n_archs * n_modes * n_subj))

echo "CIL sweep: $n_archs arch(s) × $n_modes mode(s) × $n_subj subject(s) = $total runs"
echo "  archs   : $ARCHS"
echo "  modes   : $MODES"
echo "  subjects: $SUBJECTS"
echo "  data    : $DATA"
$DRY_RUN && echo "  (dry run — no training will occur)"
echo ""

# ── main loop ────────────────────────────────────────────────────────────────
failed_runs=()
completed=0
run_idx=0

for ARCH in $ARCHS; do
  for MODE in $MODES; do
    for SUBJ in $SUBJECTS; do
      run_idx=$((run_idx + 1))
      label="${ARCH}_${MODE}_s${SUBJ}"
      log="$LOG_DIR/${label}.log"

      echo "[$(date '+%H:%M:%S')] ($run_idx/$total) $ARCH | $MODE | subject $SUBJ"

      cmd=(python hgrn/train.py
        --arch      "$ARCH"
        --scenario  cil
        --subject   "$SUBJ"
        --mode      "$MODE"
        --data_path "$DATA"
        --batch_size "$BATCH"
        --epochs_per_task "$EPOCHS"
        --lr "$LR"
      )

      if $DRY_RUN; then
        echo "  ${cmd[*]}"
        continue
      fi

      "${cmd[@]}" 2>&1 | tee "$log"
      exit_code=${PIPESTATUS[0]}

      if [ "$exit_code" -ne 0 ]; then
        echo "  FAILED (exit $exit_code) — see $log"
        failed_runs+=("$label")
      else
        completed=$((completed + 1))
        echo "  OK"
      fi
      echo ""
    done
  done
done

# ── summary ──────────────────────────────────────────────────────────────────
$DRY_RUN && exit 0

echo "════════════════════════════════════════"
echo "Completed $completed / $total runs."

if [ ${#failed_runs[@]} -gt 0 ]; then
  echo "Failed (${#failed_runs[@]}):"
  for r in "${failed_runs[@]}"; do
    echo "  ✗ $r  (log: $LOG_DIR/${r}.log)"
  done
  exit 1
else
  echo "All runs succeeded."
  echo ""
  echo "Summarize results with:"
  echo "  python analysis/compute_metrics.py"
fi
