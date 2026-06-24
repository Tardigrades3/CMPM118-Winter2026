#!/usr/bin/env bash
# run_fresh.sh
#
# Full fresh pipeline:
#   Stage 1  — Arch HPO            (shared proxy, stateless DIL, 6 subjects)
#   Stage 2  — CL-method HPO       (DIL scenario, 3 methods)
#   Stage 3  — Full DIL sweep      (all 27 subjects × all modes × both archs)
#   Stage 4  — git commit + push   (arch JSON + DIL CL JSONs + DIL eval results)
#   Stage 5  — CL-method HPO       (CIL scenario, 3 methods)
#   Stage 6  — Full CIL sweep      (10 subjects × all modes × both archs)
#   Stage 7  — git commit + push   (CIL CL JSONs + CIL eval results)
#
# Usage
# ─────
#   ./run_fresh.sh --data ./NinaProData
#   ./run_fresh.sh --data ./NinaProData --dry-run
#   ./run_fresh.sh --data ./NinaProData --skip-arch-hpo   # arch JSON already exists
#
# Flags that skip individual stages:
#   --skip-arch-hpo   skip Stage 1 (reuse existing hpo_arch_*.json)
#   --skip-dil-hpo    skip Stage 2
#   --skip-dil-sweep  skip Stage 3
#   --skip-git-mid    skip Stage 4 (intermediate git commit)
#   --skip-cil-hpo    skip Stage 5
#   --skip-cil-sweep  skip Stage 6
#   --skip-git-final  skip Stage 7
#   --dry-run         print commands without running them or committing

set -uo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
DATA="./NinaProData"
ARCH="hgrn"
HPO_SUBJ_ARCH="27 23 4 3 24 17"
HPO_SUBJ_CL="27 4 17"
DIL_EXERCISE=1
CIL_SUBJECTS="$(seq -s ' ' 1 10)"
ARCHS_SWEEP="hgrn lstm"
MODES_SWEEP="stateless stateful ewc_stateful replay_stateful herding_stateful"
EPOCHS=5
N_TRIALS_ARCH=50
N_TRIALS_CL=30
OUT_DIR="results"

SKIP_ARCH_HPO=false
SKIP_DIL_HPO=false
SKIP_DIL_SWEEP=false
SKIP_GIT_MID=false
SKIP_CIL_HPO=false
SKIP_CIL_SWEEP=false
SKIP_GIT_FINAL=false
DRY_RUN=false

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)            DATA="$2";           shift 2 ;;
    --arch)            ARCH="$2";           shift 2 ;;
    --epochs)          EPOCHS="$2";         shift 2 ;;
    --n-trials-arch)   N_TRIALS_ARCH="$2";  shift 2 ;;
    --n-trials-cl)     N_TRIALS_CL="$2";    shift 2 ;;
    --out-dir)         OUT_DIR="$2";        shift 2 ;;
    --skip-arch-hpo)   SKIP_ARCH_HPO=true;  shift   ;;
    --skip-dil-hpo)    SKIP_DIL_HPO=true;   shift   ;;
    --skip-dil-sweep)  SKIP_DIL_SWEEP=true; shift   ;;
    --skip-git-mid)    SKIP_GIT_MID=true;   shift   ;;
    --skip-cil-hpo)    SKIP_CIL_HPO=true;   shift   ;;
    --skip-cil-sweep)  SKIP_CIL_SWEEP=true; shift   ;;
    --skip-git-final)  SKIP_GIT_FINAL=true; shift   ;;
    --dry-run)         DRY_RUN=true;        shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR" logs/cil logs/dil

ARCH_JSON="$OUT_DIR/hpo_arch_${ARCH}_ex1_best.json"

# ── helpers ───────────────────────────────────────────────────────────────────
_run() {
  echo "  \$ $*"
  $DRY_RUN || "$@"
}

_banner() {
  echo ""
  echo "══════════════════════════════════════════════════"
  echo "$1"
  echo "══════════════════════════════════════════════════"
}

_git_add_if_exists() {
  local f="$1"
  if [[ -f "$f" ]]; then
    echo "  git add $f"
    $DRY_RUN || git add "$f"
  else
    echo "  (skip — not found: $f)"
  fi
}

_git_commit_push() {
  local msg="$1"
  if $DRY_RUN; then
    echo "  [dry-run] git commit -m \"$msg\" && git push origin HEAD"
    return
  fi
  if git diff --cached --quiet; then
    echo "  Nothing staged — nothing to commit."
  else
    git commit -m "$msg"
    git push origin HEAD
  fi
}

# ── Stage 1: Arch HPO ─────────────────────────────────────────────────────────
_banner "Stage 1 — Arch HPO (shared)"
echo "  arch     : $ARCH"
echo "  subjects : $HPO_SUBJ_ARCH"
echo "  trials   : $N_TRIALS_ARCH"
if $SKIP_ARCH_HPO; then
  echo "  [skipped — using existing $ARCH_JSON]"
else
  _run python hpo.py arch \
      --subjects $HPO_SUBJ_ARCH \
      --exercise "$DIL_EXERCISE" \
      --data_path "$DATA" \
      --arch "$ARCH" \
      --epochs_per_task "$EPOCHS" \
      --n_trials "$N_TRIALS_ARCH" \
      --out_dir "$OUT_DIR"
fi

# ── Stage 2: CL-method HPO — DIL ─────────────────────────────────────────────
_banner "Stage 2 — CL Method HPO (DIL scenario)"
echo "  subjects : $HPO_SUBJ_CL"
echo "  exercise : $DIL_EXERCISE"
echo "  trials   : $N_TRIALS_CL each"
if $SKIP_DIL_HPO; then
  echo "  [skipped]"
else
  for CL_MODE in ewc_stateful replay_stateful herding_stateful; do
    echo ""
    echo "── $CL_MODE (dil) ──"
    _run python hpo.py cl \
        --mode "$CL_MODE" \
        --scenario dil \
        --exercise "$DIL_EXERCISE" \
        --subjects $HPO_SUBJ_CL \
        --data_path "$DATA" \
        --arch "$ARCH" \
        --arch_params "$ARCH_JSON" \
        --epochs_per_task "$EPOCHS" \
        --n_trials "$N_TRIALS_CL" \
        --out_dir "$OUT_DIR"
  done
fi

# ── Stage 3: Full DIL sweep ───────────────────────────────────────────────────
_banner "Stage 3 — Full DIL sweep"
echo "  archs    : $ARCHS_SWEEP"
echo "  modes    : $MODES_SWEEP"
echo "  exercise : $DIL_EXERCISE  (all 27 subjects)"
if $SKIP_DIL_SWEEP; then
  echo "  [skipped]"
else
  touch /tmp/run_fresh_dil_sweep_start

  _run ./run_pipeline.sh \
      --data "$DATA" \
      --arch "$ARCH" \
      --scenario dil \
      --dil-exercise "$DIL_EXERCISE" \
      --archs-sweep "$ARCHS_SWEEP" \
      --modes-sweep "$MODES_SWEEP" \
      --epochs "$EPOCHS" \
      --out-dir "$OUT_DIR" \
      --skip-hpo
fi

# ── Stage 4: Git commit — arch + DIL HPO + DIL eval results ──────────────────
_banner "Stage 4 — Git commit: arch HPO + DIL results"
if $SKIP_GIT_MID; then
  echo "  [skipped]"
else
  _git_add_if_exists "$OUT_DIR/hpo_arch_${ARCH}_ex1_best.json"
  _git_add_if_exists "$OUT_DIR/hpo_cl_ewc_stateful_${ARCH}_dil_best.json"
  _git_add_if_exists "$OUT_DIR/hpo_cl_replay_stateful_${ARCH}_dil_best.json"
  _git_add_if_exists "$OUT_DIR/hpo_cl_herding_stateful_${ARCH}_dil_best.json"

  if [[ -f /tmp/run_fresh_dil_sweep_start ]]; then
    echo "  Staging new DIL eval results..."
    NEW_EVALS=$(find "$OUT_DIR" -name 'eval_dil_*.json' \
                     -newer /tmp/run_fresh_dil_sweep_start 2>/dev/null)
    if [[ -n "$NEW_EVALS" ]]; then
      echo "$NEW_EVALS" | while read -r f; do
        echo "  git add $f"
        $DRY_RUN || git add "$f"
      done
    else
      echo "  (no new eval_dil_*.json files found)"
    fi
  fi

  _git_commit_push "feat: arch HPO + DIL CL HPO + DIL sweep results (${ARCH})"
fi

# ── Stage 5: CL-method HPO — CIL ─────────────────────────────────────────────
_banner "Stage 5 — CL Method HPO (CIL scenario)"
echo "  subjects : $HPO_SUBJ_CL"
echo "  trials   : $N_TRIALS_CL each"
if $SKIP_CIL_HPO; then
  echo "  [skipped]"
else
  for CL_MODE in ewc_stateful replay_stateful herding_stateful; do
    echo ""
    echo "── $CL_MODE (cil) ──"
    _run python hpo.py cl \
        --mode "$CL_MODE" \
        --scenario cil \
        --subjects $HPO_SUBJ_CL \
        --data_path "$DATA" \
        --arch "$ARCH" \
        --arch_params "$ARCH_JSON" \
        --epochs_per_task "$EPOCHS" \
        --n_trials "$N_TRIALS_CL" \
        --out_dir "$OUT_DIR"
  done
fi

# ── Stage 6: Full CIL sweep ───────────────────────────────────────────────────
_banner "Stage 6 — Full CIL sweep"
echo "  archs    : $ARCHS_SWEEP"
echo "  modes    : $MODES_SWEEP"
echo "  subjects : $CIL_SUBJECTS"
if $SKIP_CIL_SWEEP; then
  echo "  [skipped]"
else
  touch /tmp/run_fresh_cil_sweep_start

  _run ./run_pipeline.sh \
      --data "$DATA" \
      --arch "$ARCH" \
      --scenario cil \
      --cil-subjects "$CIL_SUBJECTS" \
      --archs-sweep "$ARCHS_SWEEP" \
      --modes-sweep "$MODES_SWEEP" \
      --epochs "$EPOCHS" \
      --out-dir "$OUT_DIR" \
      --skip-hpo
fi

# ── Stage 7: Git commit — CIL HPO + eval results ─────────────────────────────
_banner "Stage 7 — Git commit: CIL HPO results + eval JSONs"
if $SKIP_GIT_FINAL; then
  echo "  [skipped]"
else
  _git_add_if_exists "$OUT_DIR/hpo_cl_ewc_stateful_${ARCH}_cil_best.json"
  _git_add_if_exists "$OUT_DIR/hpo_cl_replay_stateful_${ARCH}_cil_best.json"
  _git_add_if_exists "$OUT_DIR/hpo_cl_herding_stateful_${ARCH}_cil_best.json"

  if [[ -f /tmp/run_fresh_cil_sweep_start ]]; then
    echo "  Staging new CIL eval results..."
    NEW_EVALS=$(find "$OUT_DIR" -name 'eval_cil_*.json' \
                     -newer /tmp/run_fresh_cil_sweep_start 2>/dev/null)
    if [[ -n "$NEW_EVALS" ]]; then
      echo "$NEW_EVALS" | while read -r f; do
        echo "  git add $f"
        $DRY_RUN || git add "$f"
      done
    else
      echo "  (no new eval_cil_*.json files found)"
    fi
  fi

  _git_commit_push "feat: CIL HPO results and full sweep evaluations (${ARCH})"
fi

echo ""
echo "════════════════════════════════════════"
echo "run_fresh.sh complete."
echo "  Analyse results with:  python analysis/compute_metrics.py"
echo ""
echo "Closing SSH session in 5 seconds..."
sleep 5
kill -HUP $PPID
