#!/usr/bin/env bash
# Copy tuned HPO best-param JSONs from another branch into this branch's results/.
#
# The tuned hyperparameters (arch + per-method CL params) are committed on the
# aran-HGRN branch but not on different_experiments. To compare the two EWC
# formulations fairly, this branch's sweep must reuse the SAME hyperparameters —
# so pull them in before running ./run_pipeline.sh --skip-hpo.
#
# Usage:
#   ./sync_hpo_params.sh                       # from aran-HGRN (or origin/aran-HGRN) → results/
#   ./sync_hpo_params.sh origin/aran-HGRN      # explicit source ref
#   ./sync_hpo_params.sh aran-HGRN results      # explicit ref + dest dir

set -euo pipefail

SRC_REF="${1:-aran-HGRN}"
DEST="${2:-results}"

# Fall back to the remote-tracking ref if the local branch is absent (fresh EC2 checkout).
if ! git rev-parse --verify --quiet "$SRC_REF" >/dev/null; then
  if git rev-parse --verify --quiet "origin/$SRC_REF" >/dev/null; then
    SRC_REF="origin/$SRC_REF"
  else
    echo "Error: neither '$SRC_REF' nor 'origin/$SRC_REF' exists. Run 'git fetch' first." >&2
    exit 1
  fi
fi

mkdir -p "$DEST"

FILES=(
  hpo_arch_hgrn_ex1_best.json
  hpo_cl_ewc_stateful_hgrn_cil_best.json
  hpo_cl_ewc_stateful_hgrn_dil_best.json
  hpo_cl_replay_stateful_hgrn_cil_best.json
  hpo_cl_replay_stateful_hgrn_dil_best.json
  hpo_cl_herding_stateful_hgrn_cil_best.json
  hpo_cl_herding_stateful_hgrn_dil_best.json
)

echo "Syncing tuned HPO params from '$SRC_REF' → $DEST/"
n=0
for f in "${FILES[@]}"; do
  if git show "$SRC_REF:results/$f" > "$DEST/$f" 2>/dev/null; then
    echo "  ✓ $f"
    n=$((n + 1))
  else
    echo "  ✗ $f (not found on $SRC_REF — skipped)"
    rm -f "$DEST/$f"
  fi
done

echo "Done — $n/${#FILES[@]} param files synced."
echo "Now run:  ./run_pipeline.sh --data ./NinaProData --skip-hpo"
