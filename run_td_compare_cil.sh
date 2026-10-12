#!/usr/bin/env bash
# CIL (class-incremental) with TD features, all methods, no HPO.
#
# CIL = one model per subject, learning Exercise 1 -> 2 -> 3 (growing label set,
# no task ID at test). This is where REPLAY/HERDING are expected to shine (they
# reintroduce old classes so the model can form cross-class boundaries) and EWC
# is expected to be weak (it anchors weights but can't create boundaries between
# classes never seen together).
#
# Methods (all --features td), run per subject:
#   stateless        — naive baseline (lower bound)
#   replay_stateless — experience replay (concatenates old-class exemplars)
#   ewc_stateless    — EWC, λ=30 (the plastic/fair value from the DIL sweep)
#   herding_stateful — herding exemplars + CIL feature-extractor freeze
#
# Outputs → results_cil_td/   (one eval_cil_<mode>_s<subj>_*.json per run)
#
# Usage:
#   ./run_td_compare_cil.sh                              # ./NinaProData, subjects 1..10
#   ./run_td_compare_cil.sh /path/to/data "1 2 3 4 5"
#   ./run_td_compare_cil.sh /path/to/data "1 2 3" "stateless replay_stateless"
#   nohup ./run_td_compare_cil.sh &

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
SUBJECTS="${2:-1 2 3 4 5 6 7 8 9 10}"
MODES="${3:-stateless replay_stateless ewc_stateless herding_stateful}"
EPOCHS=5
RESULTS="results_cil_td"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/run_td_compare_cil_${TS}.log"
mkdir -p logs "$RESULTS"

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

# Per-mode extra args (known-good / sensible defaults; no HPO).
_extra_for_mode() {
  case "$1" in
    replay_stateless) echo "--lr 1e-3 --replay_weight 0.5 --replay_capacity 10000 --noise_std 0.01 --replay_batch_size 16" ;;
    ewc_stateless)    echo "--lr 3e-4 --ewc_lambda 30" ;;
    herding_stateful) echo "--lr 1e-3 --herding_capacity_per_class 20 --replay_batch_size 16" ;;
    *)                echo "--lr 1e-3" ;;   # stateless / stateful baselines
  esac
}

COMMON="--arch hgrn --scenario cil --data_path $DATA --features td \
        --d_model 128 --num_layers 4 --batch_size 24 \
        --epochs_per_task $EPOCHS --results_dir $RESULTS"

{
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  CIL + TD comparison — started $TS"
echo "║  data=$DATA  subjects=[$SUBJECTS]  modes=[$MODES]"
echo "╚════════════════════════════════════════════════════════════════╝"

for SUBJ in $SUBJECTS; do
  echo ""; echo "===== subject $SUBJ ====="
  for MODE in $MODES; do
    EXTRA=$(_extra_for_mode "$MODE")
    echo "-- subject $SUBJ | $MODE | $EXTRA --"
    python3 hgrn/train.py $COMMON --subject "$SUBJ" --mode "$MODE" $EXTRA
  done
done

# ── Summary: per-mode mean over subjects (final / forgetting / late-train) ────
echo ""; echo "######## CIL + TD RESULTS (mean ± std over subjects) ########"
python3 - "$RESULTS" <<'PY'
import json, glob, os, sys, statistics as st
from collections import defaultdict
def analyse(f):
    d=json.load(open(f)); m=d["metadata"]
    fin=[v["accuracy"] for v in d["final_performance"].values()]
    imm=[v["accuracy"] for v in d["immediate_performance"].values()]
    tids=list(d["final_performance"].keys())
    forg=[max(0.0,d["immediate_performance"][t]["accuracy"]-d["final_performance"][t]["accuracy"])
          for t in tids[:-1] if t in d["immediate_performance"]]
    th=d.get("training_history",{})
    late=[th[t]["epoch_accuracies"][-1] for t in list(th.keys()) if th[t].get("epoch_accuracies")]
    return (m["mode"], st.mean(fin), (st.mean(forg) if forg else 0.0),
            (st.mean(late) if late else float('nan')))
agg=defaultdict(lambda:{"fin":[],"forg":[],"late":[]})
for f in glob.glob(os.path.join(sys.argv[1],"eval_cil_*.json")):
    m,fin,forg,late=analyse(f)
    agg[m]["fin"].append(fin); agg[m]["forg"].append(forg); agg[m]["late"].append(late)
print(f"{'mode':<20}{'final (mean±std)':<22}{'forget':<12}{'late-train':<12}{'n_subj'}")
order=["stateless","stateful","replay_stateless","replay_stateful","ewc_stateless","ewc_stateful","herding_stateful"]
for mode in order:
    v=agg.get(mode)
    if not v or not v["fin"]: continue
    n=len(v["fin"]); fm=st.mean(v["fin"]); fs=st.pstdev(v["fin"]) if n>1 else 0
    gm=st.mean(v["forg"]); lt=st.mean([x for x in v["late"] if x==x])
    flag="OK" if lt>0.8 else ("FROZEN" if lt<0.6 else "margnl")
    print(f"{mode:<20}{fm:.3f} ± {fs:.3f}        {gm:<12.3f}{lt:.2f} {flag:<6}  {n}")
print("\nExpectation: replay/herding > ewc for CIL (they recreate old-class boundaries).")
print("Compare absolute level vs the raw-feature CIL runs if you have them.")
PY

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  DONE — $(date '+%Y%m%d_%H%M%S')   eval JSONs: $RESULTS/"
echo "╚════════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log: $LOG"
