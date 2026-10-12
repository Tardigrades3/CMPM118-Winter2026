#!/usr/bin/env bash
# Run the stateless DIL modes with TD (time-domain Hudgins) feature preprocessing,
# using the KNOWN-GOOD hand params (no HPO — the 6-subject proxy keeps degrading
# the full-sweep result, so we skip it and use the configs that actually worked).
#
# Modes (all --features td), 3 seeds each:
#   stateless        — naive baseline           (lr=1e-3)
#   replay_stateless — replay, rw=0.5/noise=0.01 (the config that hit 0.691/0.064 raw)
#   ewc_stateless    — EWC, lambda=12032         (raw HPO best; reasonable TD starting pt)
#
# Outputs → results_dil_td/   (raw equivalents live in results_dil_recovered/)
# Compare the two with the printed summary, especially the imm-zs column.
#
# Usage:
#   ./run_td_compare.sh                       # ./NinaProData, seeds "1 2 3"
#   ./run_td_compare.sh /path/to/data "1 2 3"
#   nohup ./run_td_compare.sh &

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
SEEDS="${2:-1 2 3}"
EPOCHS=5
RESULTS="results_dil_td"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/run_td_compare_${TS}.log"
mkdir -p logs "$RESULTS"

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

COMMON="--arch hgrn --scenario dil --exercise 1 --data_path $DATA \
        --features td --d_model 128 --num_layers 4 --batch_size 24 \
        --epochs_per_task $EPOCHS --results_dir $RESULTS"

{
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  TD-feature DIL comparison — started $TS"
echo "║  data=$DATA  seeds=[$SEEDS]  (no HPO — known-good params)"
echo "╚════════════════════════════════════════════════════════════════╝"

for S in $SEEDS; do
  echo ""; echo "===== seed $S ====="

  echo "-- stateless (naive baseline, TD) --"
  python3 hgrn/train.py $COMMON --mode stateless --lr 1e-3 --seed "$S"

  echo "-- replay_stateless (TD) --"
  python3 hgrn/train.py $COMMON --mode replay_stateless --lr 1e-3 \
    --replay_weight 0.5 --replay_capacity 10000 --noise_std 0.01 --replay_batch_size 16 \
    --seed "$S"

  echo "-- ewc_stateless (TD) --"
  python3 hgrn/train.py $COMMON --mode ewc_stateless --lr 3e-4 \
    --ewc_lambda 12032 --seed "$S"
done

# ── Summary: final / forgetting / immediate / zero-shot / imm-zs ──────────────
echo ""; echo "######## TD RESULTS (mean ± std over seeds) ########"
python3 - "$RESULTS" <<'PY'
import json, glob, os, sys, statistics as st
from collections import defaultdict
def metrics(f):
    d=json.load(open(f))
    fin=[v["accuracy"] for v in d["final_performance"].values()]
    imm=[v["accuracy"] for v in d["immediate_performance"].values()]
    fwd=[v["accuracy"] for v in d.get("forward_performance",{}).values()]
    tids=list(d["final_performance"].keys())
    forg=[max(0.0,d["immediate_performance"][t]["accuracy"]-d["final_performance"][t]["accuracy"])
          for t in tids[:-1] if t in d["immediate_performance"]]
    return (d["metadata"]["mode"], sum(fin)/len(fin),
            (sum(forg)/len(forg) if forg else 0.0),
            sum(imm)/len(imm), (sum(fwd)/len(fwd) if fwd else float('nan')))
agg=defaultdict(lambda:{"fin":[],"forg":[],"imm":[],"zs":[]})
for f in glob.glob(os.path.join(sys.argv[1],"eval_dil_*stateless*.json")):
    m,fin,forg,imm,zs=metrics(f)
    agg[m]["fin"].append(fin); agg[m]["forg"].append(forg)
    agg[m]["imm"].append(imm); agg[m]["zs"].append(zs)
print(f"{'mode':<18}{'final':<16}{'forget':<16}{'imm':<8}{'zero-shot':<11}{'imm-zs'}")
base=None
for mode in ("stateless","replay_stateless","ewc_stateless"):
    v=agg.get(mode)
    if not v or not v["fin"]:
        print(f"{mode:<18}(no runs)"); continue
    n=len(v["fin"])
    fm,fs=st.mean(v["fin"]),(st.pstdev(v["fin"]) if n>1 else 0)
    gm,gs=st.mean(v["forg"]),(st.pstdev(v["forg"]) if n>1 else 0)
    im=st.mean(v["imm"]); zs=st.mean([z for z in v["zs"] if z==z])
    if mode=="stateless": base=fm
    print(f"{mode:<18}{fm:.3f} ± {fs:.3f}    {gm:.3f} ± {gs:.3f}    {im:.3f}   {zs:.3f}      {im-zs:+.3f}")
if base is not None:
    for mode in ("replay_stateless","ewc_stateless"):
        if agg[mode]["fin"]:
            print(f"  -> {mode} lift over naive: {st.mean(agg[mode]['fin'])-base:+.3f}")
print("\nCompare vs RAW (results_dil_recovered/):  stateless 0.548 | replay 0.668 | ewc 0.691")
print("Key question: does TD RAISE final accuracy, and does imm-zs WIDEN")
print("(features broke the degeneracy) or stay flat (degeneracy is representation-independent)?")
PY

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  DONE (TD) — $(date '+%Y%m%d_%H%M%S')   eval JSONs: $RESULTS/"
echo "╚════════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log: $LOG"
