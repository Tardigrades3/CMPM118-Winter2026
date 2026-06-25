#!/usr/bin/env bash
# DIL EWC error bars — re-run the DIL arm across several seeds, then report mean±std.
#
# The single-run DIL result (Variant B EWC = 0.658) carried the story but was n=1.
# This repeats DIL for both configs and both modes across SEEDS to put error bars
# on it. CIL is untouched (it was already n=10 and showed EWC ≈ naive).
#
#   Variant A — aran-HGRN HPO params (256/2, lr 4.8e-4, λ=727)
#   Variant B — original defaults     (128/4, lr 1e-4,  λ=2000)
#
# Usage:
#   ./run_dil_seeds.sh                      # data ./NinaProData, seeds "1 2 3 4 5"
#   ./run_dil_seeds.sh /path/to/data "1 2 3"
#   nohup ./run_dil_seeds.sh &              # detach
#
# Logs → logs/dil_seeds_<timestamp>.log

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
SEEDS="${2:-1 2 3 4 5}"
MODES="stateful ewc_stateful"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/dil_seeds_${TS}.log"
mkdir -p logs

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

DIR_A="results_dil_seeds_A"
DIR_B="results_dil_seeds_B"

{
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DIL seeded re-run — started $TS"
echo "║  data=$DATA  seeds=[$SEEDS]  modes=$MODES"
echo "╚══════════════════════════════════════════════════════════════╝"

./sync_hpo_params.sh || echo "WARN: sync failed — Variant A may use defaults"

for S in $SEEDS; do
  echo ""; echo "################  SEED $S  ################"

  echo "---- Variant A (HPO params) seed $S ----"
  ./run_pipeline.sh --data "$DATA" --skip-hpo --scenario dil \
    --archs-sweep hgrn --modes-sweep "$MODES" --seed "$S" \
    --results-dir "$DIR_A"

  echo "---- Variant B (defaults) seed $S ----"
  ./run_pipeline.sh --data "$DATA" --use-defaults --scenario dil \
    --archs-sweep hgrn --modes-sweep "$MODES" --seed "$S" \
    --results-dir "$DIR_B"
done

echo ""; echo "########## SUMMARY (mean ± std over seeds) ##########"
python3 - "$DIR_A" "$DIR_B" <<'PY'
import json, glob, os, sys, statistics as st
from collections import defaultdict

def metrics(f):
    d = json.load(open(f))
    fin = [v["accuracy"] for v in d["final_performance"].values()]
    tids = list(d["final_performance"].keys())
    forg = [max(0.0, d["immediate_performance"][t]["accuracy"] - d["final_performance"][t]["accuracy"])
            for t in tids[:-1]
            if t in d["immediate_performance"] and t in d["final_performance"]]
    return sum(fin)/len(fin), (sum(forg)/len(forg) if forg else 0.0), d["metadata"].get("seed")

for label, dirn in [("Variant A (HPO params 256/2 lr4.8e-4 λ727)", sys.argv[1]),
                    ("Variant B (defaults 128/4 lr1e-4 λ2000)",    sys.argv[2])]:
    agg = defaultdict(lambda: {"fin": [], "forg": [], "seeds": []})
    for f in glob.glob(os.path.join(dirn, "eval_*.json")):
        fin, forg, seed = metrics(f)
        mode = "ewc_stateful" if "ewc_stateful" in os.path.basename(f) else "stateful"
        agg[mode]["fin"].append(fin); agg[mode]["forg"].append(forg); agg[mode]["seeds"].append(seed)
    print(f"\n=== {label}  [{dirn}] ===")
    print(f"{'mode':<14} {'final_acc (mean±std)':<22} {'forget (mean±std)':<22} {'n'}")
    for mode in ("stateful", "ewc_stateful"):
        v = agg.get(mode)
        if not v or not v["fin"]:
            print(f"{mode:<14} (no runs)"); continue
        n = len(v["fin"])
        fm, fs = st.mean(v["fin"]),  (st.pstdev(v["fin"])  if n > 1 else 0.0)
        gm, gs = st.mean(v["forg"]), (st.pstdev(v["forg"]) if n > 1 else 0.0)
        print(f"{mode:<14} {fm:.3f} ± {fs:.3f}        {gm:.3f} ± {gs:.3f}        {n}")
    # EWC lift over naive
    if agg.get("ewc_stateful", {}).get("fin") and agg.get("stateful", {}).get("fin"):
        lift = st.mean(agg["ewc_stateful"]["fin"]) - st.mean(agg["stateful"]["fin"])
        print(f"  → EWC lift over naive stateful: {lift:+.3f}")
PY

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DONE — $(date '+%Y%m%d_%H%M%S')   files: $DIR_A/  $DIR_B/"
echo "╚══════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log: $LOG"
