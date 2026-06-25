#!/usr/bin/env bash
# Replay + Herding accuracies WITHOUT HPO.
#
# Runs replay_stateful and herding_stateful with train.py's built-in default
# hyperparameters (no Optuna, no tuned JSONs) — via run_pipeline.sh --use-defaults.
# Defaults used:
#   replay : replay_weight=0.5  replay_capacity=10000  replay_batch_size=16  noise_std=0.01
#   herding: capacity_per_class=20  replay_batch_size=16
#   arch   : d_model=128  num_layers=4  lr=1e-4  wd=0.01  bs=32   (train.py defaults)
#
# Usage:
#   ./run_replay_herding.sh                       # data ./NinaProData, hgrn, CIL+DIL
#   ./run_replay_herding.sh /path/to/data
#   ./run_replay_herding.sh /path/to/data "hgrn lstm"
#   nohup ./run_replay_herding.sh &               # detach
#
# Logs → logs/replay_herding_<timestamp>.log

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
ARCHS="${2:-hgrn}"
MODES="replay_stateful herding_stateful"
SUBJECTS="$(seq -s ' ' 1 10)"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/replay_herding_${TS}.log"
RESULTS="results_replay_herding"
mkdir -p logs

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

{
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Replay + Herding (no HPO) — started $TS"
echo "║  data=$DATA  archs=$ARCHS  modes=$MODES"
echo "╚══════════════════════════════════════════════════════════════╝"

# CIL — 10 subjects per arch/mode
./run_pipeline.sh --data "$DATA" --use-defaults \
  --archs-sweep "$ARCHS" --modes-sweep "$MODES" --cil-subjects "$SUBJECTS" \
  --results-dir "$RESULTS"

# DIL — one 27-subject stream per arch/mode
./run_pipeline.sh --data "$DATA" --use-defaults --scenario dil \
  --archs-sweep "$ARCHS" --modes-sweep "$MODES" \
  --results-dir "$RESULTS"

echo ""; echo "########## ACCURACIES (no HPO, default hyperparameters) ##########"
python3 - "$RESULTS" <<'PY'
import json, glob, os, sys, statistics as st
from collections import defaultdict

def metrics(f):
    d = json.load(open(f))
    fin = [v["accuracy"] for v in d["final_performance"].values()]
    tids = list(d["final_performance"].keys())
    forg = [max(0.0, d["immediate_performance"][t]["accuracy"] - d["final_performance"][t]["accuracy"])
            for t in tids[:-1]
            if t in d["immediate_performance"] and t in d["final_performance"]]
    m = d["metadata"]
    return (m["scenario"], m["arch"], m["mode"]), sum(fin)/len(fin), (sum(forg)/len(forg) if forg else 0.0)

agg = defaultdict(lambda: {"fin": [], "forg": []})
for f in glob.glob(os.path.join(sys.argv[1], "eval_*.json")):
    key, fin, forg = metrics(f)
    agg[key]["fin"].append(fin); agg[key]["forg"].append(forg)

print(f"{'scenario':<5} {'arch':<5} {'mode':<17} {'final_acc':>16} {'forget':>16} {'n':>3}")
for key in sorted(agg):
    v = agg[key]; n = len(v["fin"])
    fm, fs = st.mean(v["fin"]),  (st.pstdev(v["fin"])  if n > 1 else 0.0)
    gm, gs = st.mean(v["forg"]), (st.pstdev(v["forg"]) if n > 1 else 0.0)
    acc = f"{fm:.3f} ± {fs:.3f}" if n > 1 else f"{fm:.3f}"
    fg  = f"{gm:.3f} ± {gs:.3f}" if n > 1 else f"{gm:.3f}"
    print(f"{key[0]:<5} {key[1]:<5} {key[2]:<17} {acc:>16} {fg:>16} {n:>3}")
PY

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DONE — $(date '+%Y%m%d_%H%M%S')   files: $RESULTS/"
echo "╚══════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log: $LOG"
