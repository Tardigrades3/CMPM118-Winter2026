#!/usr/bin/env bash
# CRITICAL CONTROL: disentangle "stateful collapse" from "no-shuffle collapse".
#
# In all prior runs, state-handling and shuffling were confounded:
#   stateful  => shuffle=False   (collapses)
#   stateless => shuffle=True    (learns)
# So we cannot tell whether the collapse is caused by carrying recurrent state
# (our claim) or simply by training without shuffling on rest-dominated, ordered
# EMG data. This runs the full 2x2 to separate them, DIL, both archs, seed 1.
#
#   A) stateful  + shuffle=TRUE   <- NEW: does state-carry collapse when shuffled?
#   B) stateless + shuffle=FALSE  <- NEW: does no-shuffle collapse without state?
#   C) stateful  + shuffle=FALSE  <- have it (collapses) — rerun for clean compare
#   D) stateless + shuffle=TRUE   <- have it (learns)    — rerun for clean compare
#
# Interpretation:
#   If A learns and B collapses  -> the collapse is the SHUFFLE, not the state.
#                                   (Major correction: "stateful collapse" is a
#                                    no-shuffle artifact.)
#   If A collapses               -> carrying state genuinely collapses; narrative
#                                   holds and the 6/24 plastic run was an artifact.
#
# Outputs -> results_shuffle_ablation/   (metadata records mode + we tag via dir)
#
# Usage on EC2:
#   nohup ./disentangle_shuffle_state.sh ./NinaProData "1" > logs/disentangle.out 2>&1 &
#   tail -f logs/disentangle.out

set -uo pipefail
export MPLBACKEND=Agg
DATA="${1:-./NinaProData}"
SEEDS="${2:-1}"
RES="results_shuffle_ablation"
EP=5
mkdir -p logs "$RES"
[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

COMMON="--scenario dil --exercise 1 --data_path $DATA --d_model 128 --num_layers 4 \
        --batch_size 24 --epochs_per_task $EP --lr 1e-3 --results_dir $RES"

run() { echo ""; echo ">>> $*"; python3 hgrn/train.py $COMMON "$@"; }

for A in hgrn lstm; do
  for S in $SEEDS; do
    run --arch "$A" --mode stateful  --shuffle true  --seed "$S"   # A
    run --arch "$A" --mode stateless --shuffle false --seed "$S"   # B
    run --arch "$A" --mode stateful  --shuffle false --seed "$S"   # C (control)
    run --arch "$A" --mode stateless --shuffle true  --seed "$S"   # D (control)
  done
done

echo ""; echo "######## 2x2 SHUFFLE x STATE — plasticity summary ########"
python3 - "$RES" <<'PY'
import json, glob, os, sys, statistics as st
from collections import defaultdict
agg=defaultdict(list)
for f in glob.glob(os.path.join(sys.argv[1],"eval_dil_*.json")):
    d=json.load(open(f)); m=d["metadata"]
    th=d.get("training_history",{})
    late=[th[t]["epoch_accuracies"][-1] for t in list(th.keys())[-10:] if th[t].get("epoch_accuracies")]
    fin=[d["final_performance"][t]["accuracy"] for t in d["final_performance"]]
    state = "stateful" if "stateful" in m.get("mode","") else "stateless"
    shuf  = m.get("shuffle_train")
    agg[(m.get("arch"),state,shuf)].append((st.mean(late) if late else float('nan'), st.mean(fin)))
print(f"{'arch':<6}{'state':<11}{'shuffle':<9}{'plast':<9}{'final':<9}{'status'}")
for k in sorted(agg, key=lambda x:(x[0],x[1],str(x[2]))):
    pl=st.mean(x[0] for x in agg[k]); fn=st.mean(x[1] for x in agg[k])
    print(f"{k[0]:<6}{k[1]:<11}{str(k[2]):<9}{pl:<9.3f}{fn:<9.3f}{'FROZEN' if pl<0.6 else 'healthy'}")
print("\nKEY: if (stateful, shuffle=True) is HEALTHY and (stateless, shuffle=False)")
print("is FROZEN -> the collapse is the SHUFFLE, not the recurrent state.")
PY
echo ""; echo "DONE — $(date '+%H:%M:%S')"
