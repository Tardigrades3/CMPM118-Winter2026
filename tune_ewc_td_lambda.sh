#!/usr/bin/env bash
# Fix the collapsing EWC-TD run by finding a λ that fits the TD feature space.
#
# Why it collapsed: λ=12032 was tuned on RAW (10-dim) features. TD features are
# 50-dim and standardised, so Fisher magnitudes differ and the same λ over-
# penalises — the accumulated penalty freezes the net after ~7 subjects (training
# loss stops descending, train acc stuck ~0.3-0.5 on later subjects).
#
# This sweeps λ on the FULL 27-subject DIL run (where the freeze actually appears
# — a short proxy would miss it), seed 1, and prints a COLLAPSE DETECTOR:
# the mean final-epoch TRAIN accuracy on the last 10 subjects. Healthy training
# keeps that ~0.9; a frozen net shows ~0.4. Pick the largest λ that stays healthy
# (max regularisation without freezing), then run multi-seed at that value.
#
# Usage:
#   ./tune_ewc_td_lambda.sh                                  # λ in {30 100 300 1000 3000}
#   ./tune_ewc_td_lambda.sh /path/to/data "100 300 1000"    # custom data + λ list
#   nohup ./tune_ewc_td_lambda.sh &

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
LAMBDAS="${2:-30 100 300 1000 3000}"
SEED="${3:-1}"
EPOCHS=5
RESULTS="results_dil_td_lambda"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/tune_ewc_td_lambda_${TS}.log"
mkdir -p logs "$RESULTS"

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

{
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  EWC-TD λ sweep — started $TS"
echo "║  data=$DATA  λ=[$LAMBDAS]  seed=$SEED  (full 27-subject DIL)"
echo "╚════════════════════════════════════════════════════════════════╝"

for LAM in $LAMBDAS; do
  echo ""; echo "######## ewc_stateless TD | λ=$LAM ########"
  python3 hgrn/train.py --arch hgrn --scenario dil --exercise 1 --mode ewc_stateless \
    --data_path "$DATA" --features td --d_model 128 --num_layers 4 \
    --lr 3e-4 --batch_size 24 --ewc_lambda "$LAM" \
    --epochs_per_task "$EPOCHS" --seed "$SEED" --results_dir "$RESULTS"
done

# ── Summary: final acc + forgetting + COLLAPSE DETECTOR per λ ─────────────────
echo ""; echo "######## EWC-TD λ SWEEP RESULTS ########"
python3 - "$RESULTS" <<'PY'
import json, glob, os, sys, statistics as st

def analyse(f):
    d=json.load(open(f)); m=d["metadata"]
    fin=[v["accuracy"] for v in d["final_performance"].values()]
    imm=[v["accuracy"] for v in d["immediate_performance"].values()]
    fwd=[v["accuracy"] for v in d.get("forward_performance",{}).values()]
    tids=list(d["final_performance"].keys())
    forg=[max(0.0,d["immediate_performance"][t]["accuracy"]-d["final_performance"][t]["accuracy"])
          for t in tids[:-1] if t in d["immediate_performance"]]
    # COLLAPSE DETECTOR: final-epoch train acc on the last 10 subjects.
    th=d.get("training_history",{})
    last10=list(th.keys())[-10:]
    late_train=[th[t]["epoch_accuracies"][-1] for t in last10 if th[t].get("epoch_accuracies")]
    return (m.get("ewc_lambda"),
            st.mean(fin), (st.mean(forg) if forg else 0.0),
            st.mean(imm), (st.mean(fwd) if fwd else float('nan')),
            (st.mean(late_train) if late_train else float('nan')))

rows=[analyse(f) for f in glob.glob(os.path.join(sys.argv[1],"eval_dil_ewc_stateless_*.json"))]
rows.sort(key=lambda r: r[0] if r[0] is not None else 0)
print(f"{'lambda':<10}{'final':<9}{'forget':<9}{'imm':<8}{'zero-shot':<11}{'late-train':<12}{'status'}")
for lam,fin,forg,imm,zs,late in rows:
    status = "OK" if late>0.80 else ("FROZEN" if late<0.60 else "marginal")
    print(f"{lam:<10.0f}{fin:<9.3f}{forg:<9.3f}{imm:<8.3f}{zs:<11.3f}{late:<12.3f}{status}")
print("\nRaw EWC (3-seed): 0.691 | TD replay (n=1): 0.758 | broken EWC-TD (λ=12032): 0.70-0.72")
print("Pick the LARGEST λ with status=OK (late-train ~0.9 = net still learning late subjects),")
print("then run that λ across seeds 1/2/3 for the final EWC-TD number.")
PY

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  DONE — $(date '+%Y%m%d_%H%M%S')   eval JSONs: $RESULTS/"
echo "╚════════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log: $LOG"
