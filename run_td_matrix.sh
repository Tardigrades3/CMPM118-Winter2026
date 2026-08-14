#!/usr/bin/env bash
# Fill the full TD arch×scenario×method matrix for the LSTM-vs-HGRN comparison.
#
# Runs {ARCHS} × {DIL over seeds, CIL over subjects} × {stateless, replay, ewc},
# all with --features td and the principled stateless modes (windowed EMG ⇒ no
# state carry). Seed/subject-matched so the HGRN-vs-LSTM comparison is fair.
#
# Default ARCHS="hgrn lstm" re-runs HGRN-TD too (cheap) so the whole matrix is
# consistent. If you trust the existing HGRN-DIL-TD runs, pass ARCHS="lstm".
#
# Outputs: DIL → results_dil_td/ , CIL → results_cil_td/  (arch is in metadata,
# so the analysis separates HGRN vs LSTM automatically).
#
# Usage:
#   ./run_td_matrix.sh                                  # hgrn+lstm, seeds 1-3, subj 1-10
#   ./run_td_matrix.sh lstm                             # just fill the LSTM gap
#   ./run_td_matrix.sh "hgrn lstm" /data "1 2 3" "1 2 3 4 5"
#   nohup ./run_td_matrix.sh lstm &

set -uo pipefail
export MPLBACKEND=Agg

ARCHS="${1:-hgrn lstm}"
DATA="${2:-./NinaProData}"
SEEDS="${3:-1 2 3}"
SUBJECTS="${4:-1 2 3 4 5 6 7 8 9 10}"
EPOCHS=5
MODES="stateless replay_stateless ewc_stateless"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/run_td_matrix_${TS}.log"
mkdir -p logs results_dil_td results_cil_td

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

_extra() {  # per-mode known-good params
  case "$1" in
    replay_stateless) echo "--lr 1e-3 --replay_weight 0.5 --replay_capacity 10000 --noise_std 0.01 --replay_batch_size 16" ;;
    ewc_stateless)    echo "--lr 3e-4 --ewc_lambda 30" ;;
    *)                echo "--lr 1e-3" ;;
  esac
}

{
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  TD matrix — archs=[$ARCHS]  seeds=[$SEEDS]  subj=[$SUBJECTS]"
echo "║  started $TS"
echo "╚════════════════════════════════════════════════════════════════╝"

for ARCH in $ARCHS; do
  # ── DIL: seeds × modes ──
  for S in $SEEDS; do
    for MODE in $MODES; do
      echo ""; echo "-- DIL | $ARCH | $MODE | seed $S --"
      python3 hgrn/train.py --arch "$ARCH" --scenario dil --exercise 1 --mode "$MODE" \
        --data_path "$DATA" --features td --d_model 128 --num_layers 4 --batch_size 24 \
        --epochs_per_task "$EPOCHS" --seed "$S" --results_dir results_dil_td $(_extra "$MODE")
    done
  done
  # ── CIL: subjects × modes ──
  for SUBJ in $SUBJECTS; do
    for MODE in $MODES; do
      echo ""; echo "-- CIL | $ARCH | $MODE | subject $SUBJ --"
      python3 hgrn/train.py --arch "$ARCH" --scenario cil --subject "$SUBJ" --mode "$MODE" \
        --data_path "$DATA" --features td --d_model 128 --num_layers 4 --batch_size 24 \
        --epochs_per_task "$EPOCHS" --results_dir results_cil_td $(_extra "$MODE")
    done
  done
done

# ── Ranked summary of the full TD matrix ─────────────────────────────────────
echo ""; echo "######## TD MATRIX — RANKED (arch × scenario × method) ########"
python3 - <<'PY'
import json, glob, os, statistics as st
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
    return (m.get("arch"),m.get("scenario"),m.get("mode"),m.get("features"),m.get("ewc_lambda"),
            st.mean(fin),(st.mean(forg) if forg else 0),(st.mean(late) if late else float('nan')))
cells=defaultdict(lambda:{"fin":[],"forg":[],"late":[]})
for d in ("results_dil_td","results_cil_td"):
    for f in glob.glob(os.path.join(d,"eval_*.json")):
        try: arch,scen,mode,feat,lam,fin,forg,late=analyse(f)
        except: continue
        if feat!="td": continue
        if mode=="ewc_stateless" and lam and lam>5000: continue   # skip frozen λ=12032
        k=(arch,scen,mode); cells[k]["fin"].append(fin); cells[k]["forg"].append(forg); cells[k]["late"].append(late)
out=[]
for (arch,scen,mode),v in cells.items():
    if not v["fin"]: continue
    n=len(v["fin"]); out.append((st.mean(v["fin"]),st.pstdev(v["fin"]) if n>1 else 0,
                                 st.mean(v["forg"]),st.mean([x for x in v["late"] if x==x]),arch,scen,mode,n))
out.sort(reverse=True)
print(f"{'rank':<5}{'arch':<6}{'scen':<5}{'mode':<18}{'final':<16}{'forg':<8}{'plast':<7}{'n'}")
for i,(fm,fs,gm,lt,arch,scen,mode,n) in enumerate(out,1):
    print(f"{i:<5}{arch:<6}{scen:<5}{mode:<18}{fm:.3f} ± {fs:.3f}    {gm:<8.3f}{lt:<7.2f}{n}")
if out:
    fm,fs,_,_,arch,scen,mode,n=out[0]
    print(f"\nBEST: {arch} / {scen} / {mode} = {fm:.3f} ± {fs:.3f}")
PY

echo ""; echo "DONE — $(date '+%Y%m%d_%H%M%S')"
} 2>&1 | tee "$LOG"
echo "Full log: $LOG"
