#!/usr/bin/env bash
# Recover (and beat) strong DIL results using stateless CL modes.
#
# Why stateless:
#   - Windowed EMG samples are independent; carrying state across them saturates
#     the recurrent state and causes majority-class collapse (all stateful modes
#     flatline at ~0.748 transfer ceiling, regardless of seed or method).
#   - Stateless modes actually learn: loss descends, accuracy climbs.
#   - Old ~0.71 replay result was with lr=1e-3, bs=24 — CIL-proxy HPO degraded
#     it by freezing lr at ~5e-4. This script co-tunes lr ON a DIL proxy.
#
# Modes run:
#   stateless        — naive fine-tuning baseline (stateless)
#   replay_stateless — experience replay, stateless (target: beat 0.71)
#   ewc_stateless    — EWC with per-batch state reset (principled variant)
#
# Usage:
#   ./tune_dil.sh                                    # data ./NinaProData, seeds "1 2 3"
#   ./tune_dil.sh /path/to/data "1 2 3 4 5"
#   ./tune_dil.sh /path/to/data "1 2 3" "27 4 17 23 3 24" 40
#   nohup ./tune_dil.sh &                            # detach
#
# Logs → logs/tune_dil_<timestamp>.log ; eval JSONs → results_dil_recovered/

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
SEEDS="${2:-1 2 3}"
HPO_SUBJ="${3:-27 4 17 23 3 24}"     # 6-subject DIL proxy
N_TRIALS="${4:-40}"
EPOCHS=5
HPO_MODES="replay_stateless ewc_stateless"
OUT_DIR="results"                     # HPO best-param JSONs land here
RESULTS="results_dil_recovered"       # final sweep eval JSONs (kept separate)
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/tune_dil_${TS}.log"
mkdir -p logs "$RESULTS"

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

# Build train.py args from a CL best-param JSON.
_args_from_json() {
  python3 -c "
import json, sys
p = json.load(open(sys.argv[1]))['params']
mode = sys.argv[2]
out  = ['--d_model','128','--num_layers','4']
out += ['--lr', str(p.get('lr', 1e-3))]
out += ['--batch_size', str(int(p.get('batch_size', 24)))]
out += ['--weight_decay', str(p.get('weight_decay', 0.01))]
if 'ewc' in mode:
    out += ['--ewc_lambda', str(p.get('ewc_lambda', 2000))]
elif 'replay' in mode:
    out += ['--replay_weight',     str(p.get('replay_weight', 0.5)),
            '--replay_capacity',   str(int(p.get('capacity', 10000))),
            '--noise_std',         str(p.get('noise_std', 0.01)),
            '--replay_batch_size', str(int(p.get('replay_batch_size', 16)))]
print(' '.join(out))
" "$1" "$2"
}

{
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  DIL recovery (stateless) — started $TS"
echo "║  data=$DATA  seeds=[$SEEDS]  proxy=[$HPO_SUBJ]  trials=$N_TRIALS"
echo "╚════════════════════════════════════════════════════════════════╝"

# ── Phase 1: DIL-aware HPO (co-tune lr on a DIL proxy) ───────────────────────
for MODE in $HPO_MODES; do
  echo ""; echo "######## HPO (DIL stateless, --tune-lr): $MODE ########"
  python3 hpo.py cl \
    --mode "$MODE" --scenario dil --exercise 1 \
    --subjects $HPO_SUBJ --data_path "$DATA" --arch hgrn \
    --tune-lr --epochs_per_task "$EPOCHS" --n_trials "$N_TRIALS" --out_dir "$OUT_DIR"
done

echo ""; echo "Discovered DIL configs:"
for MODE in $HPO_MODES; do
  J="$OUT_DIR/hpo_cl_${MODE}_hgrn_dil_best.json"
  [ -f "$J" ] && echo "  $MODE → $(python3 -c "import json;print(json.load(open('$J'))['params'])")"
done

# ── Phase 2: full 27-subject DIL sweep with discovered configs, across seeds ──
echo ""; echo "######## Full DIL sweep (27 subjects) × seeds ########"
for S in $SEEDS; do
  # naive stateless baseline at a comparable high-lr config
  echo "-- seed $S | stateless (naive baseline) --"
  python3 hgrn/train.py --arch hgrn --scenario dil --exercise 1 --mode stateless \
    --data_path "$DATA" --d_model 128 --num_layers 4 --lr 1e-3 --batch_size 24 \
    --epochs_per_task "$EPOCHS" --seed "$S" --results_dir "$RESULTS"

  for MODE in $HPO_MODES; do
    J="$OUT_DIR/hpo_cl_${MODE}_hgrn_dil_best.json"
    if [ ! -f "$J" ]; then
      echo "  WARNING: $J not found, skipping $MODE seed $S"; continue
    fi
    EXTRA=$(_args_from_json "$J" "$MODE")
    echo "-- seed $S | $MODE | $EXTRA --"
    python3 hgrn/train.py --arch hgrn --scenario dil --exercise 1 --mode "$MODE" \
      --data_path "$DATA" --epochs_per_task "$EPOCHS" --seed "$S" \
      --results_dir "$RESULTS" $EXTRA
  done
done

# ── Phase 3: summary (mean ± std over seeds) ──────────────────────────────────
echo ""; echo "######## RESULTS (mean ± std over seeds) ########"
python3 - "$RESULTS" <<'PY'
import json, glob, os, sys, statistics as st
from collections import defaultdict
def metrics(f):
    d=json.load(open(f)); fin=[v["accuracy"] for v in d["final_performance"].values()]
    tids=list(d["final_performance"].keys())
    forg=[max(0.0,d["immediate_performance"][t]["accuracy"]-d["final_performance"][t]["accuracy"])
          for t in tids[:-1] if t in d["immediate_performance"]]
    return d["metadata"]["mode"], sum(fin)/len(fin), (sum(forg)/len(forg) if forg else 0.0)
agg=defaultdict(lambda:{"fin":[],"forg":[]})
for f in glob.glob(os.path.join(sys.argv[1],"eval_dil_*.json")):
    m,fin,forg=metrics(f); agg[m]["fin"].append(fin); agg[m]["forg"].append(forg)
print(f"{'mode':<20}{'final_acc (mean±std)':<26}{'forget (mean±std)':<24}{'n'}")
base=None
for mode in ("stateless","replay_stateless","ewc_stateless"):
    v=agg.get(mode)
    if not v or not v["fin"]:
        print(f"{mode:<20}(no runs)"); continue
    n=len(v["fin"]); fm=st.mean(v["fin"]); fs=st.pstdev(v["fin"]) if n>1 else 0.0
    gm=st.mean(v["forg"]); gs=st.pstdev(v["forg"]) if n>1 else 0.0
    if mode=="stateless": base=fm
    print(f"{mode:<20}{fm:.3f} ± {fs:.3f}            {gm:.3f} ± {gs:.3f}        {n}")
if base is not None:
    for mode in ("replay_stateless","ewc_stateless"):
        if agg.get(mode,{}).get("fin"):
            print(f"  → {mode} lift over naive: {st.mean(agg[mode]['fin'])-base:+.3f}")
PY

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  DONE — $(date '+%Y%m%d_%H%M%S')   eval JSONs: $RESULTS/"
echo "║  Target to beat: old replay-DIL ≈ 0.71 final / 0.073 forgetting"
echo "╚════════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log: $LOG"
