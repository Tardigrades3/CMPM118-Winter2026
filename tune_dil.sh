#!/usr/bin/env bash
# Recover the strong DIL results (replay ~0.71 / low forgetting, and a real EWC lift)
# with a DIL-aware HPO strategy. Run this, then step away.
#
# Why the old DIL results were lost:
#   - The CL-method HPO FROZE the learning rate at the CIL/stateless-tuned ~5e-4,
#     but the best DIL replay run used lr=1e-3. So the tuned configs underperformed.
#   - aran-HGRN used single-task EWC, which gives ~zero lift in DIL.
#
# What this does differently:
#   1. Online (accumulated) EWC                      [code, already applied]
#   2. CL-HPO with --tune-lr on a DIL proxy          → co-tunes lr/batch/weight_decay
#      WITH the method params, so it rediscovers the high-lr regime DIL replay wants.
#   3. d_model=128, num_layers=4 (the good-run arch) — NOT the CIL-tuned 256/2.
#   4. Full 27-subject DIL sweep with the discovered configs, across seeds, + a naive
#      baseline, so the replay/EWC lift over naive is measured with error bars.
#
# Usage:
#   ./tune_dil.sh                                   # data ./NinaProData, seeds "1 2 3"
#   ./tune_dil.sh /path/to/data "1 2 3 4 5"
#   ./tune_dil.sh /path/to/data "1 2 3" "27 4 17 23 3 24" 40
#   nohup ./tune_dil.sh &                            # detach
#
# Logs → logs/tune_dil_<timestamp>.log ; eval JSONs → results_dil_recovered/

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
SEEDS="${2:-1 2 3}"
HPO_SUBJ="${3:-27 4 17 23 3 24}"     # 6-subject DIL proxy (broader than the old 3)
N_TRIALS="${4:-40}"
EPOCHS=5
HPO_MODES="replay_stateful ewc_stateful"
OUT_DIR="results"                     # HPO best-param JSONs land here
RESULTS="results_dil_recovered"       # final sweep eval JSONs (kept separate)
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/tune_dil_${TS}.log"
mkdir -p logs "$RESULTS"

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

# Build train.py args (arch knobs + method params) from a CL best-param JSON.
_args_from_json() {
  python3 -c "
import json, sys
p = json.load(open(sys.argv[1]))['params']
mode = sys.argv[2]
out  = ['--d_model','128','--num_layers','4']
out += ['--lr', str(p.get('lr', 1e-3))]
out += ['--batch_size', str(int(p.get('batch_size', 24)))]
out += ['--weight_decay', str(p.get('weight_decay', 0.01))]
if mode == 'ewc_stateful':
    out += ['--ewc_lambda', str(p.get('ewc_lambda', 2000))]
elif mode.startswith('replay'):
    out += ['--replay_weight',     str(p.get('replay_weight', 0.5)),
            '--replay_capacity',   str(int(p.get('capacity', 10000))),
            '--noise_std',         str(p.get('noise_std', 0.01)),
            '--replay_batch_size', str(int(p.get('replay_batch_size', 16)))]
print(' '.join(out))
" "$1" "$2"
}

{
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  DIL recovery — started $TS"
echo "║  data=$DATA  seeds=[$SEEDS]  proxy=[$HPO_SUBJ]  trials=$N_TRIALS"
echo "╚════════════════════════════════════════════════════════════════╝"

# ── Phase 1: DIL-aware HPO (co-tune lr) ───────────────────────────────────────
for MODE in $HPO_MODES; do
  echo ""; echo "######## HPO (DIL, --tune-lr): $MODE ########"
  # No --arch_params → d_model=128 num_layers=4 defaults (the good-run arch).
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
  # naive baseline at a comparable high-lr config (for the lift comparison)
  echo "-- seed $S | stateful (naive baseline) --"
  python3 hgrn/train.py --arch hgrn --scenario dil --exercise 1 --mode stateful \
    --data_path "$DATA" --d_model 128 --num_layers 4 --lr 1e-3 --batch_size 24 \
    --epochs_per_task "$EPOCHS" --seed "$S" --results_dir "$RESULTS"

  for MODE in $HPO_MODES; do
    EXTRA=$(_args_from_json "$OUT_DIR/hpo_cl_${MODE}_hgrn_dil_best.json" "$MODE")
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
print(f"{'mode':<16}{'final_acc (mean±std)':<24}{'forget (mean±std)':<24}{'n'}")
base=None
for mode in ("stateful","ewc_stateful","replay_stateful"):
    v=agg.get(mode)
    if not v or not v["fin"]:
        print(f"{mode:<16}(no runs)"); continue
    n=len(v["fin"]); fm=st.mean(v["fin"]); fs=st.pstdev(v["fin"]) if n>1 else 0.0
    gm=st.mean(v["forg"]); gs=st.pstdev(v["forg"]) if n>1 else 0.0
    if mode=="stateful": base=fm
    print(f"{mode:<16}{fm:.3f} ± {fs:.3f}        {gm:.3f} ± {gs:.3f}        {n}")
if base is not None:
    for mode in ("ewc_stateful","replay_stateful"):
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
