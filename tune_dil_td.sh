#!/usr/bin/env bash
# TD-feature variant of tune_dil.sh — same stateless DIL pipeline, but feeds the
# model time-domain (Hudgins) feature sequences instead of raw EMG windows.
#
# Purpose: test whether a robust input representation RAISES the held-out ceiling,
# and — the more important question — whether it BREAKS the transfer degeneracy
# (does immediate-on-subject finally exceed zero-shot transfer?).
#
# Outputs are kept fully separate from the raw run so you can compare directly:
#   HPO studies/JSONs : results/hpo_cl_<mode>_hgrn_dil_td_best.json   (note _td)
#   eval JSONs        : results_dil_td/
#
# Usage (identical args to tune_dil.sh):
#   ./tune_dil_td.sh                                   # ./NinaProData, seeds "1 2 3"
#   ./tune_dil_td.sh /path/to/data "1 2 3" "27 4 17 23 3 24" 40
#   nohup ./tune_dil_td.sh &

set -uo pipefail
export MPLBACKEND=Agg

DATA="${1:-./NinaProData}"
SEEDS="${2:-1 2 3}"
HPO_SUBJ="${3:-27 4 17 23 3 24}"
N_TRIALS="${4:-40}"
EPOCHS=5
HPO_MODES="replay_stateless ewc_stateless"
FEATURES="td"
OUT_DIR="results"
RESULTS="results_dil_td"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG="logs/tune_dil_td_${TS}.log"
mkdir -p logs "$RESULTS"

[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate

# Build train.py args from a CL best-param JSON (adds --features td).
_args_from_json() {
  python3 -c "
import json, sys
p = json.load(open(sys.argv[1]))['params']
mode = sys.argv[2]
out  = ['--d_model','128','--num_layers','4','--features','td']
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
echo "║  DIL recovery (stateless, TD features) — started $TS"
echo "║  data=$DATA  seeds=[$SEEDS]  proxy=[$HPO_SUBJ]  trials=$N_TRIALS"
echo "╚════════════════════════════════════════════════════════════════╝"

# ── Phase 1: DIL-aware HPO on TD features (co-tune lr) ────────────────────────
for MODE in $HPO_MODES; do
  echo ""; echo "######## HPO (DIL stateless TD, --tune-lr): $MODE ########"
  python3 hpo.py cl \
    --mode "$MODE" --scenario dil --exercise 1 \
    --subjects $HPO_SUBJ --data_path "$DATA" --arch hgrn \
    --features "$FEATURES" \
    --tune-lr --epochs_per_task "$EPOCHS" --n_trials "$N_TRIALS" --out_dir "$OUT_DIR"
done

echo ""; echo "Discovered DIL (TD) configs:"
for MODE in $HPO_MODES; do
  J="$OUT_DIR/hpo_cl_${MODE}_hgrn_dil_td_best.json"
  [ -f "$J" ] && echo "  $MODE → $(python3 -c "import json;print(json.load(open('$J'))['params'])")"
done

# ── Phase 2: full 27-subject DIL sweep (TD), across seeds ─────────────────────
echo ""; echo "######## Full DIL sweep (27 subjects, TD) × seeds ########"
for S in $SEEDS; do
  echo "-- seed $S | stateless (naive baseline, TD) --"
  python3 hgrn/train.py --arch hgrn --scenario dil --exercise 1 --mode stateless \
    --data_path "$DATA" --features "$FEATURES" \
    --d_model 128 --num_layers 4 --lr 1e-3 --batch_size 24 \
    --epochs_per_task "$EPOCHS" --seed "$S" --results_dir "$RESULTS"

  for MODE in $HPO_MODES; do
    J="$OUT_DIR/hpo_cl_${MODE}_hgrn_dil_td_best.json"
    if [ ! -f "$J" ]; then
      echo "  WARNING: $J not found, skipping $MODE seed $S"; continue
    fi
    EXTRA=$(_args_from_json "$J" "$MODE")
    echo "-- seed $S | $MODE (TD) | $EXTRA --"
    python3 hgrn/train.py --arch hgrn --scenario dil --exercise 1 --mode "$MODE" \
      --data_path "$DATA" --epochs_per_task "$EPOCHS" --seed "$S" \
      --results_dir "$RESULTS" $EXTRA
  done
done

# ── Phase 3: summary, incl. zero-shot vs immediate (the degeneracy test) ──────
echo ""; echo "######## RESULTS (TD; mean ± std over seeds) ########"
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
            (sum(imm)/len(imm) if imm else 0.0),
            (sum(fwd)/len(fwd) if fwd else float('nan')))
agg=defaultdict(lambda:{"fin":[],"forg":[],"imm":[],"zs":[]})
for f in glob.glob(os.path.join(sys.argv[1],"eval_dil_*.json")):
    m,fin,forg,imm,zs=metrics(f)
    agg[m]["fin"].append(fin); agg[m]["forg"].append(forg)
    agg[m]["imm"].append(imm); agg[m]["zs"].append(zs)
print(f"{'mode':<20}{'final':<14}{'forget':<14}{'immediate':<12}{'zero-shot':<12}{'imm-zs (real CL?)'}")
for mode in ("stateless","replay_stateless","ewc_stateless"):
    v=agg.get(mode)
    if not v or not v["fin"]:
        print(f"{mode:<20}(no runs)"); continue
    fm=st.mean(v["fin"]); im=st.mean(v["imm"]); zs=st.mean([z for z in v["zs"] if z==z])
    gm=st.mean(v["forg"])
    print(f"{mode:<20}{fm:<14.3f}{gm:<14.3f}{im:<12.3f}{zs:<12.3f}{im-zs:+.3f}")
print("\n(If imm-zs stays ~0, the transfer degeneracy survives better features —")
print(" a STRONGER paper claim: the degeneracy is representation-independent.)")
PY

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  DONE (TD) — $(date '+%Y%m%d_%H%M%S')   eval JSONs: $RESULTS/"
echo "║  Compare against raw results in results_dil_recovered/"
echo "╚════════════════════════════════════════════════════════════════╝"
} 2>&1 | tee "$LOG"

echo "Full log: $LOG"
