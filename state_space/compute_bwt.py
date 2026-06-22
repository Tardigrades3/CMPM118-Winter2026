import json
import glob
import os
import csv
import argparse
import numpy as np


def compute_metrics(json_path):
    """Compute continual-learning metrics from a single evaluation JSON.

    Expects the schema written by save_evaluation_results, containing
    'immediate_performance' (accuracy on each task right after it was learned,
    i.e. the diagonal R_{i,i} of the transfer matrix) and 'final_performance'
    (accuracy on each task after training on the last task, i.e. the bottom
    row R_{T,i}). These two are exactly what the scalar BWT / forgetting
    metrics require, so no model needs to be re-run.

    Returns a dict of metrics, or None if the file uses an older schema.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'immediate_performance' not in data or 'final_performance' not in data:
        return None

    imm = data['immediate_performance']
    fin = data['final_performance']
    tasks = list(imm.keys())
    num_tasks = len(tasks)

    imm_acc = np.array([imm[t]['accuracy'] for t in tasks])
    fin_acc = np.array([fin[t]['accuracy'] for t in tasks])
    deltas = fin_acc - imm_acc  # per-task backward transfer

    # Standard definitions (Lopez-Paz & Ranzato, 2017) average over the first
    # T-1 tasks: the final task cannot have been forgotten since immediate ==
    # final for it. With T==1 these reduce to 0 / the single task.
    if num_tasks > 1:
        bwt = float(np.mean(deltas[:-1]))
        # Forgetting: how much was lost from each task's best-seen (= immediate,
        # the only earlier checkpoint we have) to the end. Clamped at 0.
        forgetting = float(np.mean(np.maximum(0.0, -deltas[:-1])))
    else:
        bwt = 0.0
        forgetting = 0.0

    return {
        'file': os.path.basename(json_path),
        'mode': data.get('metadata', {}).get('mode', 'unknown'),
        'num_tasks': num_tasks,
        'avg_acc_final': float(np.mean(fin_acc)),       # ACC: final mean accuracy
        'avg_acc_immediate': float(np.mean(imm_acc)),   # LA: learning accuracy
        'bwt': bwt,                                      # >0 helped old tasks, <0 forgetting
        'bwt_all_tasks': float(np.mean(deltas)),         # same but incl. last task
        'forgetting': forgetting,
        'tasks': tasks,
        'per_task_delta': deltas.tolist(),
        'per_task_immediate': imm_acc.tolist(),
        'per_task_final': fin_acc.tolist(),
    }


def print_summary(results):
    header = (f"{'mode':<18} {'T':>3} {'ACC':>7} {'LA':>7} "
              f"{'BWT':>8} {'BWT(all)':>9} {'Forget':>7}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['mode']:<18} {r['num_tasks']:>3} "
              f"{r['avg_acc_final']*100:>6.1f}% {r['avg_acc_immediate']*100:>6.1f}% "
              f"{r['bwt']*100:>7.2f}% {r['bwt_all_tasks']*100:>8.2f}% "
              f"{r['forgetting']*100:>6.2f}%")
    print()
    print("ACC = avg final accuracy | LA = avg immediate (learning) accuracy")
    print("BWT = avg(final - immediate) over first T-1 tasks (negative = forgetting)")
    print("Forget = avg accuracy lost per task, clamped at 0")


def write_summary_csv(results, out_path):
    fields = ['file', 'mode', 'num_tasks', 'avg_acc_final', 'avg_acc_immediate',
              'bwt', 'bwt_all_tasks', 'forgetting']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})


def write_per_task_csv(results, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'task', 'immediate_acc', 'final_acc', 'delta'])
        for r in results:
            for task, imm, fin, d in zip(r['tasks'], r['per_task_immediate'],
                                         r['per_task_final'], r['per_task_delta']):
                w.writerow([r['mode'], task, imm, fin, d])


def main():
    parser = argparse.ArgumentParser(
        description="Compute BWT and related continual-learning metrics "
                    "from evaluation JSON files (no model rerun needed).")
    parser.add_argument('paths', nargs='*', default=['evaluations'],
                        help="JSON files and/or directories to scan "
                             "(default: ./evaluations)")
    parser.add_argument('--out-dir', default='evaluations',
                        help="Where to write summary CSV/JSON (default: ./evaluations)")
    parser.add_argument('--no-write', action='store_true',
                        help="Print only, do not write summary files")
    args = parser.parse_args()

    # Expand directories into their .json files.
    json_files = []
    for p in args.paths:
        if os.path.isdir(p):
            json_files.extend(sorted(glob.glob(os.path.join(p, '*.json'))))
        else:
            json_files.append(p)

    results = []
    skipped = []
    for jf in json_files:
        m = compute_metrics(jf)
        if m is None:
            skipped.append(os.path.basename(jf))
        else:
            results.append(m)

    if not results:
        print("No compatible evaluation files found.")
        if skipped:
            print("Skipped (older schema, no immediate/final perf):")
            for s in skipped:
                print(f"  - {s}")
        return

    results.sort(key=lambda r: r['avg_acc_final'], reverse=True)
    print_summary(results)

    if skipped:
        print()
        print("Skipped (older schema, no immediate/final performance):")
        for s in skipped:
            print(f"  - {s}")

    if not args.no_write:
        os.makedirs(args.out_dir, exist_ok=True)
        summary_csv = os.path.join(args.out_dir, 'bwt_summary.csv')
        per_task_csv = os.path.join(args.out_dir, 'bwt_per_task.csv')
        summary_json = os.path.join(args.out_dir, 'bwt_summary.json')
        write_summary_csv(results, summary_csv)
        write_per_task_csv(results, per_task_csv)
        with open(summary_json, 'w') as f:
            json.dump(results, f, indent=2)
        print()
        print(f"Wrote {summary_csv}")
        print(f"Wrote {per_task_csv}")
        print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
