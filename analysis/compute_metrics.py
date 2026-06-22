"""
Compute BWT and related continual-learning metrics from evaluation JSON files.

Usage:
    python analysis/compute_metrics.py                     # scans results/
    python analysis/compute_metrics.py results/eval_x.json
    python analysis/compute_metrics.py --no-write          # print only
"""
import json
import glob
import os
import csv
import argparse
import numpy as np


def compute_metrics(json_path):
    """Compute CL metrics from a single evaluation JSON.

    Requires 'immediate_performance' and 'final_performance' keys
    (written by hgrn/evaluation.py).
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
    deltas = fin_acc - imm_acc

    if num_tasks > 1:
        bwt = float(np.mean(deltas[:-1]))
        forgetting = float(np.mean(np.maximum(0.0, -deltas[:-1])))
    else:
        bwt = 0.0
        forgetting = 0.0

    # Infer num_classes from per_class_accuracy keys in any task entry.
    # Used as the random-chance baseline: b = 1 / num_classes.
    _sample_perf = imm[tasks[0]].get('per_class_accuracy', {})
    num_classes = len(_sample_perf) if _sample_perf else None
    random_baseline = (1.0 / num_classes) if num_classes else None

    # Forward Transfer: mean zero-shot accuracy on each task before it was trained on.
    # forward_performance[task_i] = accuracy on task i evaluated after task i-1 finished.
    # Exists for tasks 2..N (i.e. N-1 values). None if the run predates this feature.
    fwd = data.get('forward_performance', {})
    if fwd and num_tasks > 1:
        fwd_acc = np.array([fwd[t]['accuracy'] for t in tasks[1:] if t in fwd])
        ft = float(np.mean(fwd_acc)) if len(fwd_acc) > 0 else None
        ft_normalized = float(np.mean(fwd_acc) - random_baseline) if (ft is not None and random_baseline is not None) else None
    else:
        ft = None
        ft_normalized = None

    return {
        'file': os.path.basename(json_path),
        'arch': data.get('metadata', {}).get('arch', 'unknown'),
        'mode': data.get('metadata', {}).get('mode', 'unknown'),
        'scenario': data.get('metadata', {}).get('scenario', 'unknown'),
        'num_tasks': num_tasks,
        'num_classes': num_classes,
        'random_baseline': random_baseline,
        'avg_acc_final': float(np.mean(fin_acc)),
        'avg_acc_immediate': float(np.mean(imm_acc)),
        'bwt': bwt,
        'bwt_all_tasks': float(np.mean(deltas)),
        'forgetting': forgetting,
        'ft': ft,
        'ft_normalized': ft_normalized,
        'tasks': tasks,
        'per_task_delta': deltas.tolist(),
        'per_task_immediate': imm_acc.tolist(),
        'per_task_final': fin_acc.tolist(),
    }


def _fmt_pct(v):
    return f"{v*100:>6.1f}%" if v is not None else f"{'N/A':>7}"


def print_summary(results):
    header = (f"{'arch':<6} {'mode':<22} {'scenario':<6} {'T':>3} {'ACC':>7} {'LA':>7} "
              f"{'BWT':>8} {'BWT(all)':>9} {'Forget':>7} {'FT':>7} {'FT(norm)':>9}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['arch']:<6} {r['mode']:<22} {r['scenario']:<6} {r['num_tasks']:>3} "
              f"{r['avg_acc_final']*100:>6.1f}% {r['avg_acc_immediate']*100:>6.1f}% "
              f"{r['bwt']*100:>7.2f}% {r['bwt_all_tasks']*100:>8.2f}% "
              f"{r['forgetting']*100:>6.2f}% {_fmt_pct(r['ft'])} {_fmt_pct(r['ft_normalized'])}")
    print()
    print("ACC      = avg final accuracy")
    print("LA       = avg immediate (learning) accuracy")
    print("BWT      = avg(final - immediate) over first T-1 tasks  (negative = forgetting)")
    print("Forget   = avg accuracy lost per task, clamped at 0")
    print("FT       = avg zero-shot accuracy on each task before training on it")
    print("FT(norm) = FT - random baseline (1/num_classes)  (N/A for runs without forward_performance)")


def write_summary_csv(results, out_path):
    fields = ['file', 'arch', 'mode', 'scenario', 'num_tasks', 'num_classes', 'random_baseline',
              'avg_acc_final', 'avg_acc_immediate', 'bwt', 'bwt_all_tasks',
              'forgetting', 'ft', 'ft_normalized']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})


def write_per_task_csv(results, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'scenario', 'task', 'immediate_acc', 'final_acc', 'delta'])
        for r in results:
            for task, imm, fin, d in zip(r['tasks'], r['per_task_immediate'],
                                         r['per_task_final'], r['per_task_delta']):
                w.writerow([r['mode'], r['scenario'], task, imm, fin, d])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='*', default=['results'],
                        help="JSON files and/or directories to scan (default: results/)")
    parser.add_argument('--out-dir', default='results',
                        help="Where to write summary CSV/JSON (default: results/)")
    parser.add_argument('--no-write', action='store_true',
                        help="Print only, do not write summary files")
    args = parser.parse_args()

    json_files = []
    for p in args.paths:
        if os.path.isdir(p):
            json_files.extend(sorted(glob.glob(os.path.join(p, '*.json'))))
        else:
            json_files.append(p)

    results, skipped = [], []
    for jf in json_files:
        m = compute_metrics(jf)
        if m is None:
            skipped.append(os.path.basename(jf))
        else:
            results.append(m)

    if not results:
        print("No compatible evaluation files found.")
        if skipped:
            print("Skipped (missing immediate/final performance keys):")
            for s in skipped:
                print(f"  - {s}")
        return

    results.sort(key=lambda r: r['avg_acc_final'], reverse=True)
    print_summary(results)

    if skipped:
        print("\nSkipped:")
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
        print(f"\nWrote {summary_csv}")
        print(f"Wrote {per_task_csv}")
        print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
