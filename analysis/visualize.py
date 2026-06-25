"""
Visualization tools for HGRN continual learning results.

Subcommands:
    accuracy  -- bar chart of immediate vs final accuracy per task
    bwt       -- per-task backward transfer bar chart

Usage:
    python analysis/visualize.py accuracy results/eval_xxx.json
    python analysis/visualize.py bwt      results/eval_xxx.json
"""
import json
import os
import argparse
import numpy as np
import matplotlib
# Non-interactive backend: figures are written to results/plots/ via savefig and
# no GUI window pops up (so this is safe to run headless / over SSH). Honour an
# externally-set MPLBACKEND if the user wants an interactive backend instead.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

_PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "plots")


def _load(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def _plots_dir():
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    return _PLOTS_DIR


def _output_path(json_path, suffix):
    stem = os.path.splitext(os.path.basename(json_path))[0]
    return os.path.join(_plots_dir(), f"{stem}{suffix}.png")


def plot_accuracy(json_path):
    """Bar chart comparing immediate vs final accuracy per task."""
    data = _load(json_path)
    imm_data = data['immediate_performance']
    fin_data = data['final_performance']

    rows = []
    for subject in sorted(imm_data.keys(), key=lambda x: int(x.split('_')[-1])):
        rows.append({"Task": subject, "Accuracy": imm_data[subject]['accuracy'], "Type": "Immediate"})
        rows.append({"Task": subject, "Accuracy": fin_data[subject]['accuracy'], "Type": "Final (After CL)"})

    df = pd.DataFrame(rows)
    mode = data['metadata']['mode']

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="Task", y="Accuracy", hue="Type", palette="viridis")
    plt.title(f"Immediate vs. Final Accuracy — {mode}", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel("Classification Accuracy")
    plt.tight_layout()

    out = _output_path(json_path, "_accuracy")
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.close()


def plot_bwt(json_path):
    """Per-task backward transfer bar chart (red = forgetting, green = positive transfer)."""
    data = _load(json_path)
    imm_data = data['immediate_performance']
    fin_data = data['final_performance']
    mode = data['metadata']['mode']

    tasks = list(imm_data.keys())
    bwt_values = [
        (fin_data[t]['accuracy'] - imm_data[t]['accuracy']) * 100
        for t in tasks
    ]
    task_labels = [
        t.replace('subject_', 'S').replace('Subject_', 'S').replace('_Exercise_', ' E')
        for t in tasks
    ]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in bwt_values]
    bars = plt.bar(task_labels, bwt_values, color=colors)
    plt.axhline(0, color='black', linewidth=1.5)
    plt.title(f"Backward Transfer (BWT) per Task — {mode.upper()}", fontsize=16, pad=20)
    plt.ylabel("Accuracy Delta (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        offset = 2 if yval >= 0 else -3
        plt.text(bar.get_x() + bar.get_width() / 2, yval + offset,
                 f"{yval:.1f}%", ha='center', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out = _output_path(json_path, "_bwt")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_acc = sub.add_parser('accuracy', help="Immediate vs final accuracy bar chart")
    p_acc.add_argument('json_path', help="Path to evaluation JSON")

    p_bwt = sub.add_parser('bwt', help="Per-task backward transfer bar chart")
    p_bwt.add_argument('json_path', help="Path to evaluation JSON")

    args = parser.parse_args()

    if args.cmd == 'accuracy':
        plot_accuracy(args.json_path)
    elif args.cmd == 'bwt':
        plot_bwt(args.json_path)


if __name__ == "__main__":
    main()
