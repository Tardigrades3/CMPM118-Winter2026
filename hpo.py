#!/usr/bin/env python3
"""
Hyperparameter optimisation using Optuna — two-phase approach.

Phase 1 (arch):  Tune model architecture on a DIL stateless proxy.
                 Architecture HPs transfer to both DIL and CIL.

Phase 2 (cl):    Tune CL-method-specific HPs on a CIL proxy.
                 Architecture is fixed (from phase 1 best params or explicit flags).

Usage
-----
# Phase 1 — architecture (stateless DIL, representative subjects)
python hpo.py arch \\
    --subjects 27 23 4 3 24 17 \\
    --exercise 1 \\
    --data_path ./NinaProData \\
    --arch hgrn \\
    --n_trials 50

# Phase 2 — EWC method (CIL, tighter subject set)
python hpo.py cl \\
    --mode ewc_stateful \\
    --subjects 27 4 17 \\
    --data_path ./NinaProData \\
    --arch_params results/hpo_arch_hgrn_best.json \\
    --n_trials 30

# Phase 2 — Replay (pass arch params manually if no JSON yet)
python hpo.py cl \\
    --mode replay_stateful \\
    --subjects 27 4 17 \\
    --data_path ./NinaProData \\
    --d_model 128 --num_layers 4 --lr 1e-4 --weight_decay 0.01 \\
    --n_trials 30

Resume any study by re-running the same command — Optuna picks up where it left off.
Inspect results with:  optuna-dashboard sqlite:///results/<study_name>.db
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

from hgrn.model import HGRNModel
from models.lstm import LSTMModel
from hgrn import training as training_functions
from hgrn import evaluation as evaluation_functions
from hgrn.buffers import HerdingBuffer, SimpleMemoryBuffer
from preprocessing import signal
from preprocessing.streams import padding, build_cil_multi_exercise_stream

_ARCH_REGISTRY = {'hgrn': HGRNModel, 'lstm': LSTMModel}


# ── helpers ───────────────────────────────────────────────────────────────────

def _subject_loaders(subject_id, exercise, data_path, batch_size, shuffle):
    """Train/test loaders + num_classes for a single subject × exercise."""
    path = f"{data_path}/s{subject_id}/S{subject_id}_A1_E{exercise}.mat"
    data = signal.load_data(path)
    x_train, y_train, x_test, y_test, _ = signal.preprocessing_internals(data)
    num_classes = int(max(np.max(y_train), np.max(y_test))) + 1
    train_loader = DataLoader(signal.NinaProDataset(x_train, y_train),
                              batch_size=batch_size, shuffle=shuffle, collate_fn=padding)
    test_loader  = DataLoader(signal.NinaProDataset(x_test, y_test),
                              batch_size=batch_size, shuffle=False, collate_fn=padding)
    return train_loader, test_loader, num_classes


# ── Phase 1: architecture HPO (DIL stateless proxy) ──────────────────────────

def _arch_objective(trial, args, device):
    d_model      = trial.suggest_categorical('d_model',    [64, 128, 256])
    num_layers   = trial.suggest_categorical('num_layers', [2, 3, 4])
    lr           = trial.suggest_float('lr',           5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.1,  log=True)
    batch_size   = trial.suggest_categorical('batch_size', [16, 32, 64])

    criterion  = nn.CrossEntropyLoss()
    ModelClass = _ARCH_REGISTRY[args.arch]

    # num_classes is fixed per exercise — infer from first subject
    _, _, num_classes = _subject_loaders(args.subjects[0], args.exercise,
                                         args.data_path, batch_size, True)

    # Single shared model trained sequentially (matches actual DIL setup)
    model     = ModelClass(in_channels=10, d_model=d_model,
                           num_classes=num_classes, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    imm_accs = []
    try:
        for step, subj_id in enumerate(args.subjects):
            train_loader, test_loader, _ = _subject_loaders(
                subj_id, args.exercise, args.data_path, batch_size, True)

            for _ in range(args.epochs_per_task):
                training_functions.train_naive_stateless(
                    model, train_loader, optimizer, criterion, device)

            _, acc, _ = evaluation_functions.evaluate(
                model, test_loader, criterion, device)
            imm_accs.append(acc)

            trial.report(float(np.mean(imm_accs)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    finally:
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return float(np.mean(imm_accs))


# ── Phase 2: CL method HPO (CIL proxy) ───────────────────────────────────────

_CL_SEARCH_SPACES = {
    'ewc_stateful': ['ewc_lambda'],
    'replay_stateful':  ['replay_weight', 'capacity', 'noise_std', 'replay_batch_size'],
    'replay_stateless': ['replay_weight', 'capacity', 'noise_std', 'replay_batch_size'],
    'herding_stateful': ['capacity_per_class', 'replay_batch_size'],
}


def _suggest_cl_params(trial, mode):
    params = {}
    space  = _CL_SEARCH_SPACES.get(mode, [])

    if 'ewc_lambda' in space:
        params['ewc_lambda'] = trial.suggest_float('ewc_lambda', 100.0, 20000.0, log=True)

    if 'replay_weight' in space:
        params['replay_weight'] = trial.suggest_float('replay_weight', 0.1, 0.9)
    if 'capacity' in space:
        params['capacity'] = trial.suggest_categorical('capacity', [2000, 5000, 10000, 20000])
    if 'noise_std' in space:
        params['noise_std'] = trial.suggest_float('noise_std', 0.001, 0.1, log=True)
    if 'replay_batch_size' in space:
        params['replay_batch_size'] = trial.suggest_categorical('replay_batch_size', [8, 16, 32])

    if 'capacity_per_class' in space:
        params['capacity_per_class'] = trial.suggest_categorical(
            'capacity_per_class', [5, 10, 20, 50])

    return params


def _run_one_cil_subject(model, optimizer, criterion, device,
                         task_streams, mode, cl_params, epochs_per_task):
    """Full CIL pass (all exercises) for one subject.

    Returns (imm_accs, final_accs) as numpy arrays of length num_tasks.
    """
    memory_buffer  = (SimpleMemoryBuffer(capacity=cl_params.get('capacity', 10000))
                      if 'replay' in mode else None)
    herding_buffer = (HerdingBuffer(capacity_per_class=cl_params.get('capacity_per_class', 20))
                      if mode == 'herding_stateful' else None)
    fisher_dict = None
    optpar_dict = None

    imm_accs = []

    for task_idx, task in enumerate(task_streams):
        train_loader = task['train']
        test_loader  = task['test']

        # Herding freezes feature extractor after first task
        if task_idx > 0 and mode == 'herding_stateful':
            for param in model.input_proj.parameters():
                param.requires_grad = False
            for param in model.layers[0].parameters():
                param.requires_grad = False

        for _ in range(epochs_per_task):
            if mode == 'stateful':
                training_functions.train_naive_stateful(
                    model, train_loader, optimizer, criterion, device)

            elif mode in ('replay_stateful', 'replay_stateless'):
                training_functions.train_replay_stateful(
                    model, train_loader, optimizer, criterion, device,
                    memory_buffer=memory_buffer,
                    replay_batch_size=cl_params.get('replay_batch_size', 16),
                    replay_weight=cl_params.get('replay_weight', 0.5),
                    noise_std=cl_params.get('noise_std', 0.01))

            elif mode == 'ewc_stateful':
                training_functions.train_ewc_stateful(
                    model, train_loader, optimizer, criterion, device,
                    fisher_dict=fisher_dict, optpar_dict=optpar_dict,
                    ewc_lambda=cl_params.get('ewc_lambda', 2000))

            elif mode == 'herding_stateful':
                training_functions.train_replay_stateful(
                    model, train_loader, optimizer, criterion, device,
                    memory_buffer=herding_buffer,
                    replay_batch_size=cl_params.get('replay_batch_size', 16),
                    noise_std=cl_params.get('noise_std', 0.01))

        # Post-task consolidation
        if mode in ('replay_stateful', 'replay_stateless') and memory_buffer is not None:
            for seqs, lbls, masks in train_loader:
                memory_buffer.add_data(seqs, lbls, masks)

        elif mode == 'herding_stateful' and herding_buffer is not None:
            num_classes = model.head.out_features
            herding_buffer.select_exemplars(model, train_loader, device,
                                            num_classes=num_classes)

        elif mode == 'ewc_stateful':
            curr_fisher, curr_optpar = training_functions.compute_fisher(
                model, train_loader, device)
            if fisher_dict is None:
                fisher_dict, optpar_dict = curr_fisher, curr_optpar
            else:
                for name in fisher_dict:
                    fisher_dict[name] += curr_fisher[name]
                    optpar_dict[name]  = curr_optpar[name]

        _, acc, _ = evaluation_functions.evaluate(model, test_loader, criterion, device)
        imm_accs.append(acc)

    # Final evaluation across all tasks
    final_accs = []
    for task in task_streams:
        _, acc, _ = evaluation_functions.evaluate(model, task['test'], criterion, device)
        final_accs.append(acc)

    return np.array(imm_accs), np.array(final_accs)


def _cl_objective(trial, args, arch_params, device):
    cl_params  = _suggest_cl_params(trial, args.mode)
    d_model    = arch_params['d_model']
    num_layers = arch_params['num_layers']
    lr         = arch_params['lr']
    weight_decay = arch_params['weight_decay']
    batch_size = arch_params.get('batch_size', 32)

    ModelClass = _ARCH_REGISTRY[args.arch]
    criterion  = nn.CrossEntropyLoss()
    shuffle    = 'stateless' in args.mode

    subject_scores = []
    for step, subj_id in enumerate(args.subjects):
        task_streams, total_classes = build_cil_multi_exercise_stream(
            subject_id=subj_id,
            path=args.data_path,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        model     = ModelClass(in_channels=10, d_model=d_model,
                               num_classes=total_classes, num_layers=num_layers).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        try:
            imm_accs, final_accs = _run_one_cil_subject(
                model, optimizer, criterion, device,
                task_streams, args.mode, cl_params, args.epochs_per_task)
        finally:
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_final  = float(np.mean(final_accs))
        deltas     = final_accs - imm_accs
        forgetting = float(np.mean(np.maximum(0.0, -deltas[:-1]))) if len(deltas) > 1 else 0.0
        # Objective: maximise final accuracy, penalise forgetting
        score = avg_final - 0.5 * forgetting
        subject_scores.append(score)

        trial.report(float(np.mean(subject_scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(subject_scores))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='phase', required=True)

    def _common(p):
        p.add_argument('--subjects',       type=int, nargs='+', required=True,
                       help="Subject IDs to use as the HPO proxy")
        p.add_argument('--data_path',      required=True)
        p.add_argument('--arch',           default='hgrn', choices=list(_ARCH_REGISTRY))
        p.add_argument('--epochs_per_task', type=int, default=5)
        p.add_argument('--n_trials',       type=int, default=50)
        p.add_argument('--study_name',     default=None,
                       help="Optuna study name (auto-generated if omitted)")
        p.add_argument('--out_dir',        default='results',
                       help="Where to save the best-params JSON and SQLite DB")

    # ── arch sub-command ──────────────────────────────────────────────────────
    p_arch = sub.add_parser('arch', help="Phase 1: architecture HPO")
    _common(p_arch)
    p_arch.add_argument('--exercise', type=int, default=1,
                        help="Which exercise to use as the DIL proxy (default: 1)")

    # ── cl sub-command ────────────────────────────────────────────────────────
    p_cl = sub.add_parser('cl', help="Phase 2: CL method HPO")
    _common(p_cl)
    p_cl.add_argument('--mode', required=True,
                      choices=list(_CL_SEARCH_SPACES),
                      help="CL method to tune")
    # Architecture params — load from phase-1 JSON or provide individually
    p_cl.add_argument('--arch_params',
                      help="Path to phase-1 best-params JSON (e.g. results/hpo_arch_hgrn_best.json)")
    p_cl.add_argument('--d_model',      type=int,   default=128)
    p_cl.add_argument('--num_layers',   type=int,   default=4)
    p_cl.add_argument('--lr',           type=float, default=1e-4)
    p_cl.add_argument('--weight_decay', type=float, default=0.01)
    p_cl.add_argument('--batch_size',   type=int,   default=32)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── study setup ───────────────────────────────────────────────────────────
    if args.phase == 'arch':
        study_name = args.study_name or f"hpo_arch_{args.arch}_ex{args.exercise}"
    else:
        study_name = args.study_name or f"hpo_cl_{args.mode}_{args.arch}"

    storage = f"sqlite:///{args.out_dir}/{study_name}.db"

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    n_existing = len(study.trials)

    # ── run ───────────────────────────────────────────────────────────────────
    if args.phase == 'arch':
        print(f"Phase 1 — Architecture HPO ({args.arch.upper()}, DIL stateless)")
        print(f"  subjects     : {args.subjects}")
        print(f"  exercise     : {args.exercise}")
        print(f"  epochs/task  : {args.epochs_per_task}")
        print(f"  trials       : {args.n_trials}  (+ {n_existing} already done)")
        print(f"  storage      : {storage}")

        study.optimize(
            lambda trial: _arch_objective(trial, args, device),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

    elif args.phase == 'cl':
        # Resolve architecture params
        if args.arch_params and os.path.exists(args.arch_params):
            with open(args.arch_params) as f:
                saved = json.load(f)
            arch_params = saved['params']
            print(f"Loaded arch params from {args.arch_params}:")
        else:
            arch_params = {
                'd_model':      args.d_model,
                'num_layers':   args.num_layers,
                'lr':           args.lr,
                'weight_decay': args.weight_decay,
                'batch_size':   args.batch_size,
            }
            if args.arch_params:
                print(f"Warning: {args.arch_params} not found — using CLI defaults.")
            print("Architecture params (from CLI):")
        for k, v in arch_params.items():
            print(f"  {k}: {v}")

        print(f"\nPhase 2 — CL Method HPO ({args.mode}, CIL proxy)")
        print(f"  subjects    : {args.subjects}")
        print(f"  epochs/task : {args.epochs_per_task}")
        print(f"  trials      : {args.n_trials}  (+ {n_existing} already done)")
        print(f"  storage     : {storage}")

        study.optimize(
            lambda trial: _cl_objective(trial, args, arch_params, device),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

    # ── results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'═' * 55}")
    print(f"Best trial : #{best.number}   objective = {best.value:.4f}")
    print(f"Best params:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"  {k:<24} {v:.6g}")
        else:
            print(f"  {k:<24} {v}")

    out_path = os.path.join(args.out_dir, f"{study_name}_best.json")
    with open(out_path, 'w') as f:
        json.dump({'trial': best.number, 'value': best.value, 'params': best.params}, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(f"Dashboard: optuna-dashboard {storage}")


if __name__ == '__main__':
    main()
