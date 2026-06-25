import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from datetime import datetime

# Ensure the project root is on sys.path when running as a script.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hgrn.model import HGRNModel
from models.lstm import LSTMModel
from hgrn import training as training_functions
from hgrn import evaluation as evaluation_functions
from hgrn.buffers import HerdingBuffer, SimpleMemoryBuffer
from preprocessing import streams as ss_preprocessing

_ARCH_REGISTRY = {
    'hgrn': HGRNModel,
    'lstm': LSTMModel,
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_save_directory(mode, exercise):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_name = f"{mode}_ex{exercise}_{timestamp}"
    save_path = os.path.join(_PROJECT_ROOT, "checkpoints", dir_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train the HGRN Model on NinaPro Data")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['stateless', 'stateful', 'replay_stateless', 'replay_stateful',
                                 'ewc_stateful', 'herding_stateful'],
                        help="Continual learning paradigm.")
    parser.add_argument('--scenario', type=str, required=True, choices=['dil', 'cil'],
                        help="'dil': train across subjects. 'cil': train across exercises for one subject.")
    parser.add_argument('--subject', type=int, default=1,
                        help="Subject ID (only used if scenario='cil').")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--exercise', type=int, default=1,
                        help="Exercise number (only used if scenario='dil').")
    parser.add_argument('--arch', type=str, default='hgrn', choices=list(_ARCH_REGISTRY),
                        help="Model architecture (default: hgrn).")
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_per_task', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # Model architecture
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    # EWC
    parser.add_argument('--ewc_lambda', type=float, default=2000.0)
    # Replay / Herding (shared replay training path)
    parser.add_argument('--replay_capacity', type=int, default=10000)
    parser.add_argument('--replay_batch_size', type=int, default=16)
    parser.add_argument('--replay_weight', type=float, default=0.5)
    parser.add_argument('--noise_std', type=float, default=0.01)
    # Herding
    parser.add_argument('--herding_capacity_per_class', type=int, default=20)
    # Output
    parser.add_argument('--results_dir', type=str, default='results',
                        help="Directory for eval JSONs (use a separate dir to compare branches).")
    # Reproducibility / seeded repeats (for error bars)
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for model init + data shuffling. Omit for nondeterministic.")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    is_stateless = args.mode in ['stateless', 'replay_stateless']

    print(f"Initializing {args.arch.upper()} | {args.mode.upper()} | {args.scenario.upper()} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        # Faster matmuls/convolutions on T4/A10G; benchmark picks optimal cuDNN algos.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.scenario == 'dil':
        print("Building Domain-Incremental stream (subject-to-subject)...")
        task_streams, total_classes = ss_preprocessing.build_ss_task_streams(
            exercise_number=args.exercise,
            path=args.data_path,
            shuffle=is_stateless,
            batch_size=args.batch_size
        )
        print(f"Detected {total_classes} total classes for Exercise {args.exercise}.")

    elif args.scenario == 'cil':
        print(f"Building Class-Incremental stream for Subject {args.subject} (exercise-to-exercise)...")
        task_streams, total_classes = ss_preprocessing.build_cil_multi_exercise_stream(
            subject_id=args.subject,
            path=args.data_path,
            batch_size=args.batch_size,
            shuffle=is_stateless
        )
        print(f"Detected {total_classes} total classes across exercises.")

    ModelClass = _ARCH_REGISTRY[args.arch]
    model = ModelClass(in_channels=10, d_model=args.d_model,
                       num_classes=total_classes, num_layers=args.num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    save_dir = setup_save_directory(f"{args.scenario}_{args.arch}_{args.mode}", args.exercise)

    memory_buffer = SimpleMemoryBuffer(capacity=args.replay_capacity) if 'replay' in args.mode else None
    herding_buffer = HerdingBuffer(capacity_per_class=args.herding_capacity_per_class) if args.mode == 'herding_stateful' else None
    fisher_dict = None
    optpar_dict = None

    eval_results = {
        "metadata": {
            "arch": args.arch,
            "mode": args.mode,
            "scenario": args.scenario,
            "exercise": args.exercise,
            "subject": args.subject,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "epochs_per_task": args.epochs_per_task,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "ewc_lambda": args.ewc_lambda,
            "replay_capacity": args.replay_capacity,
            "replay_batch_size": args.replay_batch_size,
            "replay_weight": args.replay_weight,
            "noise_std": args.noise_std,
            "herding_capacity_per_class": args.herding_capacity_per_class,
            "seed": args.seed,
        },
        "training_history": {},
        "immediate_performance": {},
        "forward_performance": {},
        "final_performance": {}
    }

    for task_idx, task in enumerate(task_streams):
        task_id = task['task_id']
        train_loader = task['train']
        test_loader = task['test']

        # Freeze the feature extractor for herding — but ONLY for CIL. In CIL this
        # preserves stable features while the class set grows. In DIL the whole task
        # is adapting features to each new subject's domain, so freezing after the
        # first subject cripples it; keep the extractor trainable there.
        if task_idx > 0 and args.mode == 'herding_stateful' and args.scenario == 'cil':
            print("Freezing input projection and first HGRN layer to prevent feature drift (CIL only)...")
            for param in model.input_proj.parameters():
                param.requires_grad = False
            for param in model.layers[0].parameters():
                param.requires_grad = False

        print(f"\n=== Starting Task: {task_id} ===")

        task_epoch_losses = []
        task_epoch_accs = []

        for epoch in range(args.epochs_per_task):
            match args.mode:
                case 'stateless':
                    epoch_loss, epoch_acc = training_functions.train_naive_stateless(
                        model, train_loader, optimizer, criterion, device)

                case 'stateful':
                    epoch_loss, epoch_acc = training_functions.train_naive_stateful(
                        model, train_loader, optimizer, criterion, device)

                case 'replay_stateless':
                    epoch_loss, epoch_acc = training_functions.train_replay_stateless(
                        model, train_loader, optimizer, criterion, device,
                        memory_buffer=memory_buffer,
                        replay_batch_size=args.replay_batch_size)

                case 'replay_stateful':
                    epoch_loss, epoch_acc = training_functions.train_replay_stateful(
                        model, train_loader, optimizer, criterion, device,
                        memory_buffer=memory_buffer,
                        replay_batch_size=args.replay_batch_size,
                        replay_weight=args.replay_weight,
                        noise_std=args.noise_std)

                case 'ewc_stateful':
                    epoch_loss, epoch_acc = training_functions.train_ewc_stateful(
                        model, train_loader, optimizer, criterion, device,
                        fisher_dict=fisher_dict, optpar_dict=optpar_dict,
                        ewc_lambda=args.ewc_lambda)

                case 'herding_stateful':
                    epoch_loss, epoch_acc = training_functions.train_replay_stateful(
                        model, train_loader, optimizer, criterion, device,
                        memory_buffer=herding_buffer,
                        replay_batch_size=args.replay_batch_size,
                        noise_std=args.noise_std)

            task_epoch_losses.append(epoch_loss)
            task_epoch_accs.append(epoch_acc)
            print(f"Epoch {epoch+1}/{args.epochs_per_task} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        eval_results["training_history"][task_id] = {
            "epoch_losses": task_epoch_losses,
            "epoch_accuracies": task_epoch_accs
        }

        # Post-task consolidation
        match args.mode:
            case 'replay_stateless' | 'replay_stateful':
                print("Populating memory buffer...")
                for sequences, labels, masks in train_loader:
                    memory_buffer.add_data(sequences, labels, masks)
                print(f"Buffer size: {len(memory_buffer)}")

            case 'herding_stateful':
                print("Running herding to select prototypical exemplars...")
                herding_buffer.select_exemplars(model, train_loader, device, num_classes=total_classes)

            case 'ewc_stateful':
                print("Computing Fisher Information Matrix...")
                curr_fisher, curr_optpar = training_functions.compute_fisher(model, train_loader, device)
                if fisher_dict is None:
                    fisher_dict = curr_fisher
                    optpar_dict = curr_optpar
                else:
                    for name in fisher_dict:
                        fisher_dict[name] += curr_fisher[name]
                        optpar_dict[name] = curr_optpar[name]

        # Immediate evaluation
        imm_loss, imm_acc, imm_per_class = evaluation_functions.evaluate(
            model, test_loader, criterion, device)
        eval_results["immediate_performance"][task_id] = {
            "loss": imm_loss,
            "accuracy": imm_acc,
            "per_class_accuracy": imm_per_class
        }
        print(f"Immediate eval on {task_id} -> Acc: {imm_acc:.4f}")

        # Forward transfer: zero-shot accuracy on the next unseen task
        if task_idx < len(task_streams) - 1:
            next_task = task_streams[task_idx + 1]
            fwd_loss, fwd_acc, fwd_per_class = evaluation_functions.evaluate(
                model, next_task['test'], criterion, device)
            eval_results["forward_performance"][next_task['task_id']] = {
                "loss": fwd_loss,
                "accuracy": fwd_acc,
                "per_class_accuracy": fwd_per_class
            }
            print(f"Forward eval on {next_task['task_id']} (zero-shot) -> Acc: {fwd_acc:.4f}")

        # Checkpoint
        weight_filepath = os.path.join(save_dir, f"hgrn_{args.mode}_{task_id}.pt")
        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weight_filepath)

    print("\nTraining complete. Running final evaluation across all tasks...")

    for task in task_streams:
        task_id = task['task_id']
        fin_loss, fin_acc, fin_per_class = evaluation_functions.evaluate(
            model, task['test'], criterion, device)
        eval_results["final_performance"][task_id] = {
            "loss": fin_loss,
            "accuracy": fin_acc,
            "per_class_accuracy": fin_per_class
        }
        print(f"Final eval on {task_id} | Loss: {fin_loss:.4f} | Acc: {fin_acc:.4f}")

    subject_id = args.subject if args.scenario == 'cil' else None
    saved_path = evaluation_functions.save_evaluation_results(
        eval_results, f"{args.scenario}_{args.mode}", args.exercise,
        subject_id=subject_id, results_dir=args.results_dir, seed=args.seed)
    print(f"\nEvaluation saved to: {saved_path}")


if __name__ == "__main__":
    main()
