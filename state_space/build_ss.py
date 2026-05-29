import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import json
from datetime import datetime
import random

# --- Required Imports ---
import training_functions
import evaluation_functions
from fastHGRN import HGRNModel 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocessing_scripts import ss_preprocessing
# ------------------------

import torch
import random
import numpy as np

class HerdingBuffer:
    def __init__(self, capacity_per_class=20):
        self.capacity_per_class = capacity_per_class
        self.buffer = {} # {class_id: [(seq, label, mask), ...]}
    
    def __len__(self):
        """Returns the total number of exemplars stored across all classes."""
        return sum(len(exemplars) for exemplars in self.buffer.values())

    def select_exemplars(self, model, loader, device, num_classes=17):
        model.eval()
        # We need to store the raw data to retrieve it later for training
        all_data = {cls: [] for cls in range(num_classes)}
        all_feats = {cls: [] for cls in range(num_classes)}
        
        # 1. Collect all raw data and features
        with torch.no_grad():
            for x, y, m in loader:
                feats = model.get_features(x.to(device)).cpu()
                for i in range(x.size(0)):
                    cls = y[i].item()
                    all_data[cls].append((x[i].cpu(), y[i].cpu(), m[i].cpu()))
                    all_feats[cls].append(feats[i])
        
        # 2. Greedy Herding per class
        for cls in range(num_classes):
            if not all_feats[cls]: continue
            
            feats = torch.stack(all_feats[cls])
            centroid = feats.mean(dim=0)
            
            selected_indices = []
            curr_sum = torch.zeros_like(centroid)
            
            # Select exemplars
            for i in range(min(self.capacity_per_class, len(feats))):
                dists = torch.norm((curr_sum + feats) / (i + 1) - centroid, dim=1)
                # Ensure we don't pick the same index twice
                dists[selected_indices] = float('inf')
                best_idx = torch.argmin(dists).item()
                selected_indices.append(best_idx)
                curr_sum += feats[best_idx]
            
            # Store the raw data triplets for the training loop
            self.buffer[cls] = [all_data[cls][idx] for idx in selected_indices]

    def sample(self, batch_size):
        all_samples = []
        for cls in self.buffer:
            all_samples.extend(self.buffer[cls])
            
        if not all_samples: return None, None, None

        chosen = random.sample(all_samples, min(batch_size, len(all_samples)))
        
        # Unpack the triplets
        seqs = torch.stack([x[0] for x in chosen])
        labels = torch.stack([x[1] for x in chosen])
        masks = torch.stack([x[2] for x in chosen])
        
        return seqs, labels, masks

class SimpleMemoryBuffer:
    """Reservoir sampling memory buffer for Experience Replay."""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def add_data(self, sequences, labels, masks):
        for i in range(sequences.size(0)):
            if len(self.buffer) < self.capacity:
                self.buffer.append((sequences[i].cpu(), labels[i].cpu(), masks[i].cpu()))
            else:
                idx = random.randint(0, self.capacity)
                if idx < self.capacity:
                    self.buffer[idx] = (sequences[i].cpu(), labels[i].cpu(), masks[i].cpu())

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        seqs = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        masks = torch.stack([s[2] for s in samples])
        return seqs, labels, masks
        
    def __len__(self):
        return len(self.buffer)

def setup_save_directory(mode, exercise):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_name = f"weights_{mode}_ex{exercise}_{timestamp}"
    save_path = os.path.join(os.getcwd(), dir_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Train the HGRN Model on NinaPro Data")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['stateless', 'stateful', 'replay_stateless', 'replay_stateful', 'ewc_stateful', 'herding_stateful'],
                        help="Choose the Continual Learning paradigm to execute.")
    
    # --- NEW SCENARIO FLAGS ---
    parser.add_argument('--scenario', type=str, required=True, choices=['dil', 'cil'],
                        help="'dil': Train across subjects. 'cil': Train across exercises for one subject.")
    parser.add_argument('--subject', type=int, default=1, 
                        help="Subject ID to use (Only required if scenario='cil')")
    # --------------------------
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--exercise', type=int, default=1, help="Only used if scenario='dil'")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_per_task', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    is_stateless = args.mode in ['stateless', 'replay_stateless']
    
    print(f"Initializing {args.mode.upper()} training pipeline in {args.scenario.upper()} scenario...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DYNAMIC DATA ROUTING ---
    if args.scenario == 'dil':
        print("Building Domain-Incremental stream (Subject-to-Subject)...")
        task_streams = ss_preprocessing.build_ss_task_streams(
            exercise_number=args.exercise, 
            path=args.data_path, 
            shuffle=is_stateless, 
            batch_size=args.batch_size
        )
        total_classes = 17 # Hardcoded for NinaPro Ex 1 (Adjust if necessary)
        
    elif args.scenario == 'cil':
        print(f"Building Class-Incremental stream for Subject {args.subject} (Exercise-to-Exercise)...")
        task_streams, total_classes = ss_preprocessing.build_cil_multi_exercise_stream(
            subject_id=args.subject,
            path=args.data_path,
            batch_size=args.batch_size,
            shuffle=is_stateless
        )
        print(f"Detected {total_classes} total distinct gestures across exercises.")

    # Initialize Model dynamically based on total_classes
    model = HGRNModel(in_channels=10, d_model=128, num_classes=total_classes, num_layers=4).to(device)
    
    # Reset optimizer state between tasks to prevent momentum poisoning
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    save_dir = setup_save_directory(f"{args.scenario}_{args.mode}", args.exercise)

    # Continual Learning Global Variables
    memory_buffer = SimpleMemoryBuffer(capacity=10000) if 'replay' in args.mode else None
    herding_buffer = HerdingBuffer(capacity_per_class=20) if args.mode == 'herding_stateful' else None 
    fisher_dict = None
    optpar_dict = None
    
    
    # Initialize the Evaluation Results JSON structure
    eval_results = {
        "metadata": {
            "mode": args.mode,
            "exercise": args.exercise,
            "batch_size": args.batch_size,
            "epochs_per_task": args.epochs_per_task,
            "learning_rate": args.lr
        },
        "training_history": {},
        "immediate_performance": {},
        "final_performance": {}
    }

    # ==========================================
    # THE MACRO TASK LOOP
    # ==========================================
    for task_idx, task in enumerate(task_streams):
        task_id = task['task_id']
        train_loader = task['train']
        test_loader = task['test'] 
        
        if task_idx > 0 and args.mode == 'herding_stateful':
            print("Freezing early layers to prevent Feature Drift...")
            # Lock the input projection and the first HGRN layer
            for param in model.input_proj.parameters():
                param.requires_grad = False
            for param in model.layers[0].parameters():
                param.requires_grad = False

        print(f"\n=== Starting Task: {task_id} ===")
        
        task_epoch_losses = []
        task_epoch_accs = []
        
        # --- 1. The Micro Epoch Loop ---
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
                        memory_buffer=memory_buffer, replay_batch_size=16)
                
                case 'replay_stateful':
                    epoch_loss, epoch_acc = training_functions.train_replay_stateful(
                        model, train_loader, optimizer, criterion, device, 
                        memory_buffer=memory_buffer, replay_batch_size=16)
                
                case 'ewc_stateful':
                    epoch_loss, epoch_acc = training_functions.train_ewc_stateful(
                        model, train_loader, optimizer, criterion, device, 
                        fisher_dict=fisher_dict, optpar_dict=optpar_dict, ewc_lambda=2000)
                
                case 'herding_stateful':
                    epoch_loss, epoch_acc = training_functions.train_replay_stateful(
                        model, train_loader, optimizer, criterion, device, 
                        memory_buffer=herding_buffer, replay_batch_size=16)
            
            # Save the training trajectory
            task_epoch_losses.append(epoch_loss)
            task_epoch_accs.append(epoch_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs_per_task} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        # Write the trajectories to the JSON dictionary
        eval_results["training_history"][task_id] = {
            "epoch_losses": task_epoch_losses,
            "epoch_accuracies": task_epoch_accs
        }

        # --- 2. Post-Task Consolidation ---
        match args.mode:
            case 'replay_stateless' | 'replay_stateful':
                print("Populating Memory Buffer with current subject data...")
                for sequences, labels, masks in train_loader:
                    memory_buffer.add_data(sequences, labels, masks)
                print(f"Buffer size is now: {len(memory_buffer)}")

            case 'herding_stateful':
                print("Running Herding to select prototypical exemplars...")
                herding_buffer.select_exemplars(model, train_loader, device, num_classes=total_classes)

            case 'ewc_stateful':
                print("Calculating Fisher Information Matrix for consolidation...")
                curr_fisher, curr_optpar = training_functions.compute_fisher(model, train_loader, device)
                
                if fisher_dict is None:
                    fisher_dict = curr_fisher
                    optpar_dict = curr_optpar
                else:
                    for name in fisher_dict.keys():
                        fisher_dict[name] += curr_fisher[name]
                        optpar_dict[name] = curr_optpar[name]

        # --- 3. Immediate Evaluation (Backward Transfer Baseline) ---
        print("Running Immediate Evaluation...")
        imm_loss, imm_acc, imm_per_class = evaluation_functions.evaluate(model, test_loader, criterion, device)
        eval_results["immediate_performance"][task_id] = {
            "loss": imm_loss,
            "accuracy": imm_acc,
            "per_class_accuracy": imm_per_class
        }
        print(f"Immediate Eval on {task_id} -> Acc: {imm_acc:.4f}")

        # --- 4. Save Checkpoint ---
        weight_filepath = os.path.join(save_dir, f"hgrn_{args.mode}_{task_id}.pt")
        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weight_filepath)
        print(f"Saved checkpoint to {weight_filepath}")

    print("\nTraining sequence complete.")

    # ==========================================
    # FINAL CONTINUAL LEARNING EVALUATION
    # ==========================================
    print("\n==========================================")
    print("FINAL CONTINUAL LEARNING EVALUATION")
    print("==========================================")
    
    for task in task_streams:
        task_id = task['task_id']
        test_loader = task['test']
        
        fin_loss, fin_acc, fin_per_class = evaluation_functions.evaluate(model, test_loader, criterion, device)
        
        eval_results["final_performance"][task_id] = {
            "loss": fin_loss,
            "accuracy": fin_acc,
            "per_class_accuracy": fin_per_class
        }
        print(f"Evaluated on {task_id} | Loss: {fin_loss:.4f} | Accuracy: {fin_acc:.4f}")
        
    # Save the detailed JSON
    saved_path = evaluation_functions.save_evaluation_results(eval_results, args.mode, args.exercise)
    print(f"\nEvaluation metrics saved to: {saved_path}")

if __name__ == "__main__":
    main()