import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from datetime import datetime
import random

# --- Required Imports ---
import training_functions
import evaluation_functions # <-- NEW IMPORT
from state_space.fastHGRN import HGRNModel 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocessing_scripts import ss_preprocessing
# ------------------------

class SimpleMemoryBuffer:
    """Reservoir sampling memory buffer for Experience Replay."""
    def __init__(self, capacity=1000):
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
                        choices=['stateless', 'stateful', 'replay_stateless', 'replay_stateful', 'ewc_stateful'],
                        help="Choose the Continual Learning paradigm to execute.")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--exercise', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_per_task', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    is_stateless = args.mode in ['stateless', 'replay_stateless']
    
    print(f"Initializing {args.mode.upper()} training pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build the task streams
    task_streams = ss_preprocessing.build_ss_task_streams(
        exercise_number=args.exercise, 
        path=args.data_path, 
        shuffle=is_stateless, 
        batch_size=args.batch_size
    )

    model = HGRNModel(in_channels=10, d_model=128, num_classes=17, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    save_dir = setup_save_directory(args.mode, args.exercise)

    memory_buffer = SimpleMemoryBuffer(capacity=1000) if 'replay' in args.mode else None
    fisher_dict = None
    optpar_dict = None

    # ==========================================
    # THE MACRO TASK LOOP
    # ==========================================
    for task_idx, task in enumerate(task_streams):
        task_id = task['task_id']
        train_loader = task['train']

        print(f"\n=== Starting Task: {task_id} ===")
        
        for epoch in range(args.epochs_per_task):
            match args.mode:
                case 'stateless':
                    epoch_loss, epoch_acc = training_functions.train_stateless(
                        model, train_loader, optimizer, criterion, device)
                
                case 'stateful':
                    epoch_loss, epoch_acc = training_functions.train_stateful(
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
            
            print(f"Epoch {epoch+1}/{args.epochs_per_task} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        # Post-Task Consolidation
        match args.mode:
            case 'replay_stateless' | 'replay_stateful':
                print("Populating Memory Buffer with current subject data...")
                for sequences, labels, masks in train_loader:
                    memory_buffer.add_data(sequences, labels, masks)
                print(f"Buffer size is now: {len(memory_buffer)}")

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

        weight_filepath = os.path.join(save_dir, f"hgrn_{args.mode}_{task_id}.pt")
        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weight_filepath)

    print("\nTraining sequence complete.")

    # ==========================================
    # FINAL CONTINUAL LEARNING EVALUATION
    # ==========================================
    print("\n==========================================")
    print("FINAL CONTINUAL LEARNING EVALUATION")
    print("==========================================")
    
    # 1. Setup the results dictionary with configuration metadata
    eval_results = {
        "metadata": {
            "mode": args.mode,
            "exercise": args.exercise,
            "batch_size": args.batch_size,
            "epochs_per_task": args.epochs_per_task,
            "learning_rate": args.lr
        },
        "task_metrics": {}
    }
    
    # 2. Test the FINAL model against ALL subjects to measure forgetting
    for task in task_streams:
        task_id = task['task_id']
        test_loader = task['test']
        
        eval_loss, eval_acc = evaluation_functions.evaluate(model, test_loader, criterion, device)
        
        eval_results["task_metrics"][task_id] = {
            "loss": eval_loss,
            "accuracy": eval_acc
        }
        print(f"Evaluated on {task_id} | Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f}")
        
    # 3. Save to disk
    saved_path = evaluation_functions.save_evaluation_results(eval_results, args.mode, args.exercise)
    print(f"\nEvaluation metrics saved to: {saved_path}")

if __name__ == "__main__":
    main()