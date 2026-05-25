import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from datetime import datetime

# --- Required Imports ---
import training_functions
from fastHGRN import HGRNModel # Assuming HGRNModel is in the same file

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocessing_scripts import ss_preprocessing
# ------------------------

def setup_save_directory(mode, exercise):
    """Creates a timestamped directory for saving weights."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dir_name = f"weights_{mode}_ex{exercise}_{timestamp}"
    save_path = os.path.join(os.getcwd(), dir_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Train the HGRN Model on NinaPro Data")
    parser.add_argument('--mode', type=str, choices=['stateless', 'stateful'], required=True,
                        help="Choose training paradigm: 'stateless' (shuffled batches) or 'stateful' (chronological, passing hidden states).")
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Path to the root NinaPro dataset folder.")
    parser.add_argument('--exercise', type=int, default=1, 
                        help="Exercise number to process (default: 1).")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size for the DataLoader (default: 32).")
    parser.add_argument('--epochs_per_task', type=int, default=5, 
                        help="Number of epochs to train on each subject/task (default: 5).")
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help="Learning rate (default: 1e-4).")
    args = parser.parse_args()

    # Determine shuffling rule based on training mode
    is_stateless = (args.mode == 'stateless')
    print(f"Initializing {args.mode.upper()} training pipeline...")
    print(f"DataLoader shuffling set to: {is_stateless}")

    # Initialize Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build the task streams (list of dictionaries containing loaders per subject)
    task_streams = ss_preprocessing.build_ss_task_streams(
        exercise_number=args.exercise, 
        path=args.data_path, 
        shuffle=is_stateless, 
        batch_size=args.batch_size
    )

    # Initialize Model
    # Assuming NinaPro features = 12 (adjust if using 10 or 16), and 17 gesture classes for Ex 1
    model = HGRNModel(in_channels=10, d_model=128, num_classes=17, num_layers=4)
    model.to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Setup Save Directory
    save_dir = setup_save_directory(args.mode, args.exercise)
    print(f"Model weights will be saved to: {save_dir}\n")

    # --- Continual Learning Loop ---
    for task_idx, task in enumerate(task_streams):
        task_id = task['task_id']
        train_loader = task['train']
        # test_loader = task['test'] # Available for evaluating catastrophic forgetting

        print(f"=== Starting Task: {task_id} ===")
        
        for epoch in range(args.epochs_per_task):
            if is_stateless:
                epoch_loss, epoch_acc = training_functions.train_stateless(
                    model, train_loader, optimizer, criterion, device
                )
            else:
                epoch_loss, epoch_acc = training_functions.train_stateful(
                    model, train_loader, optimizer, criterion, device
                )
            
            # Now we can print both metrics safely!
            print(f"Epoch {epoch+1}/{args.epochs_per_task} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        # Save weights after completing the task (simulating a continual learning checkpoint)
        weight_filename = f"hgrn_{args.mode}_{task_id}.pt"
        weight_filepath = os.path.join(save_dir, weight_filename)
        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weight_filepath)
        print(f"Saved weights to {weight_filepath}\n")

    print("Training sequence complete.")

if __name__ == "__main__":
    main()