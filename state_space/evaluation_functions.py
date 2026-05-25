import torch
import os
import json
from datetime import datetime

def evaluate(model, test_loader, criterion, device):
    """
    Evaluates the model on a given test loader.
    Strictly stateless and gradient-free.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, labels, attention_mask in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            # 1. Stateless forward pass for clean evaluation
            logits, _ = model(x=sequences, states=None, attention_mask=attention_mask)
            
            # 2. Calculate metrics
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
    eval_loss = total_loss / len(test_loader)
    eval_acc = correct_predictions / total_samples
    
    return eval_loss, eval_acc

def save_evaluation_results(results_dict, mode, exercise):
    """
    Saves the evaluation dictionary to a dedicated evaluations folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create the evaluations directory if it doesn't exist
    save_dir = os.path.join(os.getcwd(), "evaluations")
    os.makedirs(save_dir, exist_ok=True)
    
    # Format the filename to match the training mode
    filename = f"eval_{mode}_ex{exercise}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    # Dump the results to a formatted JSON
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    return filepath