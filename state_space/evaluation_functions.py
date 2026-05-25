import torch
import os
import json
from datetime import datetime

def evaluate(model, test_loader, criterion, device, num_classes=17):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Dictionaries to track hits and totals per class
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    with torch.no_grad():
        for sequences, labels, attention_mask in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            
            logits, _ = model(x=sequences, states=None, attention_mask=attention_mask)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            
            # Global metrics
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Per-Class metrics
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
    eval_loss = total_loss / len(test_loader)
    eval_acc = correct_predictions / total_samples
    
    # Calculate percentages, avoiding division by zero
    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[f"class_{i}"] = class_correct[i] / class_total[i]
        else:
            per_class_acc[f"class_{i}"] = 0.0
    
    return eval_loss, eval_acc, per_class_acc

def save_evaluation_results(results_dict, mode, exercise):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(os.getcwd(), "evaluations")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"eval_{mode}_ex{exercise}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    return filepath