import torch
import os
import json
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for sequences, labels, attention_mask in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            logits, _ = model(x=sequences, states=None, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] = class_total.get(label, 0) + 1
                if label == predicted[i].item():
                    class_correct[label] = class_correct.get(label, 0) + 1

    eval_loss = total_loss / len(test_loader)
    eval_acc = correct_predictions / total_samples

    per_class_acc = {
        f"class_{i}": class_correct.get(i, 0) / class_total[i]
        for i in sorted(class_total.keys())
    }

    return eval_loss, eval_acc, per_class_acc


def save_evaluation_results(results_dict, mode, exercise):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"eval_{mode}_ex{exercise}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)

    return filepath
