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
            h = None
            sequences = sequences.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            logits, h = model(x=sequences, states=h, attention_mask=attention_mask)
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


def save_evaluation_results(results_dict, mode, exercise, subject_id=None, results_dir="results"):
    # Second precision + subject id so a CIL sweep (many subjects finishing within
    # the same minute) does not overwrite earlier subjects' result files.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Allow an absolute or relative results dir so a sweep can be kept separate
    # (e.g. results_diffexp/) from another branch's results for clean comparison.
    save_dir = results_dir if os.path.isabs(results_dir) else os.path.join(_PROJECT_ROOT, results_dir)
    os.makedirs(save_dir, exist_ok=True)

    subj_part = f"_s{subject_id}" if subject_id is not None else ""
    filename = f"eval_{mode}{subj_part}_ex{exercise}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)

    return filepath
