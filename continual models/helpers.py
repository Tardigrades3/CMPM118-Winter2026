
import torch
from torch import nn
import tqdm
from torchmetrics.classification import MulticlassAccuracy

def train_base(model, train, device, optimizer, loss_func, epochs = 100, accuracy_over_time = False, test = None):
    test_accs = []
    train_accs = []
    for epoch in range(epochs):
        pbar = tqdm(train, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        if accuracy_over_time and test: 
            test_accs.append(find_accuracy(model, test, device, loss_func))
            train_accs.append(find_accuracy(model, train, device, loss_func))
        for emg, label in pbar:
            emg = emg.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(emg)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
    if accuracy_over_time:
        return (test_accs, train_accs)
    else: 
        return None


def expand_classifier(model, new_size, device):
  old_fc = model.finalLayer
  in_features = old_fc.in_features
  old_out = old_fc.out_features
  newlayer = nn.Linear(in_features, old_out + new_size).to(device)

  with torch.no_grad():
    newlayer.weight[:old_out].copy_(old_fc.weight)
    newlayer.bias[:old_out].copy_(old_fc.bias)
  model.finalLayer = newlayer
  return model

def find_accuracy(model, data, device, loss_func):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        correct = 0
        for emg, labels in data:
            emg = emg.to(device)
            labels = labels.to(device)
            logits = model(emg)
            loss = loss_func(logits, labels)

            total_loss += loss.item() * emg.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += emg.size(0)
    print(f"Average Loss: {total_loss/correct}, Accuracy: {correct/total}")

def multi_class_acc_measure(train, model, device, num_classes):
    per_class_acc = MulticlassAccuracy(
        num_classes=num_classes,
        average=None
    ).to(device)

    model.eval()

    with torch.no_grad():
        for emg, label in train:
            emg = emg.to(device).float()
            label = label.to(device)
            logits = model(emg)
            preds = logits.argmax(dim=1)
            per_class_acc.update(preds, label)

    accs = per_class_acc.compute()
    return accs

def drop_rest_label(x_train, y_train, x_test, y_test):
    mask = y_train != 0
    x_train = x_train[mask]
    y_train = y_train[mask]

    mask = y_test != 0
    x_test = x_test[mask]
    y_test = y_test[mask]

    y_train -= 1
    y_test -= 1
    return x_train, y_train, x_test, y_test