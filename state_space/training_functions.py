import torch

def train_stateless(model, task_loader, optimizer, criterion, device):
    """
    Standard training loop. 
    Memory is wiped clean at the start of every sequence.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (sequences, labels, attention_mask) in enumerate(task_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        optimizer.zero_grad()
        
        # FIXED: Changed `d_model=sequences` to `x=sequences` and `h_states` to `states`
        logits, _ = model(x=sequences, states=None, attention_mask=attention_mask)
        
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = total_loss / len(task_loader)
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc


def train_stateful(model, task_loader, optimizer, criterion, device):
    """
    Continual learning training loop.
    Maintains the hidden state across batches to simulate an unbroken data stream.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    h_states = None 
    
    for batch_idx, (sequences, labels, attention_mask) in enumerate(task_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        optimizer.zero_grad()
        
        # 1. Get the actual size of the current batch
        current_batch_size = sequences.size(0)
        
        # 2. Slice the detached states to match the current batch size
        if h_states is not None:
            detached_states = []
            for h in h_states:
                if h is not None:
                    if isinstance(h, tuple):
                        # Slice the batch dimension AND detach
                        detached_states.append(tuple(state[:current_batch_size].detach() for state in h))
                    else:
                        # Slice the batch dimension AND detach
                        detached_states.append(h[:current_batch_size].detach())
                else:
                    detached_states.append(None)
            h_states = detached_states

        logits, h_states = model(x=sequences, states=h_states, attention_mask=attention_mask)
        
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = total_loss / len(task_loader)
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc