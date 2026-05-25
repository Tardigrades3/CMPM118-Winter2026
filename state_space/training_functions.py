import torch

def train_naive_stateless(model, task_loader, optimizer, criterion, device):
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

def train_naive_stateful(model, task_loader, optimizer, criterion, device):
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

def train_replay_stateless(model, task_loader, optimizer, criterion, device, memory_buffer=None, replay_batch_size=16):
    """
    Standard training loop WITH Experience Replay. 
    Memory is wiped clean at the start of every sequence.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (sequences, labels, attention_mask) in enumerate(task_loader):
        # 1. Move current task data to device
        sequences = sequences.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        # --- NEW CL LOGIC: EXPERIENCE REPLAY ---
        # 2. If we are past Task 1, the memory buffer will have data to sample
        if memory_buffer is not None and len(memory_buffer) > 0:
            
            # Pull a random batch of old data
            replay_seqs, replay_labels, replay_masks = memory_buffer.sample(replay_batch_size)
            
            # Move old data to device
            replay_seqs = replay_seqs.to(device)
            replay_labels = replay_labels.to(device)
            replay_masks = replay_masks.to(device)
            
            # Concatenate along the batch dimension (dim=0)
            # If current batch is 32 and replay is 16, the new batch size is 48
            sequences = torch.cat([sequences, replay_seqs], dim=0)
            labels = torch.cat([labels, replay_labels], dim=0)
            attention_mask = torch.cat([attention_mask, replay_masks], dim=0)
        # ---------------------------------------
        
        optimizer.zero_grad()
        
        # 3. Forward pass with the newly combined batch
        logits, _ = model(x=sequences, states=None, attention_mask=attention_mask)
        
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Tracking metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = total_loss / len(task_loader)
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

def train_replay_stateful(model, task_loader, optimizer, criterion, device, memory_buffer=None, replay_batch_size=16):
    """
    Continual learning training loop WITH Experience Replay.
    Maintains the hidden state for the current data stream, while running
    stateless forward passes for discontinuous historical replay data.
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
        
        # 1. State Detachment & Slicing for current continuous batch
        current_batch_size = sequences.size(0)
        
        if h_states is not None:
            detached_states = []
            for h in h_states:
                if h is not None:
                    if isinstance(h, tuple):
                        detached_states.append(tuple(state[:current_batch_size].detach() for state in h))
                    else:
                        detached_states.append(h[:current_batch_size].detach())
                else:
                    detached_states.append(None)
            h_states = detached_states

        # 2. First Forward Pass: Current Continuous Task
        logits_current, h_states = model(x=sequences, states=h_states, attention_mask=attention_mask)
        loss_current = criterion(logits_current, labels)
        
        # 3. Second Forward Pass: Historical Replay (if available)
        loss_replay = 0.0
        if memory_buffer is not None and len(memory_buffer) > 0:
            # Sample historical data
            replay_seqs, replay_labels, replay_masks = memory_buffer.sample(replay_batch_size)
            
            replay_seqs = replay_seqs.to(device)
            replay_labels = replay_labels.to(device)
            replay_masks = replay_masks.to(device)
            
            # Stateless forward pass for discontinuous chunks (states=None)
            logits_replay, _ = model(x=replay_seqs, states=None, attention_mask=replay_masks)
            loss_replay = criterion(logits_replay, replay_labels)

        # 4. Combine Loss and Backpropagate
        # The optimizer will update weights based on BOTH the current task and the old memories
        loss = loss_current + loss_replay
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Tracking metrics (tracking accuracy only on the current task to monitor current-task convergence)
        total_loss += loss.item()
        _, predicted = torch.max(logits_current, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = total_loss / len(task_loader)
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

def train_ewc_stateful(model, task_loader, optimizer, criterion, device, fisher_dict=None, optpar_dict=None, ewc_lambda=1000):
    """
    Continual learning training loop WITH Elastic Weight Consolidation (EWC).
    Maintains the hidden state across batches, and penalizes the optimizer
    for changing weights that were important to previous subjects.
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
        
        # 1. State Detachment & Slicing (Handles partial batches at the end of the loader)
        current_batch_size = sequences.size(0)
        
        if h_states is not None:
            detached_states = []
            for h in h_states:
                if h is not None:
                    if isinstance(h, tuple):
                        detached_states.append(tuple(state[:current_batch_size].detach() for state in h))
                    else:
                        detached_states.append(h[:current_batch_size].detach())
                else:
                    detached_states.append(None)
            h_states = detached_states

        # 2. Forward Pass
        logits, h_states = model(x=sequences, states=h_states, attention_mask=attention_mask)
        
        # 3. Standard Classification Loss
        loss = criterion(logits, labels)
        
        # --- NEW CL LOGIC: EWC PENALTY ---
        # 4. If we have finished at least one previous subject, apply the penalty
        ewc_loss = 0.0
        if fisher_dict is not None and optpar_dict is not None:
            for name, param in model.named_parameters():
                if name in fisher_dict:
                    # Fisher acts as the 'stiffness' multiplier for the spring
                    fisher = fisher_dict[name]
                    # Optpar is the 'anchor point' (the optimal weight from Subject 1)
                    optpar = optpar_dict[name]
                    
                    # Penalty = Fisher * (Current_Weight - Old_Weight)^2
                    ewc_loss += (fisher * (param - optpar).pow(2)).sum()
            
            # Combine the losses using the lambda scaling factor
            loss = loss + (ewc_lambda * ewc_loss)
        # ---------------------------------

        # 5. Backpropagate the combined loss
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

def compute_fisher(model, task_loader, device):
    """
    Computes the Fisher Information Matrix to determine weight importance.
    Called once at the very end of training on a specific subject.
    """
    model.eval()
    fisher_dict = {}
    optpar_dict = {}
    
    # Initialize the dictionaries with zeros
    for name, param in model.named_parameters():
        optpar_dict[name] = param.data.clone().to(device)
        fisher_dict[name] = torch.zeros_like(param.data).to(device)
        
    h_states = None
    
    # Run through the dataset one final time to measure gradient sensitivity
    for sequences, labels, attention_mask in task_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        current_batch_size = sequences.size(0)
        
        # Slicing logic for the states
        if h_states is not None:
            detached_states = []
            for h in h_states:
                if h is not None:
                    if isinstance(h, tuple):
                        detached_states.append(tuple(state[:current_batch_size].detach() for state in h))
                    else:
                        detached_states.append(h[:current_batch_size].detach())
                else:
                    detached_states.append(None)
            h_states = detached_states

        model.zero_grad()
        logits, h_states = model(x=sequences, states=h_states, attention_mask=attention_mask)
        
        # Calculate the log likelihood 
        log_likelihood = torch.nn.functional.log_softmax(logits, dim=1)
        
        # We use the predicted class to calculate Fisher, not the true label
        predicted_classes = logits.max(1)[1]
        
        loss = torch.nn.functional.nll_loss(log_likelihood, predicted_classes)
        loss.backward(retain_graph=True)
        
        # Accumulate the squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) / len(task_loader)
                
    return fisher_dict, optpar_dict