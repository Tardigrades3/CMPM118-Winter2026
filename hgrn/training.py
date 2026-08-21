import torch


def flatten_ewc_penalty_inputs(model, fisher_dict, optpar_dict):
    """Flatten Fisher/anchor weights and the matching live parameters into
    single vectors, in one consistent parameter order.

    Computed once per training call (not once per step, and not once per
    parameter tensor), so the per-step penalty below is a handful of large
    vectorized ops instead of ~4 small ops per parameter tensor -- the
    per-tensor Python loop was the dominant cost on GPU (many small kernel
    launches, each paying a fixed dispatch overhead regardless of how tiny
    the op is), not the arithmetic itself.
    """
    names, params = zip(*[(name, p) for name, p in model.named_parameters() if name in fisher_dict])
    fisher_flat = torch.cat([fisher_dict[name].flatten() for name in names])
    optpar_flat = torch.cat([optpar_dict[name].flatten() for name in names])
    return fisher_flat, optpar_flat, list(params)


def ewc_penalty(fisher_flat, optpar_flat, params):
    """sum_i F_i * (theta_i - theta_i*)^2 over every flattened parameter, as
    one vectorized op. Mathematically identical to summing the penalty
    tensor-by-tensor (sum of sums = sum of the concatenated whole), so this
    is a performance change only -- it does not alter the loss, the
    gradients, or downstream accuracy/forgetting."""
    params_flat = torch.cat([p.flatten() for p in params])
    return (fisher_flat * (params_flat - optpar_flat).pow(2)).sum()


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

def train_replay_stateful(model, task_loader, optimizer, criterion, device, memory_buffer=None, replay_batch_size=16, replay_weight=0.5, noise_std=0.01):
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
            
            noise = torch.randn_like(replay_seqs) * noise_std
            replay_seqs = replay_seqs + noise
            
            replay_seqs = replay_seqs.to(device)
            replay_labels = replay_labels.to(device)
            replay_masks = replay_masks.to(device)
            
            # Stateless forward pass for discontinuous chunks (states=None)
            logits_replay, _ = model(x=replay_seqs, states=None, attention_mask=replay_masks)
            loss_replay = criterion(logits_replay, replay_labels)

        # 4. Combine Loss and Backpropagate
        # The optimizer will update weights based on BOTH the current task and the old memories
        
        loss = loss_current + (replay_weight * loss_replay)
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

    use_ewc = fisher_dict is not None and optpar_dict is not None
    if use_ewc:
        fisher_flat, optpar_flat, ewc_params = flatten_ewc_penalty_inputs(model, fisher_dict, optpar_dict)

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
        
        # --- EWC PENALTY (vectorized, see flatten_ewc_penalty_inputs/ewc_penalty) ---
        if use_ewc:
            loss = loss + (ewc_lambda * ewc_penalty(fisher_flat, optpar_flat, ewc_params))
        # -------------------

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

# def train_generative_replay_stateful(model, task_loader, optimizer, criterion, device, generator=None, gen_batch_size=16):
#     """
#     Stateful training with Generative Replay.
#     The generator provides synthetic 'past' data to prevent forgetting.
#     """
#     model.train()
#     # ... (Keep existing state-detachment logic from train_replay_stateful) ...

#     for batch_idx, (sequences, labels, attention_mask) in enumerate(task_loader):
#         # 1. Standard Forward Pass (Current Task)
#         logits_current, h_states = model(x=sequences.to(device), states=h_states, ...)
#         loss = criterion(logits_current, labels.to(device))
        
#         # 2. Generative Replay: Dream up past data
#         if generator is not None:
#             generator.eval()
#             # Generate synthetic EMG signals from random noise
#             noise = torch.randn(gen_batch_size, sequences.size(1), 10).to(device)
#             with torch.no_grad():
#                 syn_seqs = generator(noise)
            
#             # The 'dream' is fed to the model to refresh memory
#             logits_syn, _ = model(x=syn_seqs, states=None, attention_mask=None)
            
#             # Distillation Loss: Model should maintain its 'knowledge' of what these 
#             # gestures look like, even as it learns the new subject.
#             # We use the previous model's output as the 'teacher'.
#             syn_labels = logits_syn.detach().argmax(dim=1)
#             loss += criterion(logits_syn, syn_labels)

#         # 3. Backprop
#         loss.backward()
#         optimizer.step()
        
#     return epoch_loss, epoch_acc

def train_ewc_stateless(model, task_loader, optimizer, criterion, device,
                        fisher_dict=None, optpar_dict=None, ewc_lambda=1000):
    """EWC with per-batch state reset — the principled choice for windowed EMG."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    use_ewc = fisher_dict is not None and optpar_dict is not None
    if use_ewc:
        fisher_flat, optpar_flat, ewc_params = flatten_ewc_penalty_inputs(model, fisher_dict, optpar_dict)

    for sequences, labels, attention_mask in task_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        logits, _ = model(x=sequences, states=None, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        if use_ewc:
            loss = loss + ewc_lambda * ewc_penalty(fisher_flat, optpar_flat, ewc_params)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(task_loader), correct_predictions / total_samples


# HELPER FUNCTIONS
def compute_fisher(model, task_loader, device, stateless=False):
    """
    Computes the Fisher Information Matrix to determine weight importance.
    Called once at the very end of training on a specific subject.

    stateless=True resets hidden state per batch (use for ewc_stateless).
    stateless=False (default) carries state across batches (original behaviour).
    """
    # NOTE: must be train() — cuDNN LSTM backward requires training mode
    model.train()
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

        if stateless:
            h_states = None
        else:
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
        loss.backward()

        # Accumulate the squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) / len(task_loader)

    return fisher_dict, optpar_dict

