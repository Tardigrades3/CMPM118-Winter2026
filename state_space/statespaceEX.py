import torch
import torch.nn as nn
import torch.nn.functional as F

class HRU(nn.Module):
    """
    Hierarchically Gated Recurrent Unit (HRU).
    Uses element-wise linear recurrence with a layer-dependent lower bound on the forget gate.
    """
    def __init__(self, d_model: int, layer_idx: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        
        # Projections for the candidate state, forget gate, and output gate
        self.proj_c = nn.Linear(d_model, d_model)
        self.proj_f = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        
        # The Hierarchical Lower Bound (\gamma)
        # Initializes higher for deeper layers so they retain memory longer.
        init_val = layer_idx / max(1, num_layers - 1) 
        self.lower_bound = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
        
        # Output normalization and projection
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Candidate State (c_t)
        c = F.silu(self.proj_c(x))
        
        # 2. Hierarchical Forget Gate (\lambda_t)
        # Formula: \lambda_t = \gamma + (1 - \gamma) * \sigma(W_f x_t)
        # We pass lower_bound through sigmoid to ensure it strictly stays between 0 and 1
        gamma = torch.sigmoid(self.lower_bound) 
        f = gamma + (1 - gamma) * torch.sigmoid(self.proj_f(x))
        
        # 3. Element-wise Linear Recurrence
        # Note: For production, you would use a parallel prefix scan (e.g., from the 'fla' library).
        # We use a sequential loop here for structural clarity.
        h = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)
        h_seq = []
        
        for t in range(seq_len):
            # h_t = \lambda_t * h_{t-1} + (1 - \lambda_t) * c_t
            h = f[:, t, :] * h + (1 - f[:, t, :]) * c[:, t, :]
            h_seq.append(h.unsqueeze(1))
            
        h_seq = torch.cat(h_seq, dim=1) # (batch_size, seq_len, d_model)
        
        # 4. Output Gating
        g = torch.sigmoid(self.proj_g(x))
        
        # Apply output gate, normalize, and project
        out = self.norm(h_seq * g)
        out = self.proj_out(out)
        
        return out

class HGRNBlock(nn.Module):
    """
    A full HGRN Layer combining the Token Mixer (HRU) and Channel Mixer (GLU).
    """
    def __init__(self, d_model: int, layer_idx: int, num_layers: int, expansion_factor: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.hru = HRU(d_model, layer_idx, num_layers)
        
        self.norm2 = nn.LayerNorm(d_model)
        # Gated Linear Unit (GLU) for channel mixing
        self.glu = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor * 2),
            nn.GLU(dim=-1),
            nn.Linear(d_model * expansion_factor, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.hru(self.norm1(x))
        x = x + self.glu(self.norm2(x))
        return x

class HGRNModel(nn.Module):
    """
    The full Network stacking multiple HGRN blocks.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            HGRNBlock(d_model, layer_idx=i, num_layers=num_layers) 
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    d_model = 128
    num_layers = 4
    
    model = HGRNModel(vocab_size, d_model, num_layers)
    
    # Dummy token ids
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)) 
    logits = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")



# example functions for stateless and stateful training

def train_stateless(model, task_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch_idx, (sequences, labels, attention_mask) in enumerate(task_loader):
        optimizer.zero_grad()
        
        # 1. Initialize hidden state to None (or zeros) for EVERY batch
        h_state = None 
        
        # Forward pass
        # The HGRN processes the sequence and returns predictions
        predictions, _ = model(sequences, h_state, attention_mask)
        
        # Calculate loss (assuming predictions are shape [Batch, Num_Classes])
        loss = criterion(predictions, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(task_loader)

def train_stateful(model, task_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    # 1. Initialize hidden state OUTSIDE the batch loop
    # It resets only when a new continual learning task (subject/exercise) begins
    h_state = None 
    
    for batch_idx, (sequences, labels, attention_mask) in enumerate(task_loader):
        optimizer.zero_grad()
        
        # 2. Detach the hidden state from the previous computation graph
        if h_state is not None:
            # If h_state is a tuple (like in some RNNs), detach all parts.
            # If it's a single tensor, just do: h_state = h_state.detach()
            if isinstance(h_state, tuple):
                h_state = tuple(h.detach() for h in h_state)
            else:
                h_state = h_state.detach()
                
        # Forward pass: Pass the detached state from the PREVIOUS batch
        predictions, h_state = model(sequences, h_state, attention_mask)
        
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(task_loader)