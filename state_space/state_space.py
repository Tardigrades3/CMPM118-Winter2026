import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F


# Import the accelerated kernel
try:
    from fla.ops.hgrn import chunk_hgrn
except ImportError:
    raise ImportError("Please install flash-linear-attention to use the CUDA kernel.")

class AcceleratedHRU(nn.Module):
    def __init__(self, hidden_dim, layer_idx, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mu_proj = nn.Linear(hidden_dim, hidden_dim)
        
        gamma_init = layer_idx / max(1, num_layers - 1) 
        self.gamma = nn.Parameter(torch.tensor([gamma_init]), requires_grad=True)
        
        # Note: Depending on the specific FLA version, the phase parameter (theta) 
        # might be handled slightly differently, but the underlying parallel scan is the same.

    def forward(self, x, h_state=None):
        # x shape: [Batch, Sequence, Features]
        
        # 1. Generate the candidate and base gates exactly as before
        c_t_seq = F.silu(self.c_proj(x))
        mu_t_seq = torch.sigmoid(self.mu_proj(x))
        
        # 2. Calculate the hierarchical forget gate
        gamma_clamped = torch.clamp(self.gamma, 0.0, 0.99)
        lambda_t_seq = gamma_clamped + (1 - gamma_clamped) * mu_t_seq
        
        # 3. Apply the CUDA-accelerated Parallel Scan
        # The kernel takes the sequence of candidates, the sequence of gates, 
        # and the initial hidden state, and computes the entire sequence in parallel.
        # It returns the output sequence and the final hidden state.
        
        out, next_h_state = chunk_hgrn(
            c_t_seq, 
            lambda_t_seq, 
            initial_state=h_state, 
            output_final_state=True # Crucial for stateful training
        )
        
        return out, next_h_state

