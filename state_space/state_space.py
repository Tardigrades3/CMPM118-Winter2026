import torch
from torch import nn
import torch.nn.functional as F

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

    def forward(self, x, h_state=None):
        # x shape incoming: [Batch, Sequence, Features]
        
        # 1. Projections (Done in whatever dtype x is, usually fp32 or bf16)
        c_t_seq = F.silu(self.c_proj(x))
        mu_t_seq = torch.sigmoid(self.mu_proj(x))
        
        gamma_clamped = torch.clamp(self.gamma, 0.0, 0.99)
        lambda_t_seq = gamma_clamped + (1 - gamma_clamped) * mu_t_seq
        
        # 2. Shape Adjustment for FLA: [Batch, Heads=1, Sequence, HeadDim]
        # We also enforce contiguity, which Triton kernels mandate.
        c_t_seq = c_t_seq.unsqueeze(1).contiguous()
        lambda_t_seq = lambda_t_seq.unsqueeze(1).contiguous()
        
        if h_state is not None:
            # Hidden state also needs the dummy head dimension: [Batch, 1, HeadDim]
            h_state = h_state.unsqueeze(1).contiguous()

        # 3. Data Type Cast
        # FLA ops require half-precision (bfloat16 or float16)
        current_dtype = c_t_seq.dtype
        if current_dtype == torch.float32:
            c_t_seq = c_t_seq.to(torch.bfloat16)
            lambda_t_seq = lambda_t_seq.to(torch.bfloat16)
            if h_state is not None:
                h_state = h_state.to(torch.bfloat16)

        # Note on lambda_t_seq: If your version of FLA expects log-decays, 
        # change the next line to `lambda_t_seq.log()`.
        out, next_h_state = chunk_hgrn(
            c_t_seq, 
            lambda_t_seq,  
            initial_state=h_state, 
            output_final_state=True 
        )
        
        # 4. Revert Shape back to [Batch, Sequence, Features] and original dtype
        out = out.squeeze(1).to(current_dtype)
        if next_h_state is not None:
            next_h_state = next_h_state.squeeze(1).to(current_dtype)
            
        return out, next_h_state