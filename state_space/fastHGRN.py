import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.hgrn import chunk_hgrn 

class FastHGRNLayer(nn.Module):
    def __init__(self, d_model, num_heads, layer_idx, num_layers):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        
        initial_gamma = layer_idx / num_layers 
        initial_gamma_logit = torch.log(torch.tensor(initial_gamma + 1e-4) / (1 - initial_gamma + 1e-4))
        self.gamma_logit = nn.Parameter(initial_gamma_logit)
        
        self.proj_f = nn.Linear(d_model, d_model)
        self.proj_i = nn.Linear(d_model, d_model)
        self.proj_c = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        
        self.out_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape
        
        f_logits = self.proj_f(x)
        i_logits = self.proj_i(x)
        c = self.proj_c(x)
        g_logits = self.proj_g(x)
        
        gamma = torch.sigmoid(self.gamma_logit)
        f = gamma + (1 - gamma) * torch.sigmoid(f_logits)
        i = F.silu(i_logits) 
        
        f = rearrange(f, 'b l (h d) -> b h l d', h=self.num_heads)
        i = rearrange(i, 'b l (h d) -> b h l d', h=self.num_heads)
        c = rearrange(c, 'b l (h d) -> b h l d', h=self.num_heads)
        
        h_out, final_state = chunk_hgrn(f, i, c, initial_state=state, output_final_state=True)
        
        h_out = rearrange(h_out, 'b h l d -> b l (h d)')
        
        g = torch.sigmoid(g_logits)
        out = g * h_out
        
        out = self.proj_out(self.out_norm(out))
        
        return out, final_state


class HGRNModel(nn.Module):
    def __init__(self, in_channels, d_model, num_classes, num_layers, num_heads=4):
        super().__init__()
        # 1. Project continuous EMG signals into the hidden dimension space
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # 2. Build the state space layers
        self.layers = nn.ModuleList([
            FastHGRNLayer(d_model, num_heads, i, num_layers) 
            for i in range(num_layers)
        ])
        
        # 3. Final gesture classification head
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, states=None, attention_mask=None):
        # x shape: (batch, seq_len, in_channels)
        x = self.input_proj(x)
        
        # Apply padding mask before recurrence so zero-padding doesn't corrupt the state
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        # Initialize the state list if none is provided
        if states is None:
            states = [None] * len(self.layers)
            
        next_states = []
        
        # Route x sequentially through layers, mapping states index-to-index
        for i, layer in enumerate(self.layers):
            x, next_state = layer(x, state=states[i])
            next_states.append(next_state)
            
        # --- Temporal Pooling ---
        # To classify the whole window, we pull the features from the very last valid time step.
        if attention_mask is not None:
            # Find the actual length of each sequence in the batch
            seq_lengths = attention_mask.sum(dim=1).long()
            # Get the index of the last valid element (length - 1)
            last_indices = (seq_lengths - 1).clamp(min=0)
            
            # Extract those specific timesteps
            batch_indices = torch.arange(x.size(0), device=x.device)
            pooled_x = x[batch_indices, last_indices, :]
        else:
            pooled_x = x[:, -1, :]
            
        # Calculate final logits
        logits = self.head(pooled_x)
        
        return logits, next_states