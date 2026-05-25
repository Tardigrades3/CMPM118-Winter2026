import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops.hgrn import chunk_hgrn 

class FastHGRNLayer(nn.Module):
    def __init__(self, d_model, layer_idx, num_layers):
        super().__init__()
        self.d_model = d_model
        
        # 1. The Hierarchical Lower Bound (Gamma)
        initial_gamma = layer_idx / num_layers 
        initial_gamma_logit = torch.log(torch.tensor(initial_gamma + 1e-4) / (1 - initial_gamma + 1e-4))
        self.gamma_logit = nn.Parameter(initial_gamma_logit)
        
        # 2. Linear Projections
        self.proj_f = nn.Linear(d_model, d_model)
        self.proj_i = nn.Linear(d_model, d_model)
        self.proj_c = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        
        # 3. Output Normalization and Projection
        self.out_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x, state=None):
        # 1. Compute all gate logits for the entire sequence at once
        f_logits = self.proj_f(x)
        i_logits = self.proj_i(x)
        c = self.proj_c(x)
        g_logits = self.proj_g(x)
        
        # 2. Apply activations and the hierarchical forget-gate bound
        gamma = torch.sigmoid(self.gamma_logit)
        f = gamma + (1 - gamma) * torch.sigmoid(f_logits)
        i = F.silu(i_logits) 
        
        # 3. Combine the input gate and candidate
        x_recurrence = i * c
        
        # 4. The FLA kernel expects the forget gate in log-space
        g = torch.log(f)
        
        # 5. THE KERNEL: Pass the 3D tensors directly (Batch, Time, Dimension)
        h_out, final_state = chunk_hgrn(
            x=x_recurrence, 
            g=g, 
            initial_state=state, 
            output_final_state=True
        )
        
        # 6. Apply output gating
        g_out = torch.sigmoid(g_logits)
        out = g_out * h_out
        
        # 7. Final norm and projection
        out = self.proj_out(self.out_norm(out))
        
        return out, final_state


class HGRNModel(nn.Module):
    def __init__(self, in_channels, d_model, num_classes, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Build the layers (no longer passing a num_heads argument)
        self.layers = nn.ModuleList([
            FastHGRNLayer(d_model, i, num_layers) 
            for i in range(num_layers)
        ])
        
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, states=None, attention_mask=None):
        x = self.input_proj(x)
        
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        if states is None:
            states = [None] * len(self.layers)
            
        next_states = []
        
        for i, layer in enumerate(self.layers):
            x, next_state = layer(x, state=states[i])
            next_states.append(next_state)
            
        # Temporal Pooling
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long()
            last_indices = (seq_lengths - 1).clamp(min=0)
            
            batch_indices = torch.arange(x.size(0), device=x.device)
            pooled_x = x[batch_indices, last_indices, :]
        else:
            pooled_x = x[:, -1, :]
            
        logits = self.head(pooled_x)
        
        return logits, next_states
    def get_features(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x, _ = layer(x)
        # Assuming we use the last hidden state for the centroid
        return x[:, -1, :]