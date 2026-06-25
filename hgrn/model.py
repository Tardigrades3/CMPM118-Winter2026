import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU-accelerated chunked HGRN scan (Triton/FLA). Falls back to a pure-PyTorch
# timestep loop when FLA is unavailable (e.g. local CPU / macOS dev boxes) so the
# same code runs on EC2 GPUs and on a laptop without an import error.
try:
    from fla.ops.hgrn import chunk_hgrn
    _HAS_FLA = True
except ImportError:
    chunk_hgrn = None
    _HAS_FLA = False


class FastHGRNLayer(nn.Module):
    def __init__(self, d_model, layer_idx, num_layers):
        super().__init__()
        self.d_model = d_model
        
        # Hierarchical gamma initialization
        initial_gamma = layer_idx / num_layers
        initial_gamma_logit = torch.log(
            torch.tensor(initial_gamma + 1e-4) / (1 - initial_gamma + 1e-4)
        )
        self.gamma_logit = nn.Parameter(initial_gamma_logit)

        # Linear projections
        self.proj_f = nn.Linear(d_model, d_model)
        self.proj_i = nn.Linear(d_model, d_model)
        self.proj_c = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)

        # Output normalization + projection
        self.out_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x, state=None):
        """
        x: (batch, time, d_model)
        state: (batch, d_model)
        """
        B, T, D = x.shape

        # Compute gates
        f_logits = self.proj_f(x)
        i_logits = self.proj_i(x)
        c = self.proj_c(x)
        g_logits = self.proj_g(x)

        gamma = torch.sigmoid(self.gamma_logit)
        f = gamma + (1 - gamma) * torch.sigmoid(f_logits)
        i = F.silu(i_logits)

        # Candidate update
        x_recurrence = i * c

        if _HAS_FLA and x.is_cuda:
            # GPU path: the FLA kernel expects the forget gate in log-space and
            # runs the whole (B, T, D) linear recurrence in one chunked launch.
            g = torch.log(f)
            h_out, final_state = chunk_hgrn(
                x=x_recurrence,
                g=g,
                initial_state=state,
                output_final_state=True,
            )
        else:
            # CPU fallback: mathematically identical sequential scan
            # (h_t = f_t * h_{t-1} + x_recurrence_t  ==  exp(log f) recurrence).
            if state is None:
                state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            outputs = []
            h = state
            for t in range(T):
                h = f[:, t] * h + x_recurrence[:, t]
                outputs.append(h.unsqueeze(1))
            h_out = torch.cat(outputs, dim=1)
            final_state = h

        # Output gating
        g_out = torch.sigmoid(g_logits)
        out = g_out * h_out

        # Final projection
        out = self.proj_out(self.out_norm(out))

        return out, final_state
class HGRNModel(nn.Module):
    def __init__(self, in_channels, d_model, num_classes, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)

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

        # Temporal pooling
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
        return x[:, -1, :]
