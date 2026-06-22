import torch
import torch.nn as nn


class LSTMLayer(nn.Module):
    """Single LSTM layer with LayerNorm and optional dropout.

    States are stored as (B, H) tensors (batch on dim-0) so that the
    existing stateful training functions can slice them correctly with
    state[:current_batch_size]. PyTorch's default LSTM state shape is
    (num_layers, B, H); we squeeze/unsqueeze across the forward pass to
    convert between the two representations.
    """
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, state=None):
        # x     : (B, T, H)
        # state : tuple (h, c) each (B, H), or None
        if state is not None:
            h, c = state
            lstm_state = (h.unsqueeze(0), c.unsqueeze(0))  # → (1, B, H)
        else:
            lstm_state = None

        out, (h_n, c_n) = self.lstm(x, lstm_state)
        out = self.drop(self.norm(out))

        # Return states as (B, H) so batch sits on dim-0
        return out, (h_n.squeeze(0), c_n.squeeze(0))


class LSTMModel(nn.Module):
    """Multi-layer LSTM baseline with the same interface as HGRNModel.

    Drop-in compatible with all training functions in hgrn/training.py
    (stateless, stateful, replay, EWC, herding) and with HerdingBuffer's
    get_features() call.

    Args:
        in_channels : number of EMG channels (10 for NinaPro DB1)
        d_model     : hidden state dimension (matches HGRN for fair comparison)
        num_classes : output head size
        num_layers  : number of stacked LSTM layers (matches HGRN for fair comparison)
        dropout     : inter-layer dropout probability (applied after LayerNorm,
                      not applied on the final layer)
    """
    def __init__(self, in_channels, d_model, num_classes, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList([
            LSTMLayer(
                input_size=d_model,
                hidden_size=d_model,
                dropout=dropout if i < num_layers - 1 else 0.0  # no dropout on final layer
            )
            for i in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, states=None, attention_mask=None):
        # x             : (B, T, in_channels)
        # states        : list of per-layer (h, c) tuples, or None
        # attention_mask: (B, T) long tensor, 1 = valid timestep
        x = self.input_proj(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        if states is None:
            states = [None] * len(self.layers)

        next_states = []
        for i, layer in enumerate(self.layers):
            x, next_state = layer(x, states[i])
            next_states.append(next_state)

        # Temporal pooling: extract the last valid timestep
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
        """Last-timestep hidden features — used by HerdingBuffer.select_exemplars."""
        x = self.input_proj(x)
        for layer in self.layers:
            x, _ = layer(x)
        return x[:, -1, :]
