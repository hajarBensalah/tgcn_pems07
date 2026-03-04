"""
models/tgcn_model.py
--------------------
Full T-GCN model: a sequence of T-GCN cells followed by a linear
output layer for multi-step traffic speed prediction.

Architecture overview
---------------------
Input  : [batch, seq_len, num_nodes, in_features]
         (B sequences of length T, each node carries F features)

Recurrence (vectorized — no Python loop over batch):
    For t = 1 ... seq_len:
        H_t = TGCNCell(X_t, H_{t-1}, A_norm)
    where H_t has shape [batch, num_nodes, hidden_dim]

Output projection (applied to final hidden state H_{seq_len}):
    Y_hat = H_{seq_len} @ W_out + b_out
    Y_hat in R^{batch x num_nodes x pred_len}

The model learns both spatial correlations (graph conv) and temporal
dynamics (GRU recurrence) end-to-end.
"""

import torch
import torch.nn as nn

from .tgcn_cell import TGCNCell


class TGCN(nn.Module):
    """
    Temporal Graph Convolutional Network (T-GCN).

    Parameters
    ----------
    num_nodes   : int — number of sensor nodes
    in_features : int — input features per node per timestep (1 for speed)
    hidden_dim  : int — T-GCN hidden state dimension
    pred_len    : int — number of future timesteps to predict
    gcn_layers  : int — GCN depth inside each gate of TGCNCell
    """

    def __init__(self, num_nodes: int, in_features: int = 1,
                 hidden_dim: int = 64, pred_len: int = 12,
                 gcn_layers: int = 1):
        super().__init__()

        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim
        self.pred_len   = pred_len

        # Core recurrent T-GCN cell (processes one timestep at a time,
        # but fully vectorized over the batch dimension)
        self.tgcn_cell = TGCNCell(
            num_nodes   = num_nodes,
            in_features = in_features,
            hidden_dim  = hidden_dim,
            gcn_layers  = gcn_layers,
        )

        # Output projection: hidden_dim -> pred_len  (applied per node)
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass over a sequence of graph snapshots.

        Parameters
        ----------
        x        : Tensor [batch, seq_len, num_nodes, in_features]
                   Normalised speed values for all sensors over the
                   historical window.
        adj_norm : Tensor [num_nodes, num_nodes]
                   Symmetrically normalised adjacency matrix (shared
                   across all batches and timesteps).

        Returns
        -------
        out : Tensor [batch, num_nodes, pred_len]
              Predicted (normalised) speed values for each sensor over
              the future prediction horizon.
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        device = x.device

        # Initialise hidden state to zeros for the entire batch at once
        # Shape: [batch, num_nodes, hidden_dim]
        h = self.tgcn_cell.init_hidden(batch_size, device)

        # --- Recurrence over time (loop only over T timesteps, not batch) ---
        for t in range(seq_len):
            # x_t: [batch, num_nodes, in_features]
            x_t = x[:, t, :, :]

            # h: [batch, num_nodes, hidden_dim]
            h = self.tgcn_cell(x_t, h, adj_norm)

        # --- Output projection ---
        # h: [batch, num_nodes, hidden_dim]
        # output_layer maps hidden_dim -> pred_len independently per node
        out = self.output_layer(h)   # [batch, num_nodes, pred_len]

        return out


# ---------------------------------------------------------------------------
# Utility: count trainable parameters
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
