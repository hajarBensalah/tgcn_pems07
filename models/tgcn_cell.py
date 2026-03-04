"""
models/tgcn_cell.py
-------------------
T-GCN Cell: one recurrent step of the Temporal Graph Convolutional Network.

Each GRU gate is replaced by a Graph Convolutional operation so that
spatial dependencies are captured together with temporal dynamics.

Equations (Zhao et al., 2020):
    u_t = sigma( GCN(A, [X_t || H_{t-1}], W_u) )             update gate
    r_t = sigma( GCN(A, [X_t || H_{t-1}], W_r) )             reset gate
    c_t = tanh(  GCN(A, [X_t || (r_t * H_{t-1})], W_c) )    candidate
    H_t = u_t * H_{t-1} + (1 - u_t) * c_t                    new hidden

Where:
    [· || ·]  denotes feature concatenation along the feature axis
    *         element-wise multiplication
    GCN       = MultiLayerGCN (see gcn.py)

Batch dimension is fully vectorized — no Python loops over the batch.
"""

import torch
import torch.nn as nn

from .gcn import MultiLayerGCN


class TGCNCell(nn.Module):
    """
    Single T-GCN recurrent cell — fully vectorized over the batch dimension.

    Parameters
    ----------
    num_nodes   : int — number of graph nodes (sensors)
    in_features : int — input feature dimension per node (1 for speed)
    hidden_dim  : int — hidden state dimension per node
    gcn_layers  : int — number of GCN layers used inside each gate
    """

    def __init__(self, num_nodes: int, in_features: int,
                 hidden_dim: int, gcn_layers: int = 1):
        super().__init__()

        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim

        # Concatenation of [X_t || H_{t-1}] has width in_features + hidden_dim
        concat_dim = in_features + hidden_dim

        # --- Update gate (u): sigmoid applied externally by TGCNCell ---
        self.gcn_u = MultiLayerGCN(concat_dim, hidden_dim, num_layers=gcn_layers)

        # --- Reset gate (r): sigmoid applied externally ---
        self.gcn_r = MultiLayerGCN(concat_dim, hidden_dim, num_layers=gcn_layers)

        # --- Candidate hidden state (c): tanh applied externally ---
        # Takes [X_t || r_t * H_{t-1}], same concat_dim
        self.gcn_c = MultiLayerGCN(concat_dim, hidden_dim, num_layers=gcn_layers)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor,
                adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute one recurrent step for a full batch.

        Parameters
        ----------
        x_t      : Tensor [batch, num_nodes, in_features]
                   Node features at the current timestep.
        h_prev   : Tensor [batch, num_nodes, hidden_dim]
                   Hidden state from the previous timestep.
                   On the very first step pass zeros (see init_hidden).
        adj_norm : Tensor [num_nodes, num_nodes]
                   Symmetrically normalised adjacency matrix.
                   Shared across all samples in the batch.

        Returns
        -------
        h_t : Tensor [batch, num_nodes, hidden_dim]
              Updated hidden state for this timestep.
        """
        # ---- Concatenate input and previous hidden state ---------------
        # xh: [B, N, in_features + hidden_dim]
        xh = torch.cat([x_t, h_prev], dim=-1)

        # ---- Update gate -----------------------------------------------
        u_t = torch.sigmoid(self.gcn_u(xh, adj_norm))   # [B, N, hidden]

        # ---- Reset gate ------------------------------------------------
        r_t = torch.sigmoid(self.gcn_r(xh, adj_norm))   # [B, N, hidden]

        # ---- Candidate hidden state ------------------------------------
        # Reset gate scales down irrelevant history: r_t * h_prev
        xrh = torch.cat([x_t, r_t * h_prev], dim=-1)    # [B, N, in+hidden]
        c_t = torch.tanh(self.gcn_c(xrh, adj_norm))     # [B, N, hidden]

        # ---- New hidden state (GRU-style interpolation) ----------------
        h_t = u_t * h_prev + (1.0 - u_t) * c_t          # [B, N, hidden]

        return h_t

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Return an all-zeros initial hidden state for a batch.

        Returns
        -------
        Tensor [batch_size, num_nodes, hidden_dim] of zeros
        """
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim,
                           device=device)
