"""
models/gcn.py
-------------
Spectral Graph Convolutional Network (GCN) layer as used inside T-GCN.

Reference:
    Kipf & Welling (2017) — Semi-Supervised Classification with GCNs
    Zhao et al. (2020)    — T-GCN: A Temporal Graph Convolutional Network
                            for Traffic Prediction

Formula:
    H = sigma( D^{-1/2} * A_hat * D^{-1/2} * X * W )

Where:
    A_hat = A + I       (adjacency matrix with self-loops added)
    D is the degree matrix of A_hat
    X  -- input node-feature matrix
    W  -- learnable weight matrix  [in_features, out_features]
    sigma -- activation function (ReLU by default)

Vectorized for batched input:
    x can be  [num_nodes, in_features]       (single graph)
           or [batch, num_nodes, in_features] (batch of graphs sharing adj)
"""

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """
    A single spectral GCN layer supporting both single and batched inputs.

    Parameters
    ----------
    in_features  : int  — dimension of input node features
    out_features : int  — dimension of output node features
    bias         : bool — whether to add a learnable bias term
    activation   : callable or None
                   Activation applied after the graph convolution.
                   Pass None to get a linear (no-activation) layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, activation=torch.relu):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation

        # Learnable weight matrix W  [in_features, out_features]
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features)
        )

        if bias:
            # Learnable bias vector [out_features]
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Xavier uniform init for weights; zero init for bias.
        Standard choices that keep gradient magnitudes stable in deep GCNs.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting both unbatched and batched inputs.

        Parameters
        ----------
        x        : Tensor
                   [num_nodes, in_features]         — single graph, or
                   [batch, num_nodes, in_features]  — batch of graphs
        adj_norm : Tensor [num_nodes, num_nodes]
                   Pre-computed symmetrically-normalised adjacency matrix
                   D^{-1/2} * A_hat * D^{-1/2}.
                   Shared across all batch samples.

        Returns
        -------
        Tensor [num_nodes, out_features] or [batch, num_nodes, out_features]
        """
        # Step 1: Linear transform of node features
        #   x @ W  -->  [..., N, out_features]
        #   torch.matmul broadcasts over leading batch dim automatically
        support = torch.matmul(x, self.weight)

        # Step 2: Graph diffusion  adj_norm * support
        #   For batched input [B, N, F]:
        #     adj_norm is [N, N], support is [B, N, F]
        #     We want output[b] = adj_norm @ support[b]
        #     Equivalent to: einsum('nm, bnh -> bmh', adj_norm, support)
        if x.dim() == 2:
            # Unbatched: [N, F]
            output = torch.mm(adj_norm, support)
        else:
            # Batched: [B, N, F]  — use einsum for clarity and efficiency
            output = torch.einsum("nm,bnh->bmh", adj_norm, support)

        # Step 3: Optional bias (broadcasts over batch and node dims)
        if self.bias is not None:
            output = output + self.bias

        # Step 4: Non-linear activation
        if self.activation is not None:
            output = self.activation(output)

        return output


class MultiLayerGCN(nn.Module):
    """
    Stack of GCNLayer modules to form a deeper GCN encoder.

    Used inside each gate of the T-GCN cell.  The last layer has no
    activation so the gate sigmoid/tanh can be applied externally.

    Parameters
    ----------
    in_features  : int — input feature dimension
    out_features : int — output feature dimension for every layer
    num_layers   : int — number of stacked GCN layers (>= 1)
    """

    def __init__(self, in_features: int, out_features: int, num_layers: int = 1):
        super().__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        layers = []
        for i in range(num_layers):
            layer_in = in_features if i == 0 else out_features
            # Intermediate layers use ReLU; final layer has no activation
            act = torch.relu if i < num_layers - 1 else None
            layers.append(GCNLayer(layer_in, out_features, activation=act))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : Tensor [N, F] or [B, N, F]
        adj_norm : Tensor [N, N]

        Returns
        -------
        Tensor [N, out_features] or [B, N, out_features]
        """
        h = x
        for layer in self.layers:
            h = layer(h, adj_norm)
        return h
