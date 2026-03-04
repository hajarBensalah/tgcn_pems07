"""
utils/graph_utils.py
--------------------
Build a normalised adjacency matrix for PEMS07 from the CSV distance file.

PEMS07.csv format: 3 columns  →  from | to | cost
  - 866 directed edges (0-based sensor indices)
  - 'cost' represents the sensor-to-sensor distance (in km)

Pipeline
--------
1. Read edge-list CSV  →  sparse distance matrix  D  [N × N]
2. Gaussian kernel (auto-calibrated σ²):
       W_ij = exp(−D_ij² / σ²)
   σ² is set to the variance of all observed distances unless overridden.
3. Sparsification: set W_ij = 0 when W_ij < threshold
4. Add self-loops:           Ã = W + I
5. Degree matrix:            D̃_ii = Σ_j Ã_ij
6. Symmetric normalisation:  A_norm = D̃^{−1/2} · Ã · D̃^{−1/2}
7. Convert to torch.Tensor on the requested device
"""

import numpy  as np
import pandas as pd
import torch


def build_adj_matrix(csv_path: str,
                     num_nodes: int,
                     sigma_sq : float = None,
                     threshold: float = 0.1,
                     device   : torch.device = None) -> torch.Tensor:
    """
    Load PEMS07.csv and build the symmetrically-normalised adjacency matrix.

    Parameters
    ----------
    csv_path  : str   — path to PEMS07.csv  (columns: from, to, cost)
    num_nodes : int   — number of sensor nodes (883)
    sigma_sq  : float — σ² for Gaussian kernel.
                        If None (default), computed as the variance of all
                        pairwise distances so the kernel is self-calibrating.
    threshold : float — sparsification cut-off; edges with W_ij < threshold
                        are set to 0.  Default 0.1 keeps ~60% of edges.
    device    : torch.device or None

    Returns
    -------
    adj_norm : torch.Tensor [num_nodes, num_nodes]
               Symmetrically-normalised adjacency matrix ready for GCN.
    """
    if device is None:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)   # columns: 'from', 'to', 'cost'

    # Detect column names flexibly (some versions use different names)
    cols = df.columns.tolist()
    if len(cols) == 3:
        src_col, dst_col, w_col = cols[0], cols[1], cols[2]
    else:
        raise ValueError(
            f"Expected 3-column CSV (from, to, cost), got columns: {cols}"
        )

    # ------------------------------------------------------------------
    # 2. Build N×N distance matrix from edge list
    # ------------------------------------------------------------------
    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for _, row in df.iterrows():
        i = int(row[src_col])
        j = int(row[dst_col])
        d = float(row[w_col])

        # Guard against 1-based indexing
        if i >= num_nodes:
            i -= 1
        if j >= num_nodes:
            j -= 1

        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d   # make symmetric

    # ------------------------------------------------------------------
    # 3. Auto-calibrate σ² from the observed distance distribution
    # ------------------------------------------------------------------
    observed_dists = dist_matrix[dist_matrix > 0]   # non-zero entries

    if sigma_sq is None:
        # Use variance of observed distances as σ² (standard practice)
        sigma_sq = float(np.var(observed_dists))
        if sigma_sq == 0:
            sigma_sq = 1.0

    print(f"[graph_utils] Distance stats: "
          f"min={observed_dists.min():.3f}, max={observed_dists.max():.3f}, "
          f"mean={observed_dists.mean():.3f}, sigma_sq={sigma_sq:.4f}")

    # ------------------------------------------------------------------
    # 4. Gaussian affinity kernel:  W_ij = exp(−D_ij² / σ²)
    #    Self-entries (diagonal) are left at 0 before self-loop addition.
    # ------------------------------------------------------------------
    np.fill_diagonal(dist_matrix, 0.0)

    W = np.zeros_like(dist_matrix)
    mask = dist_matrix > 0                          # only actual edges
    W[mask] = np.exp(-np.square(dist_matrix[mask]) / sigma_sq)

    # ------------------------------------------------------------------
    # 5. Sparsification: remove weak edges
    # ------------------------------------------------------------------
    W[W < threshold] = 0.0

    n_edges = int((W > 0).sum())
    print(f"[graph_utils] Edges after threshold={threshold}: {n_edges} "
          f"(density={n_edges / num_nodes**2:.4f})")

    # ------------------------------------------------------------------
    # 6. Add self-loops: Ã = W + I
    # ------------------------------------------------------------------
    A_hat = W + np.eye(num_nodes, dtype=np.float32)

    # ------------------------------------------------------------------
    # 7. Symmetric normalisation: D̃^{-1/2} · Ã · D̃^{-1/2}
    # ------------------------------------------------------------------
    degree   = A_hat.sum(axis=1)                    # [N]
    d_inv_sq = np.power(degree, -0.5)
    d_inv_sq[np.isinf(d_inv_sq)] = 0.0             # isolated nodes → 0
    D_inv_sq = np.diag(d_inv_sq)                   # [N, N]

    A_norm = D_inv_sq @ A_hat @ D_inv_sq           # [N, N]

    # ------------------------------------------------------------------
    # 8. Convert to torch Tensor
    # ------------------------------------------------------------------
    adj_tensor = torch.tensor(A_norm, dtype=torch.float32, device=device)

    print(f"[graph_utils] Adjacency matrix built: shape={adj_tensor.shape}, "
          f"overall density={float((adj_tensor > 0).sum()) / adj_tensor.numel():.4f}")

    return adj_tensor
