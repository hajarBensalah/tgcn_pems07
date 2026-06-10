"""
utils/graph_utils.py
--------------------
Build normalised adjacency matrices for PEMS07.

Two graph types are supported:
  - Distance graph (level 1): Gaussian kernel on physical sensor distances
  - Correlation graph (level 2): Pearson correlation of traffic speeds

Pipeline (both types)
---------------------
1. Build raw affinity matrix W  [N × N]
2. Sparsification via threshold
3. Add self-loops:           Ã = W + I
4. Symmetric normalisation:  A_norm = D̃^{−1/2} · Ã · D̃^{−1/2}
5. Convert to torch.Tensor on the requested device
"""

import numpy as np
import pandas as pd
import torch

from .data_loader import load_pems07, _interpolate_missing


def _symmetric_normalize(W: np.ndarray) -> np.ndarray:
    """Symmetrically normalise an affinity matrix with self-loops."""
    num_nodes = W.shape[0]
    A_hat = W + np.eye(num_nodes, dtype=np.float32)

    degree = A_hat.sum(axis=1)
    d_inv_sq = np.power(degree, -0.5)
    d_inv_sq[np.isinf(d_inv_sq)] = 0.0
    D_inv_sq = np.diag(d_inv_sq)

    return D_inv_sq @ A_hat @ D_inv_sq


def _to_tensor(A_norm: np.ndarray, device: torch.device) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    return torch.tensor(A_norm, dtype=torch.float32, device=device)


def build_distance_W(csv_path: str,
                     num_nodes: int,
                     sigma_sq: float = None,
                     threshold: float = 0.1) -> np.ndarray:
    """
    Build the raw distance-based affinity matrix (before normalisation).

    Parameters
    ----------
    csv_path  : path to PEMS07.csv  (columns: from, to, cost)
    num_nodes : number of sensor nodes
    sigma_sq  : σ² for Gaussian kernel (auto-calibrated if None)
    threshold : sparsification cut-off on W_ij

    Returns
    -------
    W : np.ndarray [num_nodes, num_nodes] — raw affinity weights
    """
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    if len(cols) != 3:
        raise ValueError(
            f"Expected 3-column CSV (from, to, cost), got columns: {cols}"
        )
    src_col, dst_col, w_col = cols[0], cols[1], cols[2]

    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in df.iterrows():
        i = int(row[src_col])
        j = int(row[dst_col])
        d = float(row[w_col])

        if i >= num_nodes:
            i -= 1
        if j >= num_nodes:
            j -= 1

        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    observed_dists = dist_matrix[dist_matrix > 0]
    if sigma_sq is None:
        sigma_sq = float(np.var(observed_dists))
        if sigma_sq == 0:
            sigma_sq = 1.0

    print(f"[graph_utils] Distance stats: "
          f"min={observed_dists.min():.3f}, max={observed_dists.max():.3f}, "
          f"mean={observed_dists.mean():.3f}, sigma_sq={sigma_sq:.4f}")

    np.fill_diagonal(dist_matrix, 0.0)

    W = np.zeros_like(dist_matrix)
    mask = dist_matrix > 0
    W[mask] = np.exp(-np.square(dist_matrix[mask]) / sigma_sq)
    W[W < threshold] = 0.0

    n_edges = int((W > 0).sum())
    print(f"[graph_utils] Distance edges after threshold={threshold}: "
          f"{n_edges} (density={n_edges / num_nodes**2:.4f})")

    return W


def build_corr_W(npz_path: str,
                 num_nodes: int,
                 train_ratio: float = 0.60,
                 val_ratio: float = 0.20,
                 threshold: float = 0.5) -> np.ndarray:
    """
    Build a raw correlation-based affinity matrix from train-split speeds.

    W_ij = |corr(speed_i, speed_j)| for pairs above the threshold.

    Parameters
    ----------
    npz_path    : path to PEMS07.npz
    num_nodes   : number of sensor nodes
    train_ratio : fraction of timesteps used to fit correlations
    val_ratio   : unused here, kept for API symmetry with load_pems07
    threshold   : minimum |correlation| to keep an edge

    Returns
    -------
    W : np.ndarray [num_nodes, num_nodes] — raw affinity weights
    """
    speed, train_end, _ = load_pems07(npz_path, train_ratio, val_ratio)
    speed = _interpolate_missing(speed)
    train_speed = speed[:train_end, :]

    corr = np.corrcoef(train_speed.T)   # [N, N]
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    W = np.abs(corr).astype(np.float32)
    W[W < threshold] = 0.0

    n_edges = int((W > 0).sum())
    print(f"[graph_utils] Correlation stats: "
          f"mean={np.abs(corr[corr != 0]).mean():.3f}, "
          f"max={np.abs(corr).max():.3f}")
    print(f"[graph_utils] Correlation edges after threshold={threshold}: "
          f"{n_edges} (density={n_edges / num_nodes**2:.4f})")

    return W


def normalize_adj(W: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """Normalise a raw affinity matrix and return a torch.Tensor."""
    A_norm = _symmetric_normalize(W.astype(np.float32))
    adj_tensor = _to_tensor(A_norm, device)
    print(f"[graph_utils] Adjacency matrix built: shape={adj_tensor.shape}, "
          f"density={float((adj_tensor > 0).sum()) / adj_tensor.numel():.4f}")
    return adj_tensor


def build_adj_matrix(csv_path: str,
                     num_nodes: int,
                     sigma_sq: float = None,
                     threshold: float = 0.1,
                     device: torch.device = None) -> torch.Tensor:
    """
    Build the symmetrically-normalised distance adjacency matrix.

    Backward-compatible wrapper around build_distance_W + normalize_adj.
    """
    W = build_distance_W(csv_path, num_nodes, sigma_sq, threshold)
    return normalize_adj(W, device)


def build_corr_adj_matrix(npz_path: str,
                          num_nodes: int,
                          train_ratio: float = 0.60,
                          val_ratio: float = 0.20,
                          threshold: float = 0.5,
                          device: torch.device = None) -> torch.Tensor:
    """
    Build the symmetrically-normalised correlation adjacency matrix.

    Correlations are computed on the training split only to avoid leakage.
    """
    W = build_corr_W(npz_path, num_nodes, train_ratio, val_ratio, threshold)
    return normalize_adj(W, device)


def build_adj(graph_type: str,
              num_nodes: int,
              device: torch.device = None,
              csv_path: str = None,
              npz_path: str = None,
              train_ratio: float = 0.60,
              val_ratio: float = 0.20,
              sigma_sq: float = None,
              distance_threshold: float = 0.1,
              correlation_threshold: float = 0.5) -> torch.Tensor:
    """
    Unified entry point for distance or correlation adjacency matrices.
    """
    graph_type = graph_type.lower()
    if graph_type == "distance":
        if csv_path is None:
            raise ValueError("csv_path is required for distance graph")
        return build_adj_matrix(
            csv_path=csv_path,
            num_nodes=num_nodes,
            sigma_sq=sigma_sq,
            threshold=distance_threshold,
            device=device,
        )
    if graph_type == "correlation":
        if npz_path is None:
            raise ValueError("npz_path is required for correlation graph")
        return build_corr_adj_matrix(
            npz_path=npz_path,
            num_nodes=num_nodes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            threshold=correlation_threshold,
            device=device,
        )
    raise ValueError(f"Unknown graph_type '{graph_type}'. Use 'distance' or 'correlation'.")


def ablate_edge(W: np.ndarray, i: int, j: int) -> torch.Tensor:
    """
    Remove edge (i, j) from raw affinity W and return re-normalised adjacency.
    """
    W_ablated = W.copy()
    W_ablated[i, j] = 0.0
    W_ablated[j, i] = 0.0
    return normalize_adj(W_ablated)


def get_neighbors(W: np.ndarray, node: int, top_k: int = None) -> np.ndarray:
    """
    Return neighbor indices for a node, sorted by affinity (descending).

    Parameters
    ----------
    W     : raw affinity matrix [N, N]
    node  : target sensor index
    top_k : if set, return only the top-k neighbors

    Returns
    -------
    np.ndarray of neighbor indices
    """
    weights = W[node].copy()
    weights[node] = 0.0
    neighbors = np.where(weights > 0)[0]
    if len(neighbors) == 0:
        return neighbors

    order = neighbors[np.argsort(weights[neighbors])[::-1]]
    if top_k is not None:
        order = order[:top_k]
    return order


def compare_graphs(W_dist: np.ndarray,
                     W_corr: np.ndarray,
                     top_k: int = 20) -> dict:
    """
    Compare distance and correlation graphs (edge overlap, per-node Jaccard).

    Returns summary statistics useful for reports and evaluate.py.
    """
    dist_edges = set(zip(*np.where(W_dist > 0)))
    corr_edges = set(zip(*np.where(W_corr > 0)))

    # Treat (i,j) and (j,i) as the same undirected edge
    def _undirected(edges):
        return {(min(i, j), max(i, j)) for i, j in edges if i != j}

    dist_u = _undirected(dist_edges)
    corr_u = _undirected(corr_edges)
    overlap = dist_u & corr_u
    union = dist_u | corr_u

    num_nodes = W_dist.shape[0]
    per_node_jaccard = []
    for node in range(num_nodes):
        d_nb = set(get_neighbors(W_dist, node, top_k=top_k))
        c_nb = set(get_neighbors(W_corr, node, top_k=top_k))
        if not d_nb and not c_nb:
            per_node_jaccard.append(1.0)
        else:
            per_node_jaccard.append(
                len(d_nb & c_nb) / max(len(d_nb | c_nb), 1)
            )

    return {
        "distance_edges": len(dist_u),
        "correlation_edges": len(corr_u),
        "shared_edges": len(overlap),
        "jaccard_global": len(overlap) / max(len(union), 1),
        "mean_neighbor_jaccard": float(np.mean(per_node_jaccard)),
    }
