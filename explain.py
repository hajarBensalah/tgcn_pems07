"""
explain.py
----------
XAI and graph analysis for T-GCN on PEMS07.

Generates:
1. Distance vs correlation graph comparison (heatmaps + edge overlap stats)
2. Neighbor ablation study — which neighbors most influence each prediction
3. Dynamic correlation snapshots — how traffic links change across time windows

Usage
-----
    python explain.py
    python explain.py --checkpoint outputs/checkpoints/best_model.pt
    python explain.py --sensors 5 10 50
"""

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from models import TGCN
from utils import (build_dataset, build_distance_W, build_corr_W,
                   ablate_edge, get_neighbors, compare_graphs)
from evaluate import load_checkpoint


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_graph_comparison(W_dist: np.ndarray,
                          W_corr: np.ndarray,
                          sensor_id: int,
                          top_k: int,
                          plot_dir: str):
    """Side-by-side bar charts of neighbor weights for one sensor."""
    dist_nb = get_neighbors(W_dist, sensor_id, top_k=top_k)
    corr_nb = get_neighbors(W_corr, sensor_id, top_k=top_k)
    all_nb = sorted(set(dist_nb.tolist()) | set(corr_nb.tolist()))

    dist_vals = [W_dist[sensor_id, n] for n in all_nb]
    corr_vals = [W_corr[sensor_id, n] for n in all_nb]
    labels = [str(n) for n in all_nb]

    x = np.arange(len(all_nb))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(all_nb) * 0.5), 5))
    ax.bar(x - width / 2, dist_vals, width, label="Distance graph", color="steelblue")
    ax.bar(x + width / 2, corr_vals, width, label="Correlation graph", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Neighbor sensor ID")
    ax.set_ylabel("Affinity weight")
    ax.set_title(f"Sensor {sensor_id} — Distance vs Correlation Neighbors")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, f"graph_compare_sensor_{sensor_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain] Graph comparison saved → {out_path}")


def plot_neighbor_heatmap(W: np.ndarray,
                          sensor_ids: list,
                          top_k: int,
                          title: str,
                          out_name: str,
                          plot_dir: str):
    """Heatmap of top neighbor affinities for selected sensors."""
    rows = []
    labels = []
    for sid in sensor_ids:
        nb = get_neighbors(W, sid, top_k=top_k)
        row = np.zeros(top_k)
        for k, n in enumerate(nb):
            row[k] = W[sid, n]
            labels.append(f"S{sid}→{n}") if len(labels) < top_k * len(sensor_ids) else None
        rows.append(row)

    row_labels = [f"Sensor {s}" for s in sensor_ids]
    col_labels = [f"#{i+1}" for i in range(top_k)]

    fig, ax = plt.subplots(figsize=(10, max(3, len(sensor_ids) * 1.2)))
    sns.heatmap(np.array(rows), annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=col_labels, yticklabels=row_labels, ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[explain] Heatmap saved → {out_path}")


def plot_ablation_results(results: dict, plot_dir: str):
    """Bar chart of neighbor influence (MAE increase when ablated)."""
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)

    for ax, (sensor_id, data) in zip(axes[0], results.items()):
        neighbors = data["neighbors"]
        impacts   = data["impacts"]
        baseline  = data["baseline_mae"]

        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(neighbors)))
        ax.barh([str(n) for n in neighbors], impacts, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("MAE increase (km/h) when neighbor ablated")
        ax.set_ylabel("Neighbor sensor")
        ax.set_title(f"Sensor {sensor_id}\n(baseline MAE = {baseline:.3f} km/h)")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    plt.suptitle("XAI — Neighbor Ablation (which neighbors matter most?)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, "neighbor_ablation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain] Ablation plot saved → {out_path}")


def plot_dynamic_correlation(speed: np.ndarray,
                             sensor_a: int,
                             sensor_b: int,
                             window: int,
                             plot_dir: str):
    """
    Show how rolling correlation between two sensors evolves over time.
    Illustrates dynamic traffic propagation (links change with conditions).
    """
    T = speed.shape[0]
    rolling_corr = []
    for t in range(window, T):
        seg_a = speed[t - window:t, sensor_a]
        seg_b = speed[t - window:t, sensor_b]
        if np.std(seg_a) < 1e-6 or np.std(seg_b) < 1e-6:
            rolling_corr.append(0.0)
        else:
            rolling_corr.append(float(np.corrcoef(seg_a, seg_b)[0, 1]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(speed[:, sensor_a], label=f"Sensor {sensor_a}", alpha=0.8)
    axes[0].plot(speed[:, sensor_b], label=f"Sensor {sensor_b}", alpha=0.8)
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title(f"Sensors {sensor_a} & {sensor_b} — Speed over time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(window, T), rolling_corr, color="purple", linewidth=1.2)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Timestep (5-min intervals)")
    axes[1].set_ylabel(f"Rolling correlation (window={window})")
    axes[1].set_title("Dynamic traffic link — correlation changes with congestion patterns")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"dynamic_corr_{sensor_a}_{sensor_b}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[explain] Dynamic correlation saved → {out_path}")


# ---------------------------------------------------------------------------
# XAI — neighbor ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_neighbor_ablation(model: TGCN,
                          loader,
                          adj_norm: torch.Tensor,
                          W_raw: np.ndarray,
                          scaler: dict,
                          device: torch.device,
                          sensor_id: int,
                          max_samples: int,
                          top_k: int) -> dict:
    """
    For a target sensor, ablate each neighbor and measure MAE increase.

    A large MAE increase means the model relies heavily on that neighbor
    for predicting the target sensor's speed.
    """
    neighbors = get_neighbors(W_raw, sensor_id, top_k=top_k)
    if len(neighbors) == 0:
        print(f"[explain] Sensor {sensor_id}: no neighbors in graph, skipping.")
        return None

    # Collect baseline predictions on a subset of test data
    preds_list, trues_list = [], []
    n_seen = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_pred  = model(X_batch, adj_norm)
        preds_list.append(y_pred.cpu().numpy())
        trues_list.append(y_batch.numpy())
        n_seen += X_batch.shape[0]
        if n_seen >= max_samples:
            break

    preds = np.concatenate(preds_list, axis=0)[:max_samples]
    trues = np.concatenate(trues_list, axis=0)[:max_samples]

    mu  = scaler["mean"]
    std = scaler["std"]
    preds_orig = preds * std[None, :, None] + mu[None, :, None]
    trues_orig = trues * std[None, :, None] + mu[None, :, None]

    baseline_mae = float(np.mean(np.abs(
        preds_orig[:, sensor_id, :] - trues_orig[:, sensor_id, :]
    )))

    impacts = []
    for nb in neighbors:
        adj_ablated = ablate_edge(W_raw, sensor_id, int(nb)).to(device)
        preds_abl, trues_abl = [], []

        n_seen = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred  = model(X_batch, adj_ablated)
            preds_abl.append(y_pred.cpu().numpy())
            trues_abl.append(y_batch.numpy())
            n_seen += X_batch.shape[0]
            if n_seen >= max_samples:
                break

        p = np.concatenate(preds_abl, axis=0)[:max_samples]
        t = np.concatenate(trues_abl, axis=0)[:max_samples]
        p_orig = p * std[None, :, None] + mu[None, :, None]
        t_orig = t * std[None, :, None] + mu[None, :, None]

        ablated_mae = float(np.mean(np.abs(
            p_orig[:, sensor_id, :] - t_orig[:, sensor_id, :]
        )))
        impacts.append(ablated_mae - baseline_mae)

    # Sort by impact (most influential first)
    order = np.argsort(impacts)[::-1]
    return {
        "baseline_mae": baseline_mae,
        "neighbors":    neighbors[order].tolist(),
        "impacts":      [impacts[i] for i in order],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ckpt_path: str = None, sensor_ids: list = None):
    if ckpt_path is None:
        ckpt_path = config.BEST_MODEL_PATH
    if sensor_ids is None:
        sensor_ids = config.XAI_SENSORS

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[explain] Using device: {device}")

    xai_dir = os.path.join(config.PLOT_DIR, "xai")
    os.makedirs(xai_dir, exist_ok=True)

    # Load model
    model, scaler, cfg = load_checkpoint(ckpt_path, device)
    num_nodes = cfg["num_nodes"]

    # Load data
    print("[explain] Loading data...")
    data_dict = build_dataset(
        npz_path    = config.NPZ_PATH,
        train_ratio = config.TRAIN_RATIO,
        val_ratio   = config.VAL_RATIO,
        seq_len     = config.SEQ_LEN,
        pred_len    = config.PRED_LEN,
        batch_size  = config.BATCH_SIZE,
    )
    test_loader = data_dict["test_loader"]

    # Build both graphs
    print("[explain] Building distance and correlation graphs...")
    W_dist = build_distance_W(
        csv_path  = config.CSV_PATH,
        num_nodes = num_nodes,
        sigma_sq  = config.SIGMA_SQ,
        threshold = config.DISTANCE_THRESHOLD,
    )
    W_corr = build_corr_W(
        npz_path    = config.NPZ_PATH,
        num_nodes   = num_nodes,
        train_ratio = config.TRAIN_RATIO,
        val_ratio   = config.VAL_RATIO,
        threshold   = config.CORRELATION_THRESHOLD,
    )

    graph_stats = compare_graphs(W_dist, W_corr, top_k=config.XAI_TOP_NEIGHBORS)
    print("\n[explain] Graph comparison:")
    print(f"  Distance edges     : {graph_stats['distance_edges']}")
    print(f"  Correlation edges  : {graph_stats['correlation_edges']}")
    print(f"  Shared edges       : {graph_stats['shared_edges']}")
    print(f"  Global Jaccard     : {graph_stats['jaccard_global']:.3f}")
    print(f"  Mean neighbor Jaccard (top-{config.XAI_TOP_NEIGHBORS}): "
          f"{graph_stats['mean_neighbor_jaccard']:.3f}")

    # Use the graph type the model was trained with
    graph_type = cfg.get("graph_type", "distance")
    W_model = W_dist if graph_type == "distance" else W_corr
    from utils.graph_utils import normalize_adj
    adj_norm = normalize_adj(W_model, device)

    # 1. Graph comparison plots
    print("\n[explain] Generating graph comparison plots...")
    for sid in sensor_ids:
        plot_graph_comparison(
            W_dist, W_corr, sid, config.XAI_TOP_NEIGHBORS, xai_dir
        )

    plot_neighbor_heatmap(
        W_dist, sensor_ids, config.XAI_TOP_NEIGHBORS,
        "Distance Graph — Top Neighbor Affinities", "heatmap_distance.png", xai_dir
    )
    plot_neighbor_heatmap(
        W_corr, sensor_ids, config.XAI_TOP_NEIGHBORS,
        "Correlation Graph — Top Neighbor Affinities", "heatmap_correlation.png", xai_dir
    )

    # 2. Dynamic correlation (traffic propagation over time)
    print("\n[explain] Plotting dynamic correlation (traffic propagation)...")
    from utils.data_loader import load_pems07, _interpolate_missing
    speed, _, _ = load_pems07(config.NPZ_PATH, config.TRAIN_RATIO, config.VAL_RATIO)
    speed = _interpolate_missing(speed)

    for sid in sensor_ids[:2]:
        nb = get_neighbors(W_corr, sid, top_k=1)
        if len(nb) > 0:
            plot_dynamic_correlation(
                speed, sid, int(nb[0]),
                window=config.SEQ_LEN * 6,   # ~1 hour rolling window
                plot_dir=xai_dir,
            )

    # 3. Neighbor ablation XAI
    print("\n[explain] Running neighbor ablation study...")
    ablation_results = {}
    for sid in sensor_ids:
        print(f"  Ablating neighbors of sensor {sid}...")
        result = run_neighbor_ablation(
            model=model,
            loader=test_loader,
            adj_norm=adj_norm,
            W_raw=W_model,
            scaler=scaler,
            device=device,
            sensor_id=sid,
            max_samples=config.XAI_MAX_SAMPLES,
            top_k=config.XAI_TOP_NEIGHBORS,
        )
        if result is not None:
            ablation_results[sid] = result
            top3 = list(zip(result["neighbors"][:3], result["impacts"][:3]))
            print(f"    Top-3 influential neighbors: {top3}")

    if ablation_results:
        plot_ablation_results(ablation_results, xai_dir)

    print(f"\n[explain] All XAI outputs saved to: {xai_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI analysis for T-GCN")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sensors", type=int, nargs="+", default=None,
                        help="Target sensor IDs for ablation (default: config.XAI_SENSORS)")
    args = parser.parse_args()
    main(args.checkpoint, args.sensors)
