"""
evaluate.py
-----------
Load the best T-GCN checkpoint and generate:

1. Training / validation loss curves  → outputs/plots/loss_curve.png
2. Predicted vs Ground Truth for sensors 5, 10, 50 → outputs/plots/pred_vs_true.png
3. Confusion matrix heatmap           → outputs/plots/confusion_matrix.png
4. Spatial MAE error map (883 sensors) → outputs/plots/spatial_error.png
5. Metrics summary table printed in terminal

Usage
-----
    python evaluate.py
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pt
"""

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / Windows
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import config
from models         import TGCN
from utils          import (build_dataset, build_adj, build_distance_W,
                            build_corr_W, compare_graphs, get_neighbors)
from utils.metrics  import (compute_regression_metrics,
                             compute_classification_metrics,
                             print_metrics_table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model state from a .pt checkpoint file."""
    # weights_only=False: checkpoint includes numpy scaler arrays (PyTorch 2.6+ default is True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    model = TGCN(
        num_nodes   = cfg["num_nodes"],
        in_features = 1,
        hidden_dim  = cfg["hidden_dim"],
        pred_len    = cfg["pred_len"],
        gcn_layers  = cfg["gcn_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[evaluate] Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f}, val_MAE={ckpt['val_mae']:.4f})")

    return model, ckpt["scaler"], cfg


@torch.no_grad()
def run_inference(model: TGCN,
                  loader,
                  adj_norm: torch.Tensor,
                  device: torch.device,
                  scaler: dict):
    """
    Run full inference on a DataLoader split.

    Returns
    -------
    preds_orig : np.ndarray [num_samples, num_nodes, pred_len] — km/h
    trues_orig : np.ndarray [num_samples, num_nodes, pred_len] — km/h
    """
    preds_list, trues_list = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_pred  = model(X_batch, adj_norm)
        preds_list.append(y_pred.cpu().numpy())
        trues_list.append(y_batch.numpy())

    preds = np.concatenate(preds_list, axis=0)   # [S, N, pred]
    trues = np.concatenate(trues_list, axis=0)

    mu  = scaler["mean"]   # [N]
    std = scaler["std"]    # [N]

    preds_orig = preds * std[None, :, None] + mu[None, :, None]
    trues_orig = trues * std[None, :, None] + mu[None, :, None]

    return preds_orig, trues_orig


# ---------------------------------------------------------------------------
# Plot 1 — Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(output_dir: str):
    """
    Re-plot the loss curves from the saved .npy files produced by train.py.
    If the files do not exist, skip gracefully.
    """
    train_path = os.path.join(output_dir, "train_losses.npy")
    val_path   = os.path.join(output_dir, "val_losses.npy")

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("[evaluate] Loss curve data not found — skipping plot.")
        return

    train_losses = np.load(train_path)
    val_losses   = np.load(val_path)
    epochs       = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train Loss (MAE)", linewidth=2,
            color="steelblue")
    ax.plot(epochs, val_losses, label="Val Loss (MAE)", linewidth=2,
            linestyle="--", color="darkorange")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MAE Loss (normalised units)", fontsize=12)
    ax.set_title("T-GCN — Training & Validation Loss (PEMS07)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(config.PLOT_DIR, "loss_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Loss curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Predicted vs Ground Truth for selected sensors
# ---------------------------------------------------------------------------

def plot_pred_vs_true(preds_orig: np.ndarray,
                      trues_orig: np.ndarray,
                      sensor_ids: list = None):
    """
    Plot predicted vs actual speed for a few sensors across all test samples.

    We flatten the prediction horizon: each sample has pred_len future steps,
    so we plot them in sequence.  The x-axis represents ordered 5-minute
    intervals from the first test window onwards.

    Parameters
    ----------
    preds_orig : [S, N, pred_len]
    trues_orig : [S, N, pred_len]
    sensor_ids : list of 0-based sensor indices to plot
    """
    if sensor_ids is None:
        sensor_ids = config.VIZ_SENSORS

    S, N, P = preds_orig.shape

    # Flatten samples × pred_len to form a continuous time series
    # Take every P-th sample to avoid overlapping windows in the plot
    stride = max(1, P)
    idx    = np.arange(0, S, stride)

    fig, axes = plt.subplots(len(sensor_ids), 1,
                             figsize=(14, 4 * len(sensor_ids)),
                             sharex=False)

    if len(sensor_ids) == 1:
        axes = [axes]

    for ax, sid in zip(axes, sensor_ids):
        true_ts = trues_orig[idx, sid, :].ravel()   # flatten time
        pred_ts = preds_orig[idx, sid, :].ravel()
        x       = np.arange(len(true_ts))

        ax.plot(x, true_ts, label="Ground Truth", linewidth=1.5,
                color="steelblue", alpha=0.8)
        ax.plot(x, pred_ts, label="Prediction",  linewidth=1.5,
                color="darkorange", alpha=0.8, linestyle="--")
        ax.set_title(f"Sensor {sid} — Speed Prediction vs Ground Truth",
                     fontsize=12)
        ax.set_xlabel("Time (5-min intervals)")
        ax.set_ylabel("Speed (km/h)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("T-GCN Prediction vs Ground Truth (PEMS07 Test Set)",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(config.PLOT_DIR, "pred_vs_true.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Pred-vs-True plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(clf_results: dict,
                           class_names=("Congested", "Moderate", "Free Flow")):
    """
    Plot a labelled confusion-matrix heatmap with counts and row-normalised %.

    Parameters
    ----------
    clf_results : output of compute_classification_metrics
    class_names : display names for the three traffic states
    """
    cm      = clf_results["confusion_matrix"]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Annotate with count and percentage
    annot = np.array([
        [f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=annot, fmt="",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                vmin=0, vmax=1,
                ax=ax)

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class",      fontsize=12)
    ax.set_title("T-GCN — Confusion Matrix (Traffic States, PEMS07 Test Set)",
                 fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(config.PLOT_DIR, "confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Spatial MAE error map
# ---------------------------------------------------------------------------

def plot_spatial_error(preds_orig: np.ndarray,
                       trues_orig: np.ndarray):
    """
    Compute per-sensor MAE averaged over all test samples and prediction
    steps, then display as a bar chart sorted by error magnitude.

    A true spatial map would require GPS coordinates which are not provided
    in PEMS07.npz; instead we use a ranked bar chart that conveys which
    sensors are hardest to predict — equivalent information for diagnosis.

    Parameters
    ----------
    preds_orig : [S, N, pred_len]
    trues_orig : [S, N, pred_len]
    """
    # Per-sensor MAE: mean over samples and pred steps
    sensor_mae = np.mean(np.abs(preds_orig - trues_orig), axis=(0, 2))  # [N]

    N          = len(sensor_mae)
    sorted_idx = np.argsort(sensor_mae)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: bar chart of sorted sensor MAEs
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(
        (sensor_mae[sorted_idx] - sensor_mae.min()) /
        (sensor_mae.max() - sensor_mae.min() + 1e-9)
    )
    ax.bar(np.arange(N), sensor_mae[sorted_idx], color=colors, width=1.0,
           edgecolor="none")
    ax.set_xlabel("Sensor (ranked by MAE)", fontsize=11)
    ax.set_ylabel("MAE (km/h)",             fontsize=11)
    ax.set_title("Per-Sensor MAE — Ranked",  fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Right: histogram of sensor MAEs
    ax2 = axes[1]
    ax2.hist(sensor_mae, bins=40, color="steelblue", edgecolor="white",
             alpha=0.8)
    ax2.axvline(sensor_mae.mean(), color="red", linewidth=2,
                label=f"Mean = {sensor_mae.mean():.2f} km/h")
    ax2.axvline(np.median(sensor_mae), color="orange", linewidth=2,
                linestyle="--",
                label=f"Median = {np.median(sensor_mae):.2f} km/h")
    ax2.set_xlabel("MAE (km/h)", fontsize=11)
    ax2.set_ylabel("Number of Sensors", fontsize=11)
    ax2.set_title("Distribution of Per-Sensor MAE", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("T-GCN — Spatial Error Analysis (PEMS07, 883 Sensors)",
                 fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(config.PLOT_DIR, "spatial_error.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Spatial error map saved → {out_path}")


# ---------------------------------------------------------------------------
# Plot 5 — Graph comparison (distance vs correlation)
# ---------------------------------------------------------------------------

def plot_graph_overlap(W_dist: np.ndarray,
                       W_corr: np.ndarray,
                       sensor_ids: list = None):
    """
    Visualise how distance-based and correlation-based neighbors differ
    for selected sensors.
    """
    if sensor_ids is None:
        sensor_ids = config.VIZ_SENSORS

    top_k = config.XAI_TOP_NEIGHBORS
    n = len(sensor_ids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, sid in zip(axes[0], sensor_ids):
        d_nb = set(get_neighbors(W_dist, sid, top_k=top_k).tolist())
        c_nb = set(get_neighbors(W_corr, sid, top_k=top_k).tolist())
        only_dist = d_nb - c_nb
        only_corr = c_nb - d_nb
        shared    = d_nb & c_nb

        categories = ["Shared", "Distance only", "Correlation only"]
        counts = [len(shared), len(only_dist), len(only_corr)]
        colors = ["#2ecc71", "#3498db", "#e67e22"]

        ax.bar(categories, counts, color=colors, edgecolor="white")
        ax.set_title(f"Sensor {sid} — Top-{top_k} neighbors")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    stats = compare_graphs(W_dist, W_corr, top_k=top_k)
    plt.suptitle(
        f"Graph Comparison — Global Jaccard={stats['jaccard_global']:.3f}, "
        f"shared edges={stats['shared_edges']}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(config.PLOT_DIR, "graph_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Graph comparison saved → {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main(ckpt_path: str = None, graph_type: str = None, dataset: str = "pems07"):
    if ckpt_path is None:
        ckpt_path = config.BEST_MODEL_PATH

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run train.py first."
        )

    if dataset == "casablanca":
        npz_path = config.CASABLANCA_NPZ
        csv_path = config.CASABLANCA_CSV
    else:
        npz_path = config.NPZ_PATH
        csv_path = config.CSV_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Using device: {device}")

    os.makedirs(config.PLOT_DIR,       exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model                                                          #
    # ------------------------------------------------------------------ #
    model, scaler, cfg = load_checkpoint(ckpt_path, device)
    num_nodes = cfg["num_nodes"]
    seq_len   = cfg.get("seq_len", config.SEQ_LEN)
    pred_len  = cfg.get("pred_len", config.PRED_LEN)
    if dataset == "casablanca" and "seq_len" not in cfg:
        seq_len  = config.CASABLANCA_SEQ_LEN
        pred_len = config.CASABLANCA_PRED_LEN

    # ------------------------------------------------------------------ #
    # Data                                                                #
    # ------------------------------------------------------------------ #
    print("[evaluate] Loading data...")
    data_dict = build_dataset(
        npz_path    = npz_path,
        train_ratio = config.TRAIN_RATIO,
        val_ratio   = config.VAL_RATIO,
        seq_len     = seq_len,
        pred_len    = pred_len,
        batch_size  = min(config.BATCH_SIZE, 16) if dataset == "casablanca" else config.BATCH_SIZE,
    )
    test_loader = data_dict["test_loader"]

    # ------------------------------------------------------------------ #
    # Adjacency matrix                                                    #
    # ------------------------------------------------------------------ #
    if graph_type is None:
        graph_type = cfg.get("graph_type", config.GRAPH_TYPE)

    print(f"[evaluate] Building adjacency matrix (graph_type={graph_type})...")
    adj_norm = build_adj(
        graph_type            = graph_type,
        num_nodes             = num_nodes,
        device                = device,
        csv_path              = csv_path,
        npz_path              = npz_path,
        train_ratio           = config.TRAIN_RATIO,
        val_ratio             = config.VAL_RATIO,
        sigma_sq              = config.SIGMA_SQ,
        distance_threshold    = config.DISTANCE_THRESHOLD,
        correlation_threshold = config.CORRELATION_THRESHOLD,
    )

    # Graph comparison (always computed for the report)
    print("[evaluate] Comparing distance vs correlation graphs...")
    W_dist = build_distance_W(
        csv_path=csv_path, num_nodes=num_nodes,
        sigma_sq=config.SIGMA_SQ, threshold=config.DISTANCE_THRESHOLD,
    )
    W_corr = build_corr_W(
        npz_path=npz_path, num_nodes=num_nodes,
        train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO,
        threshold=config.CORRELATION_THRESHOLD,
    )
    stats = compare_graphs(W_dist, W_corr, top_k=config.XAI_TOP_NEIGHBORS)
    print(f"  Distance edges={stats['distance_edges']}, "
          f"Correlation edges={stats['correlation_edges']}, "
          f"Shared={stats['shared_edges']}, "
          f"Jaccard={stats['jaccard_global']:.3f}")

    # ------------------------------------------------------------------ #
    # Inference on test set                                               #
    # ------------------------------------------------------------------ #
    print("[evaluate] Running inference on test set...")
    preds_orig, trues_orig = run_inference(
        model, test_loader, adj_norm, device, scaler
    )
    print(f"[evaluate] Test predictions shape: {preds_orig.shape}")

    # ------------------------------------------------------------------ #
    # Metrics                                                             #
    # ------------------------------------------------------------------ #
    reg_metrics = compute_regression_metrics(preds_orig, trues_orig)
    clf_results = compute_classification_metrics(
        preds_orig, trues_orig,
        free_flow_threshold = config.FREE_FLOW_THRESHOLD,
        moderate_threshold  = config.MODERATE_THRESHOLD,
    )

    print_metrics_table(reg_metrics, clf_results, config.CLASS_NAMES)

    # ------------------------------------------------------------------ #
    # Plots                                                               #
    # ------------------------------------------------------------------ #
    print("[evaluate] Generating plots...")
    plot_loss_curves(config.OUTPUT_DIR)
    plot_pred_vs_true(preds_orig, trues_orig, config.VIZ_SENSORS)
    plot_confusion_matrix(clf_results, config.CLASS_NAMES)
    plot_spatial_error(preds_orig, trues_orig)
    plot_graph_overlap(W_dist, W_corr, config.VIZ_SENSORS)

    print("\n[evaluate] All plots saved to:", config.PLOT_DIR)
    print("[evaluate] Run 'python explain.py' for full XAI analysis "
          "(neighbor ablation + dynamic correlation).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate T-GCN")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: config.BEST_MODEL_PATH)")
    parser.add_argument("--graph", type=str, default=None,
                        choices=["distance", "correlation"],
                        help="Override graph type (default: read from checkpoint)")
    parser.add_argument("--dataset", type=str, default="pems07",
                        choices=["pems07", "casablanca"])
    args = parser.parse_args()
    main(args.checkpoint, args.graph, args.dataset)
