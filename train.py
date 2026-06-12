"""
train.py
--------
Full training loop for T-GCN on PEMS07.

Usage
-----
    python train.py

The script:
1. Loads and preprocesses PEMS07 data
2. Builds the normalised adjacency matrix
3. Instantiates the T-GCN model
4. Trains with Adam + ReduceLROnPlateau + early stopping
5. Saves the best model checkpoint to outputs/checkpoints/best_model.pt
6. Plots training / validation loss curves
"""

import os
import random
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from models      import TGCN, count_parameters
from utils       import build_dataset, build_adj
from utils.metrics import compute_regression_metrics


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    loader,
                    adj_norm: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    criterion,
                    device: torch.device) -> float:
    """
    Run one full training epoch.

    Parameters
    ----------
    model     : T-GCN model
    loader    : DataLoader (train split)
    adj_norm  : normalised adjacency tensor on `device`
    optimizer : Adam
    criterion : MAE loss
    device    : torch device

    Returns
    -------
    float — mean batch loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)   # [B, seq, N, 1]
        y_batch = y_batch.to(device)   # [B, N, pred]

        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_batch, adj_norm)   # [B, N, pred]

        # MAE loss
        loss = criterion(y_pred, y_batch)
        loss.backward()

        # Gradient clipping (optional but stabilises GCN-RNN training)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader,
             adj_norm: torch.Tensor,
             criterion,
             device: torch.device,
             scaler: dict) -> tuple:
    """
    Evaluate the model on a data split.

    Returns
    -------
    (loss, mae_value) — both computed in the normalised space for loss,
    and in original speed units for MAE.
    """
    model.eval()
    total_loss = 0.0
    preds_list = []
    trues_list = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch, adj_norm)

        loss = criterion(y_pred, y_batch)
        total_loss += loss.item()

        preds_list.append(y_pred.cpu().numpy())
        trues_list.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)

    # Convert to original speed units for MAE reporting
    preds = np.concatenate(preds_list, axis=0)  # [samples, N, pred]
    trues = np.concatenate(trues_list, axis=0)

    # Inverse transform: un-normalise
    mu  = scaler["mean"]   # [N]
    std = scaler["std"]    # [N]

    preds_orig = preds * std[None, :, None] + mu[None, :, None]
    trues_orig = trues * std[None, :, None] + mu[None, :, None]

    mae_orig = float(np.mean(np.abs(preds_orig - trues_orig)))

    return avg_loss, mae_orig


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def main(graph_type: str = None, dataset: str = "pems07"):
    if graph_type is None:
        graph_type = config.GRAPH_TYPE
    set_seed(config.SEED)

    if dataset == "casablanca":
        npz_path = config.CASABLANCA_NPZ
        csv_path = config.CASABLANCA_CSV
        dataset_label = "Casablanca (5 boulevards)"
        seq_len = config.CASABLANCA_SEQ_LEN
        pred_len = config.CASABLANCA_PRED_LEN
        batch_size = min(config.BATCH_SIZE, 16)
    else:
        npz_path = config.NPZ_PATH
        csv_path = config.CSV_PATH
        dataset_label = "PEMS07"
        seq_len = config.SEQ_LEN
        pred_len = config.PRED_LEN
        batch_size = config.BATCH_SIZE

    # ------------------------------------------------------------------ #
    # Device                                                              #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data                                                                #
    # ------------------------------------------------------------------ #
    print(f"[train] Loading {dataset_label} dataset...")
    data_dict = build_dataset(
        npz_path    = npz_path,
        train_ratio = config.TRAIN_RATIO,
        val_ratio   = config.VAL_RATIO,
        seq_len     = seq_len,
        pred_len    = pred_len,
        batch_size  = batch_size,
    )

    train_loader = data_dict["train_loader"]
    val_loader   = data_dict["val_loader"]
    scaler       = data_dict["scaler"]
    num_nodes    = data_dict["num_nodes"]
    seq_len      = data_dict["seq_len"]
    pred_len     = data_dict["pred_len"]

    # ------------------------------------------------------------------ #
    # Adjacency matrix                                                    #
    # ------------------------------------------------------------------ #
    print(f"[train] Building adjacency matrix (graph_type={graph_type})...")
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

    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    model = TGCN(
        num_nodes   = num_nodes,
        in_features = 1,
        hidden_dim  = config.HIDDEN_DIM,
        pred_len    = pred_len,
        gcn_layers  = config.GCN_LAYERS,
    ).to(device)

    print(f"[train] Model parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------------ #
    # Loss, optimiser, scheduler                                          #
    # ------------------------------------------------------------------ #
    criterion = nn.L1Loss()   # MAE loss

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = config.LR_SCHEDULER_FACTOR,
        patience = config.LR_SCHEDULER_PATIENCE,
        min_lr   = config.LR_SCHEDULER_MIN_LR,
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                       #
    # ------------------------------------------------------------------ #
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR,       exist_ok=True)

    best_val_loss  = float("inf")
    epochs_no_impr = 0

    train_losses = []
    val_losses   = []

    print("\n" + "=" * 65)
    print(f"  Starting T-GCN training — {config.NUM_EPOCHS} epochs max")
    print("=" * 65)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        # --- Train ---
        train_loss = train_one_epoch(
            model, train_loader, adj_norm, optimizer, criterion, device
        )

        # --- Validate ---
        val_loss, val_mae = evaluate(
            model, val_loader, adj_norm, criterion, device, scaler
        )

        # --- Scheduler step ---
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # --- Log ---
        print(f"[Epoch {epoch:>3}/{config.NUM_EPOCHS}] "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_MAE: {val_mae:.4f} | "
              f"lr: {current_lr:.2e} | "
              f"time: {elapsed:.1f}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # --- Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            epochs_no_impr  = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "opt_state"  : optimizer.state_dict(),
                "val_loss"   : val_loss,
                "val_mae"    : val_mae,
                "scaler"     : scaler,
                "config"     : {
                    "num_nodes"  : num_nodes,
                    "hidden_dim" : config.HIDDEN_DIM,
                    "gcn_layers" : config.GCN_LAYERS,
                    "seq_len"    : seq_len,
                    "pred_len"   : pred_len,
                    "graph_type" : graph_type,
                    "dataset"    : dataset,
                },
            }, config.BEST_MODEL_PATH)
            print(f"  ✔ New best model saved  (val_loss={val_loss:.4f})")
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= config.PATIENCE:
                print(f"\n[train] Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {config.PATIENCE} epochs).")
                break

    print(f"\n[train] Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"[train] Checkpoint saved to: {config.BEST_MODEL_PATH}")

    # ------------------------------------------------------------------ #
    # Save loss curves data (used by evaluate.py for plotting)           #
    # ------------------------------------------------------------------ #
    np.save(os.path.join(config.OUTPUT_DIR, "train_losses.npy"),
            np.array(train_losses))
    np.save(os.path.join(config.OUTPUT_DIR, "val_losses.npy"),
            np.array(val_losses))

    # ------------------------------------------------------------------ #
    # Quick loss-curve plot saved inline                                 #
    # ------------------------------------------------------------------ #
    _save_loss_curve(train_losses, val_losses, dataset_label)

    print("[train] Loss curve saved to outputs/plots/loss_curve.png")


def _save_loss_curve(train_losses, val_losses, dataset_label="PEMS07"):
    """Save training / validation loss curves without launching evaluate.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train Loss (MAE)", linewidth=2)
    ax.plot(epochs, val_losses,   label="Val Loss (MAE)",   linewidth=2,
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss (normalised)")
    ax.set_title(f"T-GCN Training and Validation Loss — {dataset_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(config.PLOT_DIR, "loss_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T-GCN")
    parser.add_argument("--graph", type=str, default=None,
                        choices=["distance", "correlation"],
                        help="Adjacency type: distance (level 1) or correlation (level 2)")
    parser.add_argument("--dataset", type=str, default="pems07",
                        choices=["pems07", "casablanca"],
                        help="Dataset: pems07 (New York) or casablanca (5 boulevards)")
    args = parser.parse_args()
    main(args.graph, args.dataset)
