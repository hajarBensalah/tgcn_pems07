"""
config.py
---------
Central configuration file for all hyperparameters and paths used
throughout the T-GCN project. Import this module in any script to
access settings without hard-coding values.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
PLOT_DIR        = os.path.join(OUTPUT_DIR, "plots")
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

NPZ_PATH        = os.path.join(DATA_DIR, "PEMS07.npz")   # traffic speed array
CSV_PATH        = os.path.join(DATA_DIR, "PEMS07.csv")   # sensor distances

# Casablanca (5 boulevards — généré par simulation/)
CASABLANCA_NPZ   = os.path.join(DATA_DIR, "CASABLANCA05.npz")
CASABLANCA_CSV   = os.path.join(DATA_DIR, "CASABLANCA05.csv")
CASABLANCA_NODES = 5

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------
NUM_NODES   = 883           # number of sensors in PEMS07
SEQ_LEN     = 12            # number of input timesteps  (historical window)
PRED_LEN    = 12            # number of output timesteps (forecast horizon)

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20          # implicit: 1 - TRAIN - VAL

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
# Distance graph (level 1 — Gaussian kernel): w_ij = exp(−d_ij² / sigma²)
# Set to None to auto-calibrate from the data's distance variance (recommended)
SIGMA_SQ           = None
# edges with weight below this threshold are set to 0 (sparsification)
# 0.1 keeps ~60% of edges for PEMS07 distances
DISTANCE_THRESHOLD = 0.1

# Correlation graph (level 2 — traffic-driven links)
# Built from Pearson correlation of sensor speeds on the train split only.
CORRELATION_THRESHOLD = 0.5

# Graph type for train.py / evaluate.py: "distance" or "correlation"
GRAPH_TYPE = "distance"

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_DIM  = 64    # T-GCN hidden state dimension
GCN_LAYERS  = 1     # number of GCN layers inside each gate

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5

# Early stopping: stop if val loss does not improve for PATIENCE epochs
PATIENCE = 15

# ReduceLROnPlateau scheduler settings
LR_SCHEDULER_FACTOR   = 0.5    # multiply LR by this factor on plateau
LR_SCHEDULER_PATIENCE = 5      # epochs to wait before reducing LR
LR_SCHEDULER_MIN_LR   = 1e-6   # floor for the learning rate

# ---------------------------------------------------------------------------
# Classification thresholds (km/h) for traffic-state evaluation
# ---------------------------------------------------------------------------
FREE_FLOW_THRESHOLD = 60   # speed > 60  → Free Flow  (class 2)
MODERATE_THRESHOLD  = 30   # 30 < speed ≤ 60 → Moderate (class 1)
                            # speed ≤ 30 → Congested         (class 0)

CLASS_NAMES = ["Congested", "Moderate", "Free Flow"]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Visualization sensors (0-indexed)
# ---------------------------------------------------------------------------
VIZ_SENSORS = [5, 10, 50]   # sensors for pred-vs-true plot

# ---------------------------------------------------------------------------
# XAI (explain.py)
# ---------------------------------------------------------------------------
XAI_SENSORS       = [5, 10, 50]   # target sensors for neighbor ablation
XAI_MAX_SAMPLES   = 200           # test samples used for ablation (speed)
XAI_TOP_NEIGHBORS = 15            # neighbors shown in influence plots
