"""
utils/metrics.py
----------------
Evaluation metrics for T-GCN traffic speed prediction.

Regression metrics
------------------
- MAE   : Mean Absolute Error
- RMSE  : Root Mean Squared Error
- MAPE  : Mean Absolute Percentage Error  (skips near-zero true values)

Classification metrics
----------------------
Traffic states are defined by speed thresholds (in original km/h units):
    Class 0 — Congested   : speed ≤ 30
    Class 1 — Moderate    : 30 < speed ≤ 60
    Class 2 — Free Flow   : speed > 60

Using a per-class confusion-matrix we compute:
    Precision, Recall, F1 — per class + macro average
"""

import numpy as np
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def mape(y_pred: np.ndarray, y_true: np.ndarray,
         eps: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.

    True values with absolute magnitude below `eps` are excluded from the
    calculation to avoid division by near-zero values inflating MAPE.

    Parameters
    ----------
    eps : float — minimum absolute true value to include (default 1.0 km/h)
    """
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_pred[mask] - y_true[mask])
                                        / y_true[mask])))


def compute_regression_metrics(y_pred: np.ndarray,
                                y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute all regression metrics at once.

    Parameters
    ----------
    y_pred : np.ndarray — model predictions  (any shape, flattened internally)
    y_true : np.ndarray — ground-truth values (same shape)

    Returns
    -------
    dict with keys 'MAE', 'RMSE', 'MAPE'
    """
    y_pred_flat = y_pred.ravel()
    y_true_flat = y_true.ravel()

    return {
        "MAE"  : mae(y_pred_flat,  y_true_flat),
        "RMSE" : rmse(y_pred_flat, y_true_flat),
        "MAPE" : mape(y_pred_flat, y_true_flat),
    }


# ---------------------------------------------------------------------------
# Traffic-state classification helpers
# ---------------------------------------------------------------------------

def speed_to_class(speed: np.ndarray,
                   free_flow_threshold: float = 60.0,
                   moderate_threshold:  float = 30.0) -> np.ndarray:
    """
    Convert continuous speed values (km/h) to discrete traffic-state labels.

    Labels:
        0 — Congested   : speed ≤ moderate_threshold
        1 — Moderate    : moderate_threshold < speed ≤ free_flow_threshold
        2 — Free Flow   : speed > free_flow_threshold

    Parameters
    ----------
    speed : np.ndarray — speed values in original units (km/h)

    Returns
    -------
    np.ndarray of int  (same shape as speed)
    """
    labels = np.zeros_like(speed, dtype=np.int64)
    labels[speed > moderate_threshold]  = 1   # Moderate (provisional)
    labels[speed > free_flow_threshold] = 2   # Free Flow (overrides Moderate)
    return labels


def compute_classification_metrics(y_pred: np.ndarray,
                                   y_true: np.ndarray,
                                   free_flow_threshold: float = 60.0,
                                   moderate_threshold:  float = 30.0,
                                   num_classes: int = 3
                                   ) -> Dict[str, object]:
    """
    Convert predictions and ground truth to traffic-state classes, then
    compute per-class and macro-average Precision, Recall and F1.

    Parameters
    ----------
    y_pred              : np.ndarray — predicted speeds in original units
    y_true              : np.ndarray — true speeds in original units
    free_flow_threshold : float
    moderate_threshold  : float
    num_classes         : int (3)

    Returns
    -------
    dict:
        'confusion_matrix' : np.ndarray [num_classes, num_classes]
        'precision'        : np.ndarray [num_classes]  — per class
        'recall'           : np.ndarray [num_classes]  — per class
        'f1'               : np.ndarray [num_classes]  — per class
        'macro_precision'  : float
        'macro_recall'     : float
        'macro_f1'         : float
        'pred_labels'      : np.ndarray  (for external reuse)
        'true_labels'      : np.ndarray
    """
    # Convert continuous speeds to class labels
    pred_labels = speed_to_class(y_pred.ravel(), free_flow_threshold,
                                  moderate_threshold)
    true_labels = speed_to_class(y_true.ravel(), free_flow_threshold,
                                  moderate_threshold)

    # Build confusion matrix  [true_class, pred_class]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    # Per-class Precision, Recall, F1
    precision = np.zeros(num_classes, dtype=np.float64)
    recall    = np.zeros(num_classes, dtype=np.float64)
    f1        = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp   # predicted as c but not actually c
        fn = cm[c, :].sum() - tp   # actually c but predicted as something else

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precision[c] = prec
        recall[c]    = rec
        f1[c]        = f1_c

    return {
        "confusion_matrix" : cm,
        "precision"        : precision,
        "recall"           : recall,
        "f1"               : f1,
        "macro_precision"  : float(precision.mean()),
        "macro_recall"     : float(recall.mean()),
        "macro_f1"         : float(f1.mean()),
        "pred_labels"      : pred_labels,
        "true_labels"      : true_labels,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_metrics_table(reg: Dict[str, float],
                        clf: Dict[str, object],
                        class_names=("Congested", "Moderate", "Free Flow")):
    """
    Print a formatted metrics summary table to stdout.

    Parameters
    ----------
    reg         : output of compute_regression_metrics
    clf         : output of compute_classification_metrics
    class_names : sequence of three class name strings
    """
    line = "─" * 54
    print(f"\n┌{line}┐")
    print(f"│{'TEST METRICS SUMMARY':^54}│")
    print(f"├{line}┤")
    print(f"│  {'Metric':<20} {'Value':>30}  │")
    print(f"├{line}┤")
    print(f"│  {'MAE':<20} {reg['MAE']:>30.4f}  │")
    print(f"│  {'RMSE':<20} {reg['RMSE']:>30.4f}  │")
    print(f"│  {'MAPE (%)':<20} {reg['MAPE']:>30.2f}  │")
    print(f"├{line}┤")
    print(f"│  {'Macro Precision':<20} {clf['macro_precision']:>30.4f}  │")
    print(f"│  {'Macro Recall':<20} {clf['macro_recall']:>30.4f}  │")
    print(f"│  {'Macro F1':<20} {clf['macro_f1']:>30.4f}  │")
    print(f"├{line}┤")
    print(f"│  {'Class':<15} {'Prec':>8} {'Rec':>8} {'F1':>8} {'':>5}  │")
    print(f"├{line}┤")
    for i, name in enumerate(class_names):
        p = clf['precision'][i]
        r = clf['recall'][i]
        f = clf['f1'][i]
        print(f"│  {name:<15} {p:>8.4f} {r:>8.4f} {f:>8.4f} {'':>5}  │")
    print(f"└{line}┘\n")
