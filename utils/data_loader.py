"""
utils/data_loader.py
--------------------
Load PEMS07 traffic speed data, preprocess it, and build PyTorch
DataLoader objects for train / validation / test splits.

Steps performed
---------------
1. Load the .npz file (key = 'data'):  shape [T, N, C]  or  [T, N]
2. Extract the speed feature (channel 0 if multi-variate)
3. Handle missing values via linear interpolation (per sensor)
4. Z-score normalisation  (fit on train split only)
5. Temporal train / val / test split  (60 / 20 / 20 %)
6. Sliding-window sequence generation  (seq_len inputs, pred_len outputs)
7. Wrap in TensorDataset → DataLoader
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pems07(npz_path: str, train_ratio: float = 0.60,
                val_ratio: float = 0.20):
    """
    Load the PEMS07.npz file and return the raw speed array plus split info.

    Parameters
    ----------
    npz_path    : path to PEMS07.npz
    train_ratio : fraction of timesteps for training
    val_ratio   : fraction of timesteps for validation

    Returns
    -------
    speed        : np.ndarray [T, N]  — raw (possibly NaN) speed values
    train_end    : int — last training timestep index
    val_end      : int — last validation timestep index
    """
    data = np.load(npz_path, allow_pickle=True)

    # The PEMS07 .npz typically stores the array under key 'data'
    key = "data" if "data" in data else list(data.keys())[0]
    raw = data[key]                         # [T, N] or [T, N, C]

    # If multi-variate take only the first channel (speed)
    if raw.ndim == 3:
        speed = raw[:, :, 0].astype(np.float32)
    else:
        speed = raw.astype(np.float32)

    T, N = speed.shape
    print(f"[data_loader] Loaded PEMS07: T={T} timesteps, N={N} sensors")

    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    print(f"[data_loader] Split — train:[0,{train_end}), "
          f"val:[{train_end},{val_end}), test:[{val_end},{T})")

    return speed, train_end, val_end


def build_dataset(npz_path: str,
                  train_ratio : float = 0.60,
                  val_ratio   : float = 0.20,
                  seq_len     : int   = 12,
                  pred_len    : int   = 12,
                  batch_size  : int   = 64) -> dict:
    """
    Full preprocessing pipeline.  Returns DataLoaders and metadata.

    Parameters
    ----------
    npz_path    : path to PEMS07.npz
    train_ratio : fraction for training
    val_ratio   : fraction for validation
    seq_len     : number of historical timesteps as input
    pred_len    : number of future timesteps to predict
    batch_size  : DataLoader batch size

    Returns
    -------
    dict with keys:
        'train_loader'  : DataLoader
        'val_loader'    : DataLoader
        'test_loader'   : DataLoader
        'scaler'        : dict {'mean': np.ndarray, 'std': np.ndarray}
        'num_nodes'     : int
        'seq_len'       : int
        'pred_len'      : int
    """
    # ------------------------------------------------------------------ #
    # 1. Load raw data                                                    #
    # ------------------------------------------------------------------ #
    speed, train_end, val_end = load_pems07(npz_path, train_ratio, val_ratio)
    T, N = speed.shape

    orig_seq, orig_pred = seq_len, pred_len
    seq_len, pred_len = resolve_window_sizes(T, train_ratio, seq_len, pred_len)
    if seq_len != orig_seq or pred_len != orig_pred:
        print(f"[data_loader] Fenêtre ajustée : seq_len={seq_len}, pred_len={pred_len}")

    # ------------------------------------------------------------------ #
    # 2. Handle missing values (NaN → linear interpolation per sensor)   #
    # ------------------------------------------------------------------ #
    speed = _interpolate_missing(speed)
    print(f"[data_loader] NaN handling complete. "
          f"Remaining NaNs: {np.isnan(speed).sum()}")

    # ------------------------------------------------------------------ #
    # 3. Z-score normalisation (fit ONLY on training data)               #
    # ------------------------------------------------------------------ #
    train_data = speed[:train_end, :]
    mu  = np.nanmean(train_data, axis=0)   # [N] — per-sensor mean
    std = np.nanstd(train_data,  axis=0)   # [N] — per-sensor std
    std[std == 0] = 1.0                    # avoid division by zero

    speed_norm = (speed - mu) / std        # [T, N]

    scaler = {"mean": mu, "std": std}

    # ------------------------------------------------------------------ #
    # 4. Split into train / val / test                                   #
    # ------------------------------------------------------------------ #
    train_arr = speed_norm[:train_end]
    val_arr   = speed_norm[train_end:val_end]
    test_arr  = speed_norm[val_end:]

    # ------------------------------------------------------------------ #
    # 5. Sliding window sequences                                        #
    # ------------------------------------------------------------------ #
    X_train, y_train = _create_sequences(train_arr, seq_len, pred_len)
    X_val,   y_val   = _create_sequences(val_arr,   seq_len, pred_len)
    X_test,  y_test  = _create_sequences(test_arr,  seq_len, pred_len)

    print(f"[data_loader] Sequence shapes — "
          f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ------------------------------------------------------------------ #
    # 6. Convert to PyTorch tensors and build DataLoaders                #
    # ------------------------------------------------------------------ #
    def _to_loader(X, y, shuffle):
        X_t = torch.tensor(X, dtype=torch.float32)  # [samples, seq, N, 1]
        y_t = torch.tensor(y, dtype=torch.float32)  # [samples, N, pred]
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=False, pin_memory=True, num_workers=0)

    train_loader = _to_loader(X_train, y_train, shuffle=True)
    val_loader   = _to_loader(X_val,   y_val,   shuffle=False)
    test_loader  = _to_loader(X_test,  y_test,  shuffle=False)

    return {
        "train_loader" : train_loader,
        "val_loader"   : val_loader,
        "test_loader"  : test_loader,
        "scaler"       : scaler,
        "num_nodes"    : N,
        "seq_len"      : seq_len,
        "pred_len"     : pred_len,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpolate_missing(speed: np.ndarray) -> np.ndarray:
    """
    Replace NaN values in the speed array with linearly interpolated values.

    Processing is done sensor-by-sensor (column-by-column) using pandas
    interpolate(), which handles interior NaNs via linear interpolation
    and fills leading/trailing NaNs via forward/backward fill.

    Parameters
    ----------
    speed : np.ndarray [T, N]

    Returns
    -------
    np.ndarray [T, N] with NaNs replaced
    """
    T, N = speed.shape
    result = speed.copy()

    for n in range(N):
        col = pd.Series(speed[:, n])
        col = col.interpolate(method="linear", limit_direction="both")
        result[:, n] = col.values

    return result.astype(np.float32)


def resolve_window_sizes(T: int, train_ratio: float,
                         seq_len: int, pred_len: int) -> tuple[int, int]:
    """
    Ajuste seq_len / pred_len si le jeu de données est court
    (typiquement Casablanca après simulation SUMO).
    """
    train_T = int(T * train_ratio)
    needed = seq_len + pred_len

    while train_T < needed and seq_len > 2:
        seq_len -= 1
        pred_len -= 1
        needed = seq_len + pred_len

    if train_T < needed:
        min_T = int(np.ceil(needed / train_ratio))
        raise ValueError(
            f"Jeu de données trop court : T={T} pas, partition train={train_T}. "
            f"Il faut au moins seq_len+pred_len={needed} pas dans le train "
            f"(soit T>={min_T} au total).\n"
            f"Relancez : python -m simulation.run_pipeline --step simulate "
            f"puis --step export"
        )

    return seq_len, pred_len


def _create_sequences(arr: np.ndarray, seq_len: int,
                      pred_len: int):
    """
    Build input / target pairs using a sliding window over time.

    For each valid starting position t:
        X[t] = arr[t : t + seq_len]          shape [seq_len, N]
        y[t] = arr[t + seq_len : t + seq_len + pred_len]  shape [pred_len, N]

    Parameters
    ----------
    arr      : np.ndarray [T, N]
    seq_len  : int
    pred_len : int

    Returns
    -------
    X : np.ndarray [num_samples, seq_len, N, 1]
    y : np.ndarray [num_samples, N, pred_len]
    """
    T, N = arr.shape
    num_samples = T - seq_len - pred_len + 1

    X = np.zeros((num_samples, seq_len, N, 1),   dtype=np.float32)
    y = np.zeros((num_samples, N, pred_len),      dtype=np.float32)

    for i in range(num_samples):
        X[i] = arr[i : i + seq_len, :, np.newaxis]          # [seq, N, 1]
        y[i] = arr[i + seq_len : i + seq_len + pred_len, :].T  # [N, pred]

    return X, y


def inverse_transform(arr: np.ndarray, scaler: dict) -> np.ndarray:
    """
    Undo Z-score normalisation.

    Parameters
    ----------
    arr    : np.ndarray [..., N]  — normalised values (last dim = sensors)
    scaler : dict with 'mean' and 'std' arrays of shape [N]

    Returns
    -------
    np.ndarray same shape as arr, in original speed units
    """
    return arr * scaler["std"] + scaler["mean"]
