"""
simulation/export_tgcn.py
-------------------------
Exporte les données MongoDB (split=tgcn) vers le format attendu par T-GCN :
  - CASABLANCA05.npz  : matrice [T, N] des vitesses
  - CASABLANCA05.csv  : graphe de distances entre boulevards
"""

import math
import os

import numpy as np
import pandas as pd

from simulation.config import (
    BOULEVARDS,
    TGCN_CSV_PATH,
    TGCN_DATA_DIR,
    TGCN_NPZ_PATH,
)
from simulation.mongodb.repository import MongoRepository


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance géodésique entre deux points (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def build_distance_csv(output_path: str = TGCN_CSV_PATH) -> str:
    """
    Construit le graphe de distances entre les 5 boulevards
    (format compatible avec graph_utils.build_distance_W).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    n = len(BOULEVARDS)

    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = BOULEVARDS[i]["center"]
            lat2, lon2 = BOULEVARDS[j]["center"]
            dist = haversine_km(lat1, lon1, lat2, lon2)
            rows.append({"from": i, "to": j, "cost": round(dist, 4)})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[Export] Graphe distances: {output_path} ({len(rows)} arêtes)")
    return output_path


def export_npz(repo: MongoRepository, run_id: str = None,
               output_path: str = TGCN_NPZ_PATH,
               split: str = "tgcn") -> str:
    """
    Exporte la matrice de vitesses [T, N] depuis MongoDB vers .npz.

    Parameters
    ----------
    repo        : MongoRepository
    run_id      : ID de simulation (dernière run si None)
    output_path : chemin du fichier .npz
    split       : 'tgcn' pour entraînement, 'test' pour évaluation, None pour tout
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if run_id is None:
        latest = repo.get_latest_run()
        if not latest:
            raise ValueError("Aucune simulation trouvée dans MongoDB.")
        run_id = latest["run_id"]

    result = repo.get_timeseries_matrix(run_id, split=split)
    matrix = result["data"]

    if matrix is None or matrix.size == 0:
        raise ValueError(f"Aucune donnée pour run_id={run_id}, split={split}")

    np.savez_compressed(output_path, data=matrix.astype(np.float32))

    T, N = matrix.shape
    print(f"[Export] NPZ: {output_path} — shape [{T}, {N}], split={split}")
    return output_path


def export_all(repo: MongoRepository, run_id: str = None):
    """Export complet : NPZ (tgcn + test) + CSV distances."""
    build_distance_csv()

    export_npz(repo, run_id, TGCN_NPZ_PATH, split="tgcn")

    test_path = TGCN_NPZ_PATH.replace(".npz", "_test.npz")
    try:
        export_npz(repo, run_id, test_path, split="test")
    except ValueError:
        print("[Export] Pas de données test séparées.")

    print("\n[Export] Prêt pour T-GCN !")
    print(f"  Entraînement : python train.py --dataset casablanca")
    print(f"  Fichiers     : {TGCN_NPZ_PATH}, {TGCN_CSV_PATH}")


if __name__ == "__main__":
    repo = MongoRepository()
    export_all(repo)
    repo.close()
