"""
simulation/view_metrics.py
--------------------------
Visualise les métriques par boulevard depuis MongoDB.
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import matplotlib.pyplot as plt
import numpy as np

from simulation.config import BOULEVARDS, OUTPUT_DIR
from simulation.metrics import print_metrics_table
from simulation.mongodb.repository import MongoRepository

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")


def plot_speed_timeseries(run_id: str = None):
    """Graphique vitesse moyenne par boulevard dans le temps."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    repo = MongoRepository()
    if run_id is None:
        latest = repo.get_latest_run()
        if not latest:
            print("Aucune simulation.")
            return
        run_id = latest["run_id"]

    for split, label in [("test", "Test"), ("tgcn", "T-GCN")]:
        result = repo.get_timeseries_matrix(run_id, split=split)
        matrix = result["data"]
        if matrix is None:
            continue

        T, N = matrix.shape
        fig, ax = plt.subplots(figsize=(12, 5))
        steps = np.arange(T)

        for i, b in enumerate(BOULEVARDS[:N]):
            ax.plot(steps, matrix[:, i], label=b["name"], linewidth=1.5)

        ax.set_xlabel("Pas de temps (intervalle simulation)")
        ax.set_ylabel("Vitesse moyenne (km/h)")
        ax.set_title(f"Vitesse par boulevard — {label} (run: {run_id[:8]}...)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = os.path.join(PLOT_DIR, f"speed_{split}_{run_id[:8]}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[Plot] {out}")

    metrics = repo.get_metrics_by_boulevard(run_id)
    print_metrics_table(metrics)
    repo.close()


if __name__ == "__main__":
    plot_speed_timeseries()
