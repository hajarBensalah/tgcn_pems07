"""
simulation/metrics.py
---------------------
Calcul des métriques de trafic par boulevard :
  - vitesse moyenne (km/h)
  - débit (véhicules/h)
  - densité (véh/km)
  - taux d'occupation (%)
  - niveau de service (Congested / Moderate / Free Flow)
"""

import numpy as np

from simulation.config import BOULEVARDS, FREE_FLOW_THRESHOLD, MODERATE_THRESHOLD


def classify_traffic_state(avg_speed_kmh: float) -> str:
    if avg_speed_kmh <= MODERATE_THRESHOLD:
        return "Congested"
    if avg_speed_kmh <= FREE_FLOW_THRESHOLD:
        return "Moderate"
    return "Free Flow"


def compute_step_metrics(boulevard_data: dict) -> list[dict]:
    """
    Calcule les métriques pour un pas de simulation.

    Parameters
    ----------
    boulevard_data : dict[boulevard_id → {speed, flow, density, occupancy}]

    Returns
    -------
    list[dict] — une entrée par boulevard
    """
    results = []
    for b in BOULEVARDS:
        bid = b["id"]
        d = boulevard_data.get(bid, {})

        avg_speed = float(d.get("avg_speed_kmh", 0.0))
        flow = float(d.get("flow_vph", 0.0))
        density = float(d.get("density_vpk", 0.0))
        occupancy = float(d.get("occupancy_pct", 0.0))

        results.append({
            "boulevard_id": bid,
            "boulevard_name": b["name"],
            "orientation": b["orientation"],
            "avg_speed_kmh": round(avg_speed, 2),
            "flow_vph": round(flow, 1),
            "density_vpk": round(density, 2),
            "occupancy_pct": round(occupancy, 2),
            "traffic_state": classify_traffic_state(avg_speed),
        })

    return results


def aggregate_run_metrics(timeseries_records: list[dict]) -> list[dict]:
    """
    Agrège les métriques sur toute la durée de simulation par boulevard.

    Parameters
    ----------
    timeseries_records : liste de documents MongoDB (split=test ou tgcn)

    Returns
    -------
    list[dict] — statistiques globales par boulevard
    """
    by_boulevard: dict[int, list] = {b["id"]: [] for b in BOULEVARDS}

    for rec in timeseries_records:
        bid = rec["boulevard_id"]
        if bid in by_boulevard:
            by_boulevard[bid].append(rec)

    aggregated = []
    for b in BOULEVARDS:
        bid = b["id"]
        records = by_boulevard[bid]
        if not records:
            continue

        speeds = [r["avg_speed_kmh"] for r in records]
        flows = [r.get("flow_vph", 0) for r in records]
        densities = [r.get("density_vpk", 0) for r in records]

        aggregated.append({
            "boulevard_id": bid,
            "boulevard_name": b["name"],
            "orientation": b["orientation"],
            "mean_speed_kmh": round(float(np.mean(speeds)), 2),
            "std_speed_kmh": round(float(np.std(speeds)), 2),
            "min_speed_kmh": round(float(np.min(speeds)), 2),
            "max_speed_kmh": round(float(np.max(speeds)), 2),
            "mean_flow_vph": round(float(np.mean(flows)), 1),
            "mean_density_vpk": round(float(np.mean(densities)), 2),
            "num_samples": len(records),
            "dominant_state": _dominant_state(speeds),
        })

    return aggregated


def _dominant_state(speeds: list[float]) -> str:
    states = [classify_traffic_state(s) for s in speeds]
    from collections import Counter
    return Counter(states).most_common(1)[0][0]


def print_metrics_table(metrics: list[dict]):
    """Affiche un tableau des métriques dans la console."""
    print("\n" + "=" * 90)
    print(f"{'Boulevard':<30} {'V.moy':>8} {'Débit':>10} {'Densité':>10} {'État':>12}")
    print("=" * 90)

    for m in metrics:
        name = m.get("boulevard_name", "?")[:28]
        speed = m.get("mean_speed_kmh", m.get("avg_speed_kmh", 0))
        flow = m.get("mean_flow_vph", m.get("flow_vph", 0))
        density = m.get("mean_density_vpk", m.get("density_vpk", 0))
        state = m.get("dominant_state", m.get("traffic_state", ""))
        print(f"{name:<30} {speed:>7.1f} {flow:>9.0f} {density:>9.1f} {state:>12}")

    print("=" * 90)
