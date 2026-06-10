"""
simulation/sumo/run_simulation.py
---------------------------------
Lance la simulation SUMO via TraCI et collecte les vitesses par boulevard.
Stocke les résultats dans MongoDB.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np

from simulation.config import (
    BOULEVARDS,
    NET_FILE,
    SIMULATION_DURATION,
    SIMULATION_STEP,
    SUMO_CFG,
    SUMO_GUI_DELAY_MS,
    WARMUP_STEPS,
)
from simulation.metrics import aggregate_run_metrics, compute_step_metrics, print_metrics_table
from simulation.mongodb.repository import MongoRepository
from simulation.sumo.build_network import find_sumo_binary, get_sumo_home
from simulation.sumo.generate_demand import _parse_net_edges


def _setup_sumo_path():
    """Ajoute SUMO/tools au PYTHONPATH pour importer traci."""
    sumo_home = get_sumo_home()
    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.append(tools)


def _load_edge_lengths(net_path: str = NET_FILE) -> dict[str, float]:
    """Longueurs des arêtes (m) depuis le fichier réseau — évite TraCI getLength."""
    return {e["id"]: e["length"] for e in _parse_net_edges(net_path)}


def run_sumo_simulation(
    boulevard_edges: dict,
    repo: MongoRepository,
    run_id: str,
    use_gui: bool = False,
) -> list[dict]:
    """
    Exécute SUMO avec TraCI et collecte les métriques par boulevard.

    Parameters
    ----------
    boulevard_edges : mapping boulevard_id → liste d'edge_ids SUMO
    repo            : instance MongoRepository
    run_id          : identifiant de la run
    use_gui         : utiliser sumo-gui au lieu de sumo

    Returns
    -------
    list[dict] — métriques agrégées par boulevard
    """
    _setup_sumo_path()

    try:
        import traci
    except ImportError:
        raise ImportError(
            "Module traci introuvable. Installez SUMO et définissez SUMO_HOME.\n"
            "Alternative: python -m simulation.run_pipeline --demo"
        )

    sumo_bin = find_sumo_binary("sumo-gui" if use_gui else "sumo")
    cfg_path = os.path.abspath(SUMO_CFG)

    cmd = [sumo_bin, "-c", cfg_path, "--start", "--quit-on-end",
           "--step-length", "1"]
    if use_gui:
        cmd.extend(["--delay", str(SUMO_GUI_DELAY_MS)])
        print(f"[TraCI] Mode GUI — delay={SUMO_GUI_DELAY_MS}ms (zoomez sur le centre-ville)")
    traci.start(cmd)

    all_records = []
    step_count = 0
    sample_interval = SIMULATION_STEP
    edge_lengths = _load_edge_lengths()

    print(f"[TraCI] Simulation {SIMULATION_DURATION}s, échantillonnage /{sample_interval}s...")

    try:
        while traci.simulation.getMinExpectedNumber() > 0 or traci.simulation.getTime() < SIMULATION_DURATION:
            traci.simulationStep()
            current_time = int(traci.simulation.getTime())

            if current_time < WARMUP_STEPS:
                continue

            if current_time % sample_interval != 0:
                continue

            boulevard_data = _collect_boulevard_data(boulevard_edges, edge_lengths)
            step_metrics = compute_step_metrics(boulevard_data)

            records = []
            for m in step_metrics:
                records.append({
                    "run_id": run_id,
                    "step": step_count,
                    "sim_time_s": current_time,
                    "boulevard_id": m["boulevard_id"],
                    "boulevard_name": m["boulevard_name"],
                    "avg_speed_kmh": m["avg_speed_kmh"],
                    "flow_vph": m["flow_vph"],
                    "density_vpk": m["density_vpk"],
                    "occupancy_pct": m["occupancy_pct"],
                    "traffic_state": m["traffic_state"],
                    "recorded_at": datetime.now(timezone.utc),
                })

            all_records.extend(records)
            step_count += 1

            if step_count % 10 == 0:
                print(f"  step {step_count} — t={current_time}s")

    finally:
        traci.close()

    repo.insert_timeseries_batch(all_records)
    repo.assign_splits(run_id)

    test_records = [r for r in all_records if r.get("split") != "tgcn"]
    aggregated = aggregate_run_metrics(all_records)
    repo.store_boulevard_metrics(run_id, aggregated)

    print_metrics_table(aggregated)
    return aggregated


def _collect_boulevard_data(boulevard_edges: dict,
                            edge_lengths: dict[str, float]) -> dict:
    """Collecte vitesse, débit et densité par boulevard depuis TraCI."""
    import traci

    data = {}

    for b in BOULEVARDS:
        bid = b["id"]
        edges = boulevard_edges.get(bid, [])

        speeds = []
        occupancies = []
        vehicle_count = 0
        total_length = 0.0

        for edge_id in edges:
            try:
                n_veh = traci.edge.getLastStepVehicleNumber(edge_id)
                mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                occ = traci.edge.getLastStepOccupancy(edge_id)
                length = edge_lengths.get(edge_id, 0.0)

                if n_veh > 0 and mean_speed >= 0:
                    speeds.append(mean_speed * 3.6)
                occupancies.append(occ)
                vehicle_count += n_veh
                total_length += length
            except traci.TraCIException:
                continue

        avg_speed = float(np.mean(speeds)) if speeds else 30.0
        flow_vph = vehicle_count * (3600.0 / max(SIMULATION_STEP, 1))
        density = (vehicle_count / max(total_length / 1000.0, 0.01)) if total_length > 0 else 0.0
        avg_occupancy = float(np.mean(occupancies)) if occupancies else min(100.0, density * 7.0)

        data[bid] = {
            "avg_speed_kmh": avg_speed,
            "flow_vph": flow_vph,
            "density_vpk": density,
            "occupancy_pct": avg_occupancy,
        }

    return data


def run_demo_simulation(repo: MongoRepository, run_id: str) -> list[dict]:
    """
    Mode démo sans SUMO : génère des données synthétiques réalistes
    pour tester MongoDB et l'export TGCN.
    """
    print("[DEMO] Génération de données synthétiques (SUMO non disponible)...")

    num_steps = SIMULATION_DURATION // SIMULATION_STEP
    rng = np.random.default_rng(42)

    base_speeds = {0: 35, 1: 42, 2: 38, 3: 45, 4: 50}
    base_flows = {0: 800, 1: 600, 2: 700, 3: 550, 4: 400}

    all_records = []

    for step in range(num_steps):
        hour_factor = 0.7 + 0.3 * np.sin(2 * np.pi * step / num_steps)
        boulevard_data = {}

        for b in BOULEVARDS:
            bid = b["id"]
            noise = rng.normal(0, 5)
            speed = max(5, base_speeds[bid] * hour_factor + noise)
            flow = base_flows[bid] * hour_factor + rng.normal(0, 50)
            density = flow / max(speed, 1) * 0.5

            boulevard_data[bid] = {
                "avg_speed_kmh": speed,
                "flow_vph": flow,
                "density_vpk": density,
                "occupancy_pct": min(100, density * 7),
            }

        step_metrics = compute_step_metrics(boulevard_data)
        for m in step_metrics:
            all_records.append({
                "run_id": run_id,
                "step": step,
                "sim_time_s": step * SIMULATION_STEP,
                "boulevard_id": m["boulevard_id"],
                "boulevard_name": m["boulevard_name"],
                "avg_speed_kmh": m["avg_speed_kmh"],
                "flow_vph": m["flow_vph"],
                "density_vpk": m["density_vpk"],
                "occupancy_pct": m["occupancy_pct"],
                "traffic_state": m["traffic_state"],
                "recorded_at": datetime.now(timezone.utc),
            })

    repo.insert_timeseries_batch(all_records)
    repo.assign_splits(run_id)

    aggregated = aggregate_run_metrics(all_records)
    repo.store_boulevard_metrics(run_id, aggregated)
    print_metrics_table(aggregated)
    return aggregated
