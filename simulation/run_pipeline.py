"""
simulation/run_pipeline.py
--------------------------
Pipeline complet : OSM → SUMO → MongoDB → Export T-GCN

Usage
-----
  # Mode démo (sans SUMO, pour tester MongoDB + T-GCN)
  python -m simulation.run_pipeline --demo

  # Pipeline complet avec SUMO
  python -m simulation.run_pipeline

  # Étapes individuelles
  python -m simulation.run_pipeline --step osm
  python -m simulation.run_pipeline --step sumo
  python -m simulation.run_pipeline --step simulate
  python -m simulation.run_pipeline --step export

  # Afficher les métriques d'une run
  python -m simulation.run_pipeline --metrics
"""

import argparse
import os
import sys

# Ajouter la racine du projet au path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def step_osm():
    from simulation.osm.download_osm import download_osm
    download_osm()


def step_build_network():
    from simulation.sumo.build_network import build_network
    build_network()


def step_setup_demand(vehicle_count: int = None):
    from simulation.sumo.generate_demand import setup_all
    return setup_all(vehicle_count=vehicle_count)


def step_simulate(demo: bool = False, use_gui: bool = False, skip_setup: bool = False,
                  vehicle_count: int = None):
    from simulation.config import TRAFFIC_VEHICLE_COUNT
    from simulation.mongodb.repository import MongoRepository
    from simulation.sumo.generate_demand import map_boulevards_to_edges
    from simulation.sumo.run_simulation import run_demo_simulation, run_sumo_simulation

    vehicle_count = vehicle_count or TRAFFIC_VEHICLE_COUNT

    repo = MongoRepository()
    try:
        repo.ping()
        print("[MongoDB] Connexion OK")
    except Exception as e:
        print(f"[ERREUR] MongoDB inaccessible: {e}")
        print("  Démarrez MongoDB: mongod  ou  docker run -d -p 27017:27017 mongo")
        sys.exit(1)

    repo.init_boulevards()

    run_id = repo.create_run({
        "mode": "demo" if demo else "sumo",
        "vehicle_count": vehicle_count,
    })
    print(f"[Run] ID: {run_id}")

    if demo:
        aggregated = run_demo_simulation(repo, run_id)
        boulevard_edges = {}
    else:
        if skip_setup:
            boulevard_edges = map_boulevards_to_edges()
        else:
            boulevard_edges = step_setup_demand(vehicle_count=vehicle_count)
        try:
            aggregated = run_sumo_simulation(boulevard_edges, repo, run_id, use_gui)
        except (ImportError, FileNotFoundError) as e:
            print(f"[WARN] SUMO indisponible ({e}), bascule en mode démo...")
            aggregated = run_demo_simulation(repo, run_id)

    summary = {
        "num_steps": sum(m.get("num_samples", 0) for m in aggregated) // max(len(aggregated), 1),
        "boulevards": [m["boulevard_name"] for m in aggregated],
        "mean_speeds": {m["boulevard_name"]: m["mean_speed_kmh"] for m in aggregated},
    }
    repo.finish_run(run_id, summary)

    print(f"\n[OK] Simulation terminée — run_id: {run_id}")
    repo.close()
    return run_id


def step_export(run_id: str = None):
    from simulation.export_tgcn import export_all
    from simulation.mongodb.repository import MongoRepository

    repo = MongoRepository()
    export_all(repo, run_id)
    repo.close()


def show_metrics(run_id: str = None):
    from simulation.metrics import print_metrics_table
    from simulation.mongodb.repository import MongoRepository

    repo = MongoRepository()
    if run_id is None:
        latest = repo.get_latest_run()
        if not latest:
            print("Aucune simulation trouvée.")
            return
        run_id = latest["run_id"]

    test_metrics = repo.get_metrics_by_boulevard(run_id)
    print(f"\nMétriques — run_id: {run_id}")
    print_metrics_table(test_metrics)

    ts_test = repo.get_timeseries_matrix(run_id, split="test")
    ts_tgcn = repo.get_timeseries_matrix(run_id, split="tgcn")
    print(f"\nDonnées test  : {ts_test['data'].shape if ts_test['data'] is not None else 'vide'}")
    print(f"Données T-GCN : {ts_tgcn['data'].shape if ts_tgcn['data'] is not None else 'vide'}")
    repo.close()


def main():
    parser = argparse.ArgumentParser(description="Pipeline SUMO + OSM + MongoDB → T-GCN")
    parser.add_argument("--step", choices=["all", "osm", "sumo", "simulate", "export", "metrics"],
                        default="all")
    parser.add_argument("--demo", action="store_true",
                        help="Mode démo sans SUMO (données synthétiques)")
    parser.add_argument("--gui", action="store_true", help="Afficher SUMO-GUI")
    parser.add_argument("--metrics", action="store_true",
                        help="Afficher les métriques (raccourci pour --step metrics)")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--vehicles", type=int, default=None,
                        help="Nombre de véhicules (ex: 25000). Défaut: config TRAFFIC_VEHICLE_COUNT")
    args = parser.parse_args()

    if args.metrics:
        args.step = "metrics"

    vehicle_count = args.vehicles

    if args.step in ("all", "osm") and not args.demo:
        step_osm()

    if args.step in ("all", "sumo") and not args.demo:
        step_build_network()
        step_setup_demand(vehicle_count=vehicle_count)

    run_id = args.run_id
    if args.step in ("all", "simulate"):
        skip_setup = args.step == "all" and not args.demo
        run_id = step_simulate(demo=args.demo, use_gui=args.gui, skip_setup=skip_setup,
                               vehicle_count=vehicle_count)

    if args.step in ("all", "export"):
        step_export(run_id)

    if args.step == "metrics":
        show_metrics(args.run_id)


if __name__ == "__main__":
    main()
