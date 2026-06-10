"""
simulation/sumo/generate_demand.py
----------------------------------
Génère les fichiers de demande (routes) et détecteurs E1 pour les 5 boulevards.
"""

import os
import xml.etree.ElementTree as ET

from simulation.config import (
    BOULEVARDS,
    DETECTOR_FILE,
    NET_FILE,
    ROUTE_FILE,
    SIMULATION_DURATION,
    SUMO_CFG,
    TRAFFIC_VEHICLE_COUNT,
)
from simulation.sumo.build_network import run_sumo_tool


def _parse_net_edges(net_path: str) -> list[dict]:
    """Extrait les arêtes du réseau avec leurs noms de rue."""
    tree = ET.parse(net_path)
    root = tree.getroot()
    edges = []

    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if edge_id.startswith(":"):
            continue

        name = edge.get("name", "") or ""
        length = float(edge.get("length", "100"))
        edges.append({"id": edge_id, "name": name, "length": length})

    return edges


def map_boulevards_to_edges(net_path: str = NET_FILE) -> dict[int, list[str]]:
    """
    Associe chaque boulevard aux arêtes SUMO correspondantes
    (matching par nom de rue OSM).
    """
    edges = _parse_net_edges(net_path)
    mapping = {b["id"]: [] for b in BOULEVARDS}

    for edge in edges:
        name_lower = edge["name"].lower()
        for b in BOULEVARDS:
            for osm_name in b["osm_names"]:
                if osm_name.lower() in name_lower or name_lower in osm_name.lower():
                    mapping[b["id"]].append(edge["id"])
                    break

    # Fallback : si aucune arête trouvée, distribuer les arêtes principales
    unassigned = [e for e in edges if e["length"] > 200]
    for b in BOULEVARDS:
        if not mapping[b["id"]] and unassigned:
            chunk = unassigned[:max(1, len(unassigned) // 5)]
            mapping[b["id"]] = [e["id"] for e in chunk]
            unassigned = unassigned[len(chunk):]

    for b in BOULEVARDS:
        print(f"  [{b['name']}] → {len(mapping[b['id']])} arêtes")

    return mapping


def generate_routes(net_path: str = NET_FILE, route_path: str = ROUTE_FILE) -> str:
    """Génère routes et types de véhicules avec randomTrips.py (script SUMO)."""
    os.makedirs(os.path.dirname(route_path), exist_ok=True)

    net_abs = os.path.abspath(net_path)
    route_abs = os.path.abspath(route_path)
    period = max(0.5, SIMULATION_DURATION / TRAFFIC_VEHICLE_COUNT)

    args = [
        "-n", net_abs,
        "-r", route_abs,
        "-e", str(SIMULATION_DURATION),
        "-p", str(period),
        "--validate",
        "--trip-attributes", 'departLane="best" departSpeed="max" departPos="random"',
    ]

    print(f"[SUMO] Génération demande trafic via randomTrips.py (period={period:.1f}s)...")
    result = run_sumo_tool("randomTrips", args, cwd=os.path.dirname(net_abs))

    if result.returncode != 0:
        print(f"[WARN] randomTrips: {result.stderr or result.stdout}")
        _write_minimal_routes(route_path)
    else:
        print(f"[SUMO] Routes créées: {route_path}")

    return route_path


def _write_minimal_routes(route_path: str):
    """Routes minimales si randomTrips échoue."""
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5",
                  sigma="0.5", length="5", maxSpeed="50")
    ET.SubElement(root, "flow", id="flow_0", type="car", begin="0",
                  end=str(SIMULATION_DURATION), number="200", departSpeed="max")
    tree = ET.ElementTree(root)
    tree.write(route_path, encoding="UTF-8", xml_declaration=True)


def _get_lane_lengths(net_path: str) -> dict[str, float]:
    """Retourne {lane_id: length_m} depuis le réseau SUMO."""
    tree = ET.parse(net_path)
    lengths = {}
    for lane in tree.getroot().iter("lane"):
        lane_id = lane.get("id")
        if lane_id:
            lengths[lane_id] = float(lane.get("length", "0"))
    return lengths


def _safe_detector_pos(lane_id: str, lane_lengths: dict[str, float]) -> float | None:
    """Position sûre au milieu de la voie (évite l'erreur 'beyond lane end')."""
    length = lane_lengths.get(lane_id, 0.0)
    if length < 2.0:
        return None
    pos = length * 0.5
    return round(max(0.5, min(pos, length - 0.5)), 2)


def generate_detectors(boulevard_edges: dict, net_path: str = NET_FILE,
                       detector_path: str = DETECTOR_FILE) -> str:
    """Crée des détecteurs E1 (induction loops) sur chaque boulevard."""
    root = ET.Element("additionalFile")
    lane_lengths = _get_lane_lengths(net_path)
    count = 0

    for b in BOULEVARDS:
        edges = boulevard_edges.get(b["id"], [])
        for i, edge_id in enumerate(edges[:3]):
            lane_id = f"{edge_id}_0"
            pos = _safe_detector_pos(lane_id, lane_lengths)
            if pos is None:
                continue

            det_id = f"det_b{b['id']}_e{i}"
            ET.SubElement(
                root,
                "inductionLoop",
                id=det_id,
                lane=lane_id,
                pos=str(pos),
                friendlyPos="true",
                freq="60",
                file=f"detector_b{b['id']}.xml",
            )
            count += 1

    tree = ET.ElementTree(root)
    tree.write(detector_path, encoding="UTF-8", xml_declaration=True)
    print(f"[SUMO] Détecteurs créés: {detector_path} ({count} boucles)")
    return detector_path


def generate_sumocfg(net_path: str = NET_FILE, route_path: str = ROUTE_FILE,
                     detector_path: str = DETECTOR_FILE,
                     cfg_path: str = SUMO_CFG) -> str:
    """Génère le fichier de configuration SUMO."""
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=os.path.basename(net_path))
    ET.SubElement(inp, "route-files", value=os.path.basename(route_path))
    ET.SubElement(inp, "additional-files", value=os.path.basename(detector_path))

    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value=str(SIMULATION_DURATION))

    tree = ET.ElementTree(root)
    tree.write(cfg_path, encoding="UTF-8", xml_declaration=True)
    print(f"[SUMO] Config: {cfg_path}")
    return cfg_path


def setup_all(net_path: str = NET_FILE) -> dict:
    """Pipeline complet : mapping, routes, détecteurs, config."""
    print("[SUMO] Mapping boulevards → arêtes...")
    boulevard_edges = map_boulevards_to_edges(net_path)
    generate_routes(net_path)
    generate_detectors(boulevard_edges, net_path)
    generate_sumocfg(net_path)
    return boulevard_edges


if __name__ == "__main__":
    setup_all()
