"""
simulation/config.py
--------------------
Configuration for SUMO + OSM simulation of 5 Casablanca boulevards
and MongoDB storage.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ---------------------------------------------------------------------------
# Casablanca — centre-ville (grille de 5 boulevards parallèles/perpendiculaires)
# ---------------------------------------------------------------------------
CASABLANCA_CENTER = (33.5731, -7.5898)  # lat, lon

# Bbox globale pour télécharger le réseau OSM (≈ 3 km × 3 km)
OSM_BBOX = {
    "south": 33.5580,
    "west":  -7.6200,
    "north": 33.5880,
    "east":  -7.5600,
}

# ---------------------------------------------------------------------------
# 5 boulevards : 2 axes N-S parallèles + 2 axes E-O parallèles + 1 connecteur
# ---------------------------------------------------------------------------
BOULEVARDS = [
    {
        "id": 0,
        "name": "Boulevard Mohammed V",
        "orientation": "N-S",
        "osm_names": ["Boulevard Mohammed V", "Mohammed V"],
        "center": (33.5731, -7.5898),
    },
    {
        "id": 1,
        "name": "Boulevard Zerktouni",
        "orientation": "N-S",
        "osm_names": ["Boulevard Zerktouni", "Zerktouni"],
        "center": (33.5731, -7.6050),
    },
    {
        "id": 2,
        "name": "Boulevard Hassan II",
        "orientation": "E-O",
        "osm_names": ["Boulevard Hassan II", "Hassan II"],
        "center": (33.5820, -7.5970),
    },
    {
        "id": 3,
        "name": "Avenue des FAR",
        "orientation": "E-O",
        "osm_names": ["Avenue des FAR", "des FAR"],
        "center": (33.5650, -7.5970),
    },
    {
        "id": 4,
        "name": "Boulevard Brahim Roudani",
        "orientation": "connecteur",
        "osm_names": ["Boulevard Brahim Roudani", "Brahim Roudani"],
        "center": (33.5731, -7.5970),
    },
]

NUM_BOULEVARDS = len(BOULEVARDS)

# ---------------------------------------------------------------------------
# Chemins SUMO
# ---------------------------------------------------------------------------
SUMO_DIR = os.path.join(BASE_DIR, "sumo_files")
OSM_DIR = os.path.join(BASE_DIR, "osm")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

OSM_FILE = os.path.join(OSM_DIR, "casablanca.osm.xml")
NET_FILE = os.path.join(SUMO_DIR, "casablanca.net.xml")
ROUTE_FILE = os.path.join(SUMO_DIR, "casablanca.rou.xml")
DETECTOR_FILE = os.path.join(SUMO_DIR, "casablanca.add.xml")
SUMO_CFG = os.path.join(SUMO_DIR, "casablanca.sumocfg")

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SIMULATION_DURATION = 3600       # secondes (1 heure simulée)
SIMULATION_STEP = 30             # pas d'échantillonnage (30 s → plus de points pour T-GCN)
WARMUP_STEPS = 300                 # secondes de chauffe avant collecte

# Nombre cible de véhicules sur toute la simulation (plus = trafic plus dense)
# Exemple : 2500 (~léger), 25000 (~très dense, ~7 véh/s sur 3600 s)
TRAFFIC_VEHICLE_COUNT = 3000 #Changeant les valeurs en 3000 véhicules
# Période minimum entre deux départs (randomTrips -p)
TRAFFIC_MIN_PERIOD = 0.1
# Ralenti GUI (ms entre chaque pas) — uniquement avec --gui
SUMO_GUI_DELAY_MS = 80

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "tgcn_casablanca")

COLLECTIONS = {
    "boulevards": "boulevards",
    "runs": "simulation_runs",
    "timeseries": "speed_timeseries",
    "metrics": "boulevard_metrics",
    "splits": "dataset_splits",
}

# Répartition des données : test (visualisation) vs TGCN (entraînement)
TEST_RATIO = 0.40          # 40 % pour tests / métriques
TGCN_RATIO = 0.60          # 60 % pour entraînement T-GCN

# ---------------------------------------------------------------------------
# Export TGCN
# ---------------------------------------------------------------------------
TGCN_DATA_DIR = os.path.join(PROJECT_DIR, "data")
TGCN_NPZ_PATH = os.path.join(TGCN_DATA_DIR, "CASABLANCA05.npz")
TGCN_CSV_PATH = os.path.join(TGCN_DATA_DIR, "CASABLANCA05.csv")

# ---------------------------------------------------------------------------
# SUMO_HOME (Windows / Linux)
# ---------------------------------------------------------------------------
SUMO_HOME = os.environ.get("SUMO_HOME", "")

# ---------------------------------------------------------------------------
# Seuils classification trafic (alignés sur config.py T-GCN)
# ---------------------------------------------------------------------------
FREE_FLOW_THRESHOLD = 60
MODERATE_THRESHOLD = 30
