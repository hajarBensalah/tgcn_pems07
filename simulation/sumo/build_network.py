"""
simulation/sumo/build_network.py
--------------------------------
Construit le réseau SUMO à partir du fichier OSM via netconvert.
"""

import os
import subprocess
import shutil
import sys

from simulation.config import (
    OSM_FILE,
    NET_FILE,
    SUMO_DIR,
    SUMO_HOME,
)

DEFAULT_SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"


def get_sumo_home() -> str:
    """Retourne SUMO_HOME (variable d'env ou chemin par défaut Windows)."""
    candidates = [
        SUMO_HOME,
        os.environ.get("SUMO_HOME", ""),
        DEFAULT_SUMO_HOME,
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "SUMO_HOME introuvable.\n"
        "Installez SUMO: https://eclipse.dev/sumo/docs/Installing/index.html\n"
        "Puis définissez SUMO_HOME (ex: C:\\Program Files (x86)\\Eclipse\\Sumo)"
    )


def find_sumo_binary(name: str) -> str:
    """Trouve un binaire SUMO (netconvert, sumo, etc.) dans bin/."""
    sumo_home = get_sumo_home()
    path = os.path.join(sumo_home, "bin", name)
    if os.path.isfile(path + ".exe"):
        return path + ".exe"
    if os.path.isfile(path):
        return path

    found = shutil.which(name)
    if found:
        return found

    raise FileNotFoundError(
        f"Binaire '{name}' introuvable dans {os.path.join(sumo_home, 'bin')}.\n"
        "Vérifiez votre installation SUMO."
    )


def find_sumo_tool(script_name: str) -> str:
    """
    Trouve un script Python SUMO dans tools/ (ex: randomTrips.py).
    Retourne la commande complète [python, script.py].
    """
    sumo_home = get_sumo_home()
    script_path = os.path.join(sumo_home, "tools", script_name)
    if not script_path.endswith(".py"):
        script_path += ".py"

    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Script SUMO '{script_name}' introuvable: {script_path}\n"
            "Vérifiez que SUMO est installé avec le dossier tools/."
        )

    return script_path


def run_sumo_tool(script_name: str, args: list[str], cwd: str = None) -> subprocess.CompletedProcess:
    """Exécute un script Python SUMO (randomTrips, etc.)."""
    script_path = find_sumo_tool(script_name)
    env = os.environ.copy()
    env["SUMO_HOME"] = get_sumo_home()

    cmd = [sys.executable, script_path, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def build_network(osm_path: str = OSM_FILE, net_path: str = NET_FILE) -> str:
    """
    Convertit OSM → réseau SUMO (.net.xml).

    Returns
    -------
    str : chemin du fichier réseau
    """
    os.makedirs(os.path.dirname(net_path), exist_ok=True)

    if not os.path.isfile(osm_path):
        raise FileNotFoundError(
            f"Fichier OSM manquant: {osm_path}\n"
            "Lancez d'abord: python -m simulation.osm.download_osm"
        )

    netconvert = find_sumo_binary("netconvert")
    print(f"[SUMO] Conversion OSM → réseau avec {netconvert}...")

    cmd = [
        netconvert,
        "--osm-files", osm_path,
        "--output-file", net_path,
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.default-type", "actuated",
        "--output.street-names", "true",
        "--keep-edges.by-vclass", "passenger",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"netconvert a échoué:\n{result.stderr}\n{result.stdout}"
        )

    print(f"[SUMO] Réseau créé: {net_path}")
    return net_path


if __name__ == "__main__":
    build_network()
