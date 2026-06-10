"""
simulation/osm/download_osm.py
------------------------------
Télécharge les données OpenStreetMap pour la zone Casablanca
via l'API Overpass.
"""

import os
import urllib.error
import urllib.parse
import urllib.request

from simulation.config import OSM_BBOX, OSM_FILE

try:
    import requests
    from requests import RequestException
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    RequestException = Exception

# Miroirs Overpass (essai séquentiel si le premier échoue)
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]

USER_AGENT = "TGCN-Casablanca/1.0 (traffic simulation; academic project)"


def build_overpass_query(bbox: dict) -> str:
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    return (
        f'[out:xml][timeout:120];'
        f'('
        f'way["highway"~"^(primary|secondary|tertiary|trunk|motorway|primary_link|secondary_link)$"]'
        f'({s},{w},{n},{e});'
        f');'
        f'(._;>;);'
        f'out body;'
    )


def _download_via_requests(url: str, query: str, timeout: int = 180) -> bytes:
    """Via requests (SSL + encodage plus fiable sous Windows)."""
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    resp = requests.post(
        url,
        data={"data": query},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content


def _download_via_get(url: str, query: str, timeout: int = 180) -> bytes:
    """GET — format le plus compatible avec Overpass."""
    encoded = urllib.parse.urlencode({"data": query})
    full_url = f"{url}?{encoded}"
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_via_post(url: str, query: str, timeout: int = 180) -> bytes:
    """POST — body data=<query> (format officiel Overpass)."""
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "*/*",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_osm(output_path: str = OSM_FILE, bbox: dict = None) -> str:
    """
    Télécharge le fichier OSM XML pour la bbox Casablanca.

    Returns
    -------
    str : chemin du fichier OSM téléchargé
    """
    bbox = bbox or OSM_BBOX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 1000:
        size_kb = os.path.getsize(output_path) / 1024
        print(f"[OSM] Fichier existant réutilisé: {output_path} ({size_kb:.1f} KB)")
        return output_path

    query = build_overpass_query(bbox)
    print(f"[OSM] Téléchargement zone Casablanca ({bbox})...")

    errors = []
    for url in OVERPASS_URLS:
        methods = []
        if HAS_REQUESTS:
            methods.append(("requests", _download_via_requests))
        methods.extend([("GET", _download_via_get), ("POST", _download_via_post)])

        for method_name, download_fn in methods:
            try:
                print(f"[OSM] Essai {method_name} sur {url}...")
                content = download_fn(url, query)
                if len(content) < 500:
                    raise RuntimeError(f"Réponse trop courte ({len(content)} octets)")
                with open(output_path, "wb") as f:
                    f.write(content)
                size_kb = len(content) / 1024
                print(f"[OSM] Fichier sauvegardé: {output_path} ({size_kb:.1f} KB)")
                return output_path
            except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError,
                    OSError, RequestException) as e:
                errors.append(f"{method_name} {url}: {e}")
                continue

    raise RuntimeError(
        "Échec téléchargement OSM après plusieurs tentatives:\n"
        + "\n".join(f"  - {err}" for err in errors)
        + f"\n\nTéléchargez manuellement depuis https://www.openstreetmap.org "
        f"et placez le fichier dans {output_path}"
    )


if __name__ == "__main__":
    download_osm()
