# T-GCN — Prédiction du trafic routier

Projet de **Temporal Graph Convolutional Network (T-GCN)** pour la prévision des vitesses de trafic, avec deux jeux de données :

| Jeu de données | Région | Nœuds | Source |
|----------------|--------|-------|--------|
| **PEMS07** | New York | 883 capteurs | Données réelles (benchmark) |
| **Casablanca** | 5 boulevards | 5 | Simulation SUMO + OpenStreetMap |

Le pipeline Casablanca simule le trafic, stocke les résultats dans **MongoDB**, exporte un dataset compatible T-GCN, puis entraîne le même modèle.

---

## Architecture

```
OpenStreetMap ──► SUMO (netconvert + TraCI) ──► MongoDB
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ▼                               ▼
                              split test (40%)              split T-GCN (60%)
                                    │                               │
                                    └────────► CASABLANCA05.npz ◄───┘
                                                    │
                                                    ▼
                                              T-GCN (PyTorch)
```

**PEMS07** (branche parallèle) : entraînement direct depuis `data/PEMS07.npz` + graphe distance ou corrélation.

---

## Prérequis

- **Python** 3.10 ou 3.11
- **PyTorch** 2.x (GPU optionnel)
- **MongoDB** (Docker recommandé) — pour la simulation Casablanca
- **Eclipse SUMO** 1.27+ — pour la simulation réelle ([télécharger](https://sumo.dlr.de/docs/Downloads.php))

---

## Installation

```powershell
git clone https://github.com/hajarBensalah/tgcn_pems07.git
cd TGCN
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

### PyTorch GPU (optionnel)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### MongoDB (Docker)

```powershell
docker run -d --name mongo-tgcn -p 27017:27017 mongo:7
docker start mongo-tgcn
```

### SUMO

1. Installer `sumo-win64-1.27.0.msi`
2. Définir la variable d'environnement :

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

Variables optionnelles (fichier `.env` ou environnement) :

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB=tgcn_casablanca
SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

---

## Données

### PEMS07 (New York)

Placer dans `data/` :

| Fichier | Description |
|---------|-------------|
| `PEMS07.npz` | Matrice de vitesses `[T, 883]` |
| `PEMS07.csv` | Distances entre capteurs (graphe) |

> `PEMS07.csv` est déjà présent. `PEMS07.npz` doit être obtenu séparément (benchmark public).

### Casablanca (généré par simulation)

| Fichier | Description |
|---------|-------------|
| `CASABLANCA05.npz` | Vitesses simulées `[T, 5]` |
| `CASABLANCA05.csv` | Distances entre boulevards |

Ces fichiers sont créés automatiquement par le pipeline SUMO (voir ci-dessous).

**5 boulevards modélisés :**

| ID | Boulevard | Orientation |
|----|-----------|-------------|
| 0 | Boulevard Mohammed V | N–S |
| 1 | Boulevard Zerktouni | N–S |
| 2 | Boulevard Hassan II | E–O |
| 3 | Avenue des FAR | E–O |
| 4 | Boulevard Brahim Roudani | Connecteur |

---

## Utilisation — PEMS07

```powershell
# Entraînement (graphe distance)
python train.py --graph distance

# Entraînement (graphe corrélation)
python train.py --graph correlation

# Évaluation
python evaluate.py

# Explicabilité (XAI)
python explain.py
```

Résultats : `outputs/checkpoints/best_model.pt`, graphiques dans `outputs/plots/`.

---

## Utilisation — Simulation SUMO (Casablanca)

### Pipeline complet

```powershell
docker start mongo-tgcn
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"

python -m simulation.run_pipeline
```

### Étape par étape

```powershell
python -m simulation.run_pipeline --step osm        # Télécharger OpenStreetMap
python -m simulation.run_pipeline --step sumo       # Construire le réseau SUMO
python -m simulation.run_pipeline --step simulate     # Simulation TraCI → MongoDB
python -m simulation.run_pipeline --step export       # Export CASABLANCA05.npz
python -m simulation.run_pipeline --metrics           # Métriques par boulevard
```

### Options utiles

```powershell
# Mode démo sans SUMO (données synthétiques)
python -m simulation.run_pipeline --demo

# Voir la simulation graphiquement
python -m simulation.run_pipeline --step simulate --gui

# Trafic dense : 25 000 véhicules
python -m simulation.run_pipeline --vehicles 25000

# Graphiques des vitesses
python -m simulation.view_metrics
```

### Entraînement T-GCN sur Casablanca

```powershell
python train.py --dataset casablanca --graph distance
python evaluate.py --dataset casablanca
```

---

## Métriques collectées (SUMO)

Pour chaque boulevard, à chaque pas de temps :

- Vitesse moyenne (km/h)
- Débit (véhicules/heure)
- Densité (véh./km)
- Taux d'occupation (%)
- État du trafic : **Congested** (≤ 30), **Moderate** (≤ 60), **Free Flow** (> 60)

**Répartition MongoDB :** 40 % test · 60 % entraînement T-GCN

---

## Structure du projet

```
TGCN/
├── config.py                 # Hyperparamètres globaux
├── train.py                  # Entraînement T-GCN
├── evaluate.py               # Évaluation + graphiques
├── explain.py                # Analyse XAI
├── models/                   # T-GCN, GCN, cellule GRU
├── utils/                    # Data loader, graphes, métriques
├── data/                     # PEMS07 + CASABLANCA05
├── simulation/               # Pipeline SUMO + OSM + MongoDB
│   ├── config.py
│   ├── run_pipeline.py       # Orchestrateur principal
│   ├── osm/                  # Téléchargement OpenStreetMap
│   ├── sumo/                 # Réseau, routes, simulation TraCI
│   ├── mongodb/              # Stockage MongoDB
│   └── export_tgcn.py        # Export vers .npz / .csv
├── outputs/                  # Checkpoints, plots (généré)
└── rapport/                  # Rapport LaTeX
```

---

## Configuration principale

| Paramètre | PEMS07 | Casablanca |
|-----------|--------|------------|
| `SEQ_LEN` / `PRED_LEN` | 12 / 12 | 6 / 6 |
| Nœuds | 883 | 5 |
| Split | 60 / 20 / 20 % | idem (via `data_loader`) |

Paramètres simulation (`simulation/config.py`) :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `SIMULATION_DURATION` | 3600 s | Durée simulée |
| `SIMULATION_STEP` | 30 s | Échantillonnage |
| `TRAFFIC_VEHICLE_COUNT` | 2500 | Véhicules cibles |
| `TEST_RATIO` | 0.40 | Part réservée aux tests |

---

## Dépannage

| Problème | Solution |
|----------|----------|
| `MongoDB inaccessible` | `docker start mongo-tgcn` |
| `SUMO_HOME introuvable` | Définir `$env:SUMO_HOME` |
| `traci introuvable` | Installer SUMO + redémarrer le terminal |
| OSM HTTP 406 | Relancer `--step osm` (requête corrigée) |
| `negative dimensions` (train Casablanca) | `python -m simulation.run_pipeline --step export` puis réentraîner |
| Pas de voitures en GUI | Zoomer sur le centre, attendre t > 300 s |
| Simulation 25k véh. lente | Ne pas utiliser `--gui`, prévoir 30 min+ |

---

## Références

- T-GCN : [Zhao et al., 2019](https://arxiv.org/abs/1911.09360)
- SUMO : [https://eclipse.dev/sumo/](https://eclipse.dev/sumo/)
- PEMS07 : benchmark public de prédiction du trafic

---

## Licence

Projet académique — voir le dépôt pour les détails.
