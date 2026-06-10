# Simulation SUMO + OSM + MongoDB — Casablanca

Pipeline pour simuler le trafic sur **5 boulevards de Casablanca** avec SUMO (OpenStreetMap), stocker les données dans **MongoDB**, et exporter vers **T-GCN**.

## Architecture

```
OpenStreetMap (Overpass API)
        ↓
   netconvert (SUMO)
        ↓
   Simulation TraCI
        ↓
     MongoDB
    ↙        ↘
 split=test   split=tgcn
 (métriques)  (entraînement)
        ↓
  CASABLANCA05.npz + .csv
        ↓
      T-GCN
```

## 5 boulevards (grille Casablanca)

| ID | Boulevard | Orientation |
|----|-----------|-------------|
| 0 | Boulevard Mohammed V | N-S |
| 1 | Boulevard Zerktouni | N-S (parallèle) |
| 2 | Boulevard Hassan II | E-O |
| 3 | Avenue des FAR | E-O (parallèle) |
| 4 | Boulevard Brahim Roudani | Connecteur |

## Prérequis

### 1. Python
```powershell
pip install -r requirements.txt
```

### 2. MongoDB
```powershell
# Option Docker
docker run -d --name mongo-tgcn -p 27017:27017 mongo:7

# Ou installer MongoDB Community: https://www.mongodb.com/try/download/community
```

### 3. SUMO (optionnel pour mode réel)
1. Télécharger : https://eclipse.dev/sumo/docs/Installing/index.html
2. Définir la variable d'environnement :
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

> **Sans SUMO** : utilisez `--demo` pour générer des données synthétiques et tester MongoDB + T-GCN.

## Utilisation

### Mode démo (recommandé pour débuter)
```powershell
cd C:\Users\ilias\Desktop\TGCN
python -m simulation.run_pipeline --demo
```

### Pipeline complet (OSM + SUMO + MongoDB)
```powershell
python -m simulation.run_pipeline
```

### Étapes individuelles
```powershell
python -m simulation.run_pipeline --step osm       # Télécharger OSM
python -m simulation.run_pipeline --step sumo      # Construire réseau SUMO
python -m simulation.run_pipeline --step simulate # Lancer simulation
python -m simulation.run_pipeline --step export    # Exporter vers T-GCN
python -m simulation.run_pipeline --metrics        # Afficher métriques
```

### Visualiser les métriques
```powershell
python -m simulation.view_metrics
```

## MongoDB — Collections

| Collection | Contenu |
|------------|---------|
| `boulevards` | Définition des 5 boulevards |
| `simulation_runs` | Métadonnées des runs |
| `speed_timeseries` | Vitesse/débit/densité par pas de temps |
| `boulevard_metrics` | Métriques agrégées par boulevard |
| `dataset_splits` | Répartition test / tgcn |

### Répartition des données
- **40 % test** : visualisation, évaluation des métriques
- **60 % tgcn** : export vers `CASABLANCA05.npz` pour entraînement

## Métriques par boulevard

Pour chaque boulevard, le pipeline calcule :
- **Vitesse moyenne** (km/h)
- **Débit** (véhicules/heure)
- **Densité** (véh/km)
- **Taux d'occupation** (%)
- **État du trafic** : Congested (≤30), Moderate (≤60), Free Flow (>60)

## Entraînement T-GCN sur Casablanca

Après export :
```powershell
python train.py --dataset casablanca
python evaluate.py --dataset casablanca
```

Fichiers générés :
- `data/CASABLANCA05.npz` — matrice [T, 5] vitesses
- `data/CASABLANCA05.csv` — distances entre boulevards

## Lien avec PEMS07 (New York)

Le dataset PEMS07 (883 capteurs, New York) sert de **référence méthodologique** :
- Même format de données `[T, N]`
- Même fenêtre temporelle (seq_len=12, pred_len=12)
- Même pipeline T-GCN (graphe distance + corrélation)

Casablanca remplace New York avec **N=5 boulevards** au lieu de 883 capteurs.

## Dépannage

| Problème | Solution |
|----------|----------|
| MongoDB inaccessible | `docker start mongo-tgcn` ou démarrer `mongod` |
| traci introuvable | Installer SUMO + `SUMO_HOME`, ou `--demo` |
| OSM timeout | Relancer `--step osm` ou télécharger manuellement |
| Aucune arête trouvée | Vérifier les noms OSM dans `simulation/config.py` |
