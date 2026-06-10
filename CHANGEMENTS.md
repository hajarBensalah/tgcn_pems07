# T-GCN PEMS07 — Nouveautés (graphe corrélation + XAI)

Ce document décrit les changements ajoutés au projet pour répondre aux remarques du professeur :

- **Niveau 1** : graphe basé sur la distance physique (baseline existante)
- **Niveau 2** : graphe basé sur la corrélation du trafic
- **Propagation dynamique** : visualisation de l’évolution des liens trafic dans le temps
- **Explicabilité (XAI)** : analyse par ablation des voisins

---

## Résumé des changements

### Fichiers modifiés

| Fichier | Changement |
|---|---|
| `config.py` | Paramètres pour le graphe corrélation et l’XAI |
| `utils/graph_utils.py` | Construction des graphes distance + corrélation, comparaison, ablation |
| `utils/__init__.py` | Export des nouvelles fonctions |
| `train.py` | Option `--graph distance\|correlation` |
| `evaluate.py` | Comparaison automatique des deux graphes |

### Fichier ajouté

| Fichier | Rôle |
|---|---|
| `explain.py` | Script XAI : ablation des voisins, heatmaps, corrélation dynamique |

---

## Nouveaux paramètres (`config.py`)

```python
CORRELATION_THRESHOLD = 0.5   # seuil |corrélation| pour garder une arête
GRAPH_TYPE = "distance"       # graphe par défaut à l'entraînement

XAI_SENSORS       = [5, 10, 50]   # capteurs analysés par explain.py
XAI_MAX_SAMPLES   = 200           # échantillons test pour l'ablation
XAI_TOP_NEIGHBORS = 15            # voisins affichés dans les graphiques
```

---

## Prérequis

Depuis le dossier du projet (`TGCN`) :

```powershell
pip install -r requirements.txt
```

Vérifiez que les données sont présentes :

- `data/PEMS07.npz`
- `data/PEMS07.csv`

---

## Comment lancer le projet

### Étape 1 — Entraîner le modèle

**Baseline (niveau 1) — graphe distance :**

```powershell
python train.py --graph distance
```

**Niveau 2 — graphe corrélation :**

```powershell
python train.py --graph correlation
```

Le meilleur modèle est sauvegardé dans :

```
outputs/checkpoints/best_model.pt
```

> Pour comparer les deux approches dans un rapport, entraînez les deux variantes et renommez les checkpoints si besoin, par exemple :
>
> ```powershell
> copy outputs\checkpoints\best_model.pt outputs\checkpoints\best_model_distance.pt
> python train.py --graph correlation
> copy outputs\checkpoints\best_model.pt outputs\checkpoints\best_model_correlation.pt
> ```

---

### Étape 2 — Évaluer le modèle

```powershell
python evaluate.py
```

Avec un checkpoint précis :

```powershell
python evaluate.py --checkpoint outputs/checkpoints/best_model.pt
```

Forcer un type de graphe (si le checkpoint ne le contient pas) :

```powershell
python evaluate.py --graph distance
python evaluate.py --graph correlation
```

**Sorties générées** dans `outputs/plots/` :

| Fichier | Description |
|---|---|
| `loss_curve.png` | Courbes train / validation |
| `pred_vs_true.png` | Prédictions vs réalité (capteurs 5, 10, 50) |
| `confusion_matrix.png` | Matrice de confusion (états de trafic) |
| `spatial_error.png` | Erreur MAE par capteur |
| `graph_comparison.png` | **Nouveau** — comparaison voisins distance vs corrélation |

Le terminal affiche aussi les statistiques de comparaison des graphes (Jaccard, arêtes partagées).

---

### Étape 3 — Analyse XAI (explicabilité)

```powershell
python explain.py
```

Avec options :

```powershell
python explain.py --checkpoint outputs/checkpoints/best_model.pt
python explain.py --sensors 5 10 50
```

**Sorties générées** dans `outputs/plots/xai/` :

| Fichier | Description |
|---|---|
| `graph_compare_sensor_*.png` | Voisins distance vs corrélation pour un capteur |
| `heatmap_distance.png` | Top voisins du graphe distance |
| `heatmap_correlation.png` | Top voisins du graphe corrélation |
| `neighbor_ablation.png` | **XAI** — impact de chaque voisin sur la prédiction |
| `dynamic_corr_*_*.png` | **Propagation dynamique** — corrélation roulante dans le temps |

---

## Workflow recommandé pour le rapport

```powershell
# 1. Baseline distance
python train.py --graph distance
python evaluate.py
python explain.py

# 2. Variante corrélation
python train.py --graph correlation
python evaluate.py --graph correlation
python explain.py --checkpoint outputs/checkpoints/best_model.pt
```

### Structure suggérée du rapport

1. **Méthode niveau 1** : graphe gaussien sur distances physiques (`PEMS07.csv`)
2. **Méthode niveau 2** : graphe de corrélation des vitesses (train set)
3. **Résultats** : tableau MAE / RMSE distance vs corrélation (`evaluate.py`)
4. **Comparaison des graphes** : `graph_comparison.png` + stats Jaccard
5. **XAI** : `neighbor_ablation.png` — quels capteurs influencent la prédiction
6. **Dynamique** : `dynamic_corr_*.png` — les liens trafic changent selon l’heure / la congestion

---

## Réponses aux questions du professeur

| Question | Réponse dans ce projet |
|---|---|
| *La distance physique, c’est le niveau 1 ?* | Oui → `python train.py --graph distance` |
| *Comment le trafic se propage dynamiquement ?* | Corrélation roulante dans `explain.py` (`dynamic_corr_*.png`) |
| *Où est l’explicabilité (XAI) ?* | Ablation des voisins dans `explain.py` (`neighbor_ablation.png`) |

---

## Détails techniques

### Graphe distance (niveau 1)

- Source : `data/PEMS07.csv` (colonnes `from`, `to`, `cost`)
- Formule : `W_ij = exp(−d_ij² / σ²)`
- Sparsification : `DISTANCE_THRESHOLD = 0.1`

### Graphe corrélation (niveau 2)

- Source : vitesses dans `data/PEMS07.npz` (split train uniquement)
- Formule : `W_ij = |corr(vitesse_i, vitesse_j)|`
- Sparsification : `CORRELATION_THRESHOLD = 0.5`

### Ablation XAI

Pour un capteur cible, chaque voisin est retiré du graphe un par un.  
Si le MAE augmente fortement, le modèle dépend beaucoup de ce voisin pour prédire le capteur cible.

---

## Dépannage

| Problème | Solution |
|---|---|
| `Checkpoint not found` | Lancer `python train.py` d’abord |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `PEMS07.npz` introuvable | Vérifier le dossier `data/` |
| Ablation trop lente | Réduire `XAI_MAX_SAMPLES` dans `config.py` |

---

## Arborescence des sorties

```
outputs/
├── checkpoints/
│   └── best_model.pt
└── plots/
    ├── loss_curve.png
    ├── pred_vs_true.png
    ├── confusion_matrix.png
    ├── spatial_error.png
    ├── graph_comparison.png
    └── xai/
        ├── graph_compare_sensor_5.png
        ├── heatmap_distance.png
        ├── heatmap_correlation.png
        ├── neighbor_ablation.png
        └── dynamic_corr_5_*.png
```
