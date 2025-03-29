# YOLO Object Detection & Knowledge Graph Visualization

## Description

Ce projet utilise YOLO pour la détection d'objets dans les images et construit un graphe de connaissances à partir des concepts détectés en interrogeant ConceptNet et GraphDB.

## Structure du projet

```
.
├── images/                 # Dossier contenant les images d'arrière-plan
├── logs/                   # Dossier contenant les logs de l'application
├── models/                 # Dossier contenant le modèle YOLO
├── temp_images/            # Dossier temporaire pour stocker les images uploadées
├── main.py                 # Script principal : détection d'objets et traitement RDF
├── ui.py                   # Interface utilisateur avec Streamlit
├── requirements.txt        # Liste des dépendances Python
├── README.md               # Documentation du projet
```

## Installation

### Prérequis

- Python 3.8+
- [GraphDB](https://www.ontotext.com/products/graphdb/) en cours d'exécution sur `http://localhost:7200`
- Un modèle YOLO pré-entrainé (`yolo11x.pt`)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancement de l'application

```bash
streamlit run ui.py
```

### Fonctionnalités

1. **Détection d'objets** : Upload d'une image ou prise d'une photo avec la webcam.
2. **Génération de graphe de connaissances** : Extraction des relations entre concepts via ConceptNet et stockage dans GraphDB.
3. **Visualisation** : Affichage du graphe de connaissances et des descriptions RDF.

## Exemples

- **Détection d'objets** : Un téléphone est détecté, il est envoyé à ConceptNet pour récupérer les relations et intégré à GraphDB.
- **Visualisation RDF** : Le RDF associé aux concepts détectés peut être affiché et exporté.
