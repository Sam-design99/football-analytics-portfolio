# ⚽ Football Analytics Portfolio

Plateforme complète d'analyse de données footballistiques construite en Python.  
Pipeline de données · Visualisations interactives · Analyse vidéo par computer vision.

---

## Aperçu

Ce projet a pour objectif de démontrer des compétences en **data analyse appliquée au sport professionnel**, à travers un dashboard interactif déployé en ligne et un module d'analyse vidéo par intelligence artificielle.

**Dashboard en ligne →** [football-analytics-portfolio.streamlit.app](https://football-analytics-portfolio.streamlit.app)  
**GitLab →** [gitlab.com/leumas78260/football-analytics-portfolio](https://gitlab.com/leumas78260/football-analytics-portfolio)

---

## Fonctionnalités

### Module 1 — Dashboard analytique ✅

- Sélection dynamique : compétition → saison → match
- **Shot map** interactif avec xG (taille des bulles = probabilité de but)
- **xG timeline** : évolution du xG cumulé des deux équipes minute par minute
- **Heatmap** de présence d'un joueur (Kernel Density Estimation)
- **Radar comparatif** : jusqu'à 3 joueurs sur 5 métriques normalisées
- **Réseau de passes** : nœuds = joueurs, épaisseur = fréquence de combinaison
- Tableaux de stats joueurs et équipes filtrables

### Module 2 — Analyse vidéo 🚧 (en cours)

- Téléchargement de vidéos YouTube via `yt-dlp`
- Détection des joueurs frame par frame avec **YOLO v8**
- Tracking persistant des joueurs avec **ByteTrack**
- Extraction automatique d'événements : sprint, tir, passe, dribble
- Export au format compatible StatsBomb pour recoupement des données

---

## Stack technique

| Catégorie | Outils |
|---|---|
| Données | StatsBomb Open Data · statsbombpy |
| Traitement | Python 3.12 · Pandas · NumPy · SciPy |
| Visualisation | Plotly · mplsoccer |
| Interface | Streamlit |
| Computer Vision | OpenCV · YOLO v8 (ultralytics) · ByteTrack (supervision) |
| Téléchargement vidéo | yt-dlp |
| Versionning | GitLab + GitHub (mirror) |
| Déploiement | Streamlit Community Cloud |

---

## Structure du projet

```
football-analytics-portfolio/
│
├── src/
│   ├── data_pipeline.py        # Collecte et traitement des données StatsBomb
│   ├── visualizations.py       # Graphiques Plotly (shot map, heatmap, radar...)
│   ├── video_downloader.py     # Téléchargement vidéos YouTube (yt-dlp)
│   ├── video_tracker.py        # Détection et tracking joueurs (YOLO + OpenCV)
│   └── event_extractor.py      # Extraction automatique d'événements
│
├── dashboard/
│   └── app.py                  # Application Streamlit principale
│
├── data/
│   ├── raw/                    # Données brutes (ignorées par git)
│   └── processed/              # Données traitées
│
├── notebooks/                  # Exploration et prototypage Jupyter
├── tests/                      # Tests unitaires
├── requirements.txt
├── packages.txt                # Packages système (Streamlit Cloud)
└── README.md
```

---

## Installation

```bash
# 1. Cloner le projet
git clone https://gitlab.com/leumas78260/football-analytics-portfolio.git
cd football-analytics-portfolio

# 2. Créer et activer l'environnement virtuel
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

Ouvre automatiquement [http://localhost:8501](http://localhost:8501)

---

## Données

Le projet utilise **StatsBomb Open Data** — données open source gratuites, sans clé API.  
Compétitions disponibles : FIFA World Cup 2018, Champions League, Bundesliga, et plus.

Les données sont téléchargées automatiquement au premier lancement via `statsbombpy`.

---

## Roadmap

- [x] Pipeline de données StatsBomb
- [x] Shot map · Heatmap · Radar · Réseau de passes · xG timeline
- [x] Dashboard Streamlit multi-onglets
- [x] Déploiement Streamlit Cloud
- [ ] Téléchargement vidéo YouTube
- [ ] Détection joueurs YOLO v8
- [ ] Tracking ByteTrack
- [ ] Extraction automatique d'événements
- [ ] Recoupement données vidéo + StatsBomb
- [ ] Tests unitaires
- [ ] Stats comparatives sur une saison entière

---

## Auteur

**Samuel** — Data Analyst passionné par le football et la data science sportive.

*Projet portfolio — avril 2025*