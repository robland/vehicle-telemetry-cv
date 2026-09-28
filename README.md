# Télémétrie Véhicule & Computer Vision

**Conception d'un système de télémétrie intelligente combinant Computer Vision, IoT et données véhicule, avec détection de passagers par YOLO11n et intégration progressive de données GPS, IMU, OBD-II, embeddings et flux MQTT/Kafka.**

> Projet personnel orienté **Edge AI, Computer Vision, IoT, Data Engineering et systèmes distribués**.

---

## 🎯 Objectif

Ce projet vise à concevoir progressivement une architecture capable de **monitorer une flotte de véhicules en temps réel** en combinant plusieurs sources de données :

* 📷 **Computer Vision** : détection et suivi des personnes ;
* 🧠 **Embeddings** : représentation vectorielle des personnes détectées ;
* 📍 **GPS** : position, vitesse, altitude et cap ;
* 📐 **IMU** : accéléromètre et gyroscope ;
* 🚗 **OBD-II** : données provenant du véhicule ;
* 📡 **MQTT** : transmission des données depuis le véhicule ;
* ⚡ **Kafka** : ingestion distribuée des flux ;
* 🔥 **Apache Flink** : traitement temps réel et analyse des événements.

L'architecture est conçue pour évoluer progressivement d'un **prototype mono-véhicule** vers une simulation de flotte de **1 → 10 → 100 → 1 000 véhicules**.

---

## 🏗️ Architecture cible

```text
                         ┌─────────────────────┐
                         │      Véhicule       │
                         │                     │
                         │  📷 Camera          │
                         │  📍 GPS             │
                         │  📐 IMU             │
                         │  🚗 OBD-II          │
                         └──────────┬──────────┘
                                    │
                                    │
                    ┌───────────────▼───────────────┐
                    │          Edge Layer           │
                    │                               │
                    │  YOLO11n                     │
                    │  Person Detection             │
                    │  Tracking                     │
                    │  Embeddings                   │
                    │  Data preprocessing            │
                    └───────────────┬───────────────┘
                                    │
                             MQTT / IoT
                                    │
                    ┌───────────────▼───────────────┐
                    │          Data Layer            │
                    │                               │
                    │           Kafka                │
                    │       Event Streaming          │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │       Stream Processing        │
                    │                               │
                    │        Apache Flink             │
                    │                               │
                    │  • Event processing             │
                    │  • Aggregation                  │
                    │  • Tracking                    │
                    │  • Anomaly detection            │
                    └───────────────┬───────────────┘
                                    │
                           ┌────────▼────────┐
                           │   Applications   │
                           │                  │
                           │ Monitoring       │
                           │ Analytics        │
                           │ Alerts            │
                           └──────────────────┘
```

---

## 🔐 Principe de confidentialité des données

Les **images issues de la caméra ne sont pas transmises au serveur**.

Le traitement Computer Vision est réalisé en amont afin d'extraire uniquement les informations nécessaires au système.

Le flux peut notamment contenir :

```text
Image
   │
   ▼
YOLO11n
   │
   ▼
Person Detection
   │
   ├── Bounding Box
   ├── Confidence
   ├── Track ID
   └── Embedding
             │
             ▼
        Données dérivées
             │
             ▼
          MQTT
```

L'objectif est donc de séparer :

**données brutes → traitement Edge → données dérivées → infrastructure distribuée**

---

# 🚧 État du projet

Le projet est développé progressivement.

### ✅ Implémenté

* [x] Lecture de vidéos enregistrées
* [x] Prétraitement des images
* [x] Letterboxing `1920×1080 → 640×640`
* [x] Détection de personnes avec **YOLO11n**
* [x] Conversion des coordonnées de bounding boxes
* [x] Normalisation des bounding boxes
* [x] Validation des objets métier
* [x] Filtrage des classes non pertinentes
* [x] Gestion de plusieurs personnes par image
* [x] Pipeline vidéo
* [x] Frame skipping
* [x] Calcul de métriques
* [x] Tracking initial avec `SimpleTracker`
* [x] Visualisation des tracks
* [x] Tests automatisés avec `pytest`

### 🚧 En développement / prévu

* [ ] Extraction d'embeddings
* [ ] Évaluation FMR / FNMR et seuil de similarité
* [ ] Tracking multi-personnes plus robuste
* [ ] Intégration GPS
* [ ] Intégration IMU
* [ ] Intégration OBD-II
* [ ] Génération d'événements télémétriques
* [ ] Communication MQTT
* [ ] Ingestion Kafka
* [ ] Traitement temps réel avec Apache Flink
* [ ] Simulation de plusieurs véhicules
* [ ] Passage à une flotte simulée 1 → 10 → 100 → 1 000 véhicules
* [ ] Monitoring et observabilité
* [ ] Déploiement conteneurisé

---

# 👁️ Computer Vision

La première version du système se concentre sur la détection et le suivi des passagers.

Le pipeline actuel suit le principe :

```text
Video
  │
  ▼
Frame extraction
  │
  ▼
Letterbox
  │
  ▼
YOLO11n
  │
  ▼
Person Detection
  │
  ▼
Bounding Box normalization
  │
  ▼
Tracking
  │
  ▼
Events / Metrics
```

## Prétraitement

Les images sources sont actuellement considérées avec une résolution de :

```text
1920 × 1080
```

Elles sont adaptées à l'entrée du modèle :

```text
640 × 640
```

avec conservation du ratio d'aspect grâce au **letterboxing**.

Les transformations appliquées permettent ensuite de reconvertir les coordonnées des bounding boxes vers le référentiel de l'image originale.

---

# 🧠 Person Detection

Le modèle utilisé dans le prototype est :

**YOLO11n**

La détection est limitée à la classe :

```text
person
```

Chaque détection est convertie vers un objet métier `PersonDetection`.

Exemple conceptuel :

```json
{
  "class_name": "person",
  "confidence": 0.91,
  "bbox": {
    "x_min": 0.32,
    "y_min": 0.18,
    "x_max": 0.51,
    "y_max": 0.73
  }
}
```

Les bounding boxes sont normalisées afin de rendre le format indépendant de la résolution d'image.

---

# 👤 Tracking

Une première couche de tracking permet d'associer un identifiant aux personnes détectées au fil des frames.

```text
Frame N

Person A ──────── Track ID 01
Person B ──────── Track ID 02


Frame N+1

Person A ──────── Track ID 01
Person B ──────── Track ID 02
Person C ──────── Track ID 03
```

L'identifiant de tracking pourra ensuite être associé à un embedding.

À terme :

```text
taxi_id + track_id
```

constituera l'identifiant logique d'une personne suivie dans le véhicule.

---

# 🧬 Embeddings

Une étape ultérieure du projet consiste à remplacer ou compléter le tracking géométrique par une représentation vectorielle des personnes.

Principe :

```text
Person Crop
     │
     ▼
Embedding Model
     │
     ▼
Vector
     │
     ▼
Similarity Search
```

Les dimensions envisagées sont configurables :

```text
32
64
128
512
1024
```

L'objectif est notamment d'étudier le compromis entre :

* dimension des vecteurs ;
* précision ;
* coût de calcul ;
* bande passante ;
* stockage ;
* temps d'inférence.

---

# 📡 Télémétrie

Le système est conçu pour intégrer plusieurs sources de données.

## Vision

```json
{
  "taxi_id": "taxi_001",
  "timestamp": "...",
  "event_type": "person_detected",
  "camera_id": "cabin",
  "track_id": "track_001",
  "bbox": {
    "x_min": 0.32,
    "y_min": 0.18,
    "x_max": 0.51,
    "y_max": 0.73
  },
  "embedding": [],
  "confidence": 0.91
}
```

## GPS

```json
{
  "taxi_id": "taxi_001",
  "timestamp": "...",
  "latitude": 48.8566,
  "longitude": 2.3522,
  "altitude": 35.0,
  "speed": 32.5,
  "heading": 180.0
}
```

## IMU

```json
{
  "taxi_id": "taxi_001",
  "timestamp": "...",
  "ax": 0.12,
  "ay": -0.04,
  "az": 9.81,
  "gx": 0.01,
  "gy": 0.03,
  "gz": -0.02
}
```

## OBD-II

Le système pourra également intégrer des données véhicule telles que :

```text
vehicle_speed
engine_rpm
engine_load
coolant_temperature
throttle_position
...
```

Les fréquences d'acquisition des différents capteurs sont configurables afin d'adapter le système aux contraintes de calcul, de stockage et de communication.

---

# 📦 Data Engineering

L'architecture cible repose sur une chaîne événementielle :

```text
Edge
 │
 │ MQTT
 ▼
Kafka
 │
 ▼
Apache Flink
 │
 ├── Filtering
 ├── Aggregation
 ├── Event detection
 ├── Correlation
 └── Real-time analytics
 │
 ▼
Storage / Monitoring / Applications
```

Cette architecture permettra notamment d'étudier le passage d'un système embarqué mono-véhicule vers une architecture distribuée capable de traiter un grand nombre de flux simultanément.

---

# 🧪 Tests

Le projet adopte une approche **test-driven / test-oriented** pour les composants critiques.

Les tests couvrent notamment :

* transformations d'images ;
* conversion des bounding boxes ;
* normalisation des coordonnées ;
* validation des objets `PersonDetection` ;
* filtrage des détections ;
* traitement de plusieurs personnes ;
* pipeline vidéo ;
* tracking.

Exécution :

```bash
pytest
```

---

# 🛠️ Stack technique

### Computer Vision

* Python
* YOLO11n
* OpenCV
* NumPy

### Data & AI

* Machine Learning
* Deep Learning
* Embeddings
* Vector similarity

### IoT & Streaming

* MQTT
* Apache Kafka
* Apache Flink

### Testing & Engineering

* Pytest
* Git
* Docker *(prévu)*

---

# 📁 Structure du projet

La structure évoluera avec l'intégration des différentes briques.

```text
vehicle-telemetry-cv/
│
├── src/
│   ├── vision/
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── preprocessing.py
│   │   └── ...
│   │
│   ├── telemetry/
│   │   ├── gps.py
│   │   ├── imu.py
│   │   ├── obd.py
│   │   └── ...
│   │
│   ├── embeddings/
│   │   └── ...
│   │
│   └── streaming/
│       ├── mqtt/
│       ├── kafka/
│       └── flink/
│
├── tests/
│   ├── vision/
│   ├── telemetry/
│   └── ...
│
├── configs/
│
├── notebooks/
│
├── data/
│   └── .gitkeep
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

---

# 🔬 Roadmap

## Phase 1 — Computer Vision

```text
Video
 ↓
YOLO11n
 ↓
Person Detection
 ↓
Tracking
```

**Statut : terminé**

---

## Phase 2 — Person Re-identification

```text
Detection
 ↓
Person Crop
 ↓
Embedding
 ↓
Similarity
 ↓
Identity Association
```

**Statut : en développement**

---

## Phase 3 — Vehicle Telemetry

```text
Camera ──┐
GPS ─────┤
IMU ─────┼──► Edge Data Layer
OBD-II ──┘
```

**Statut : prévu**

---

## Phase 4 — IoT & Streaming

```text
Vehicle
   ↓
 MQTT
   ↓
 Kafka
   ↓
Flink
```

**Statut : prévu**

---

## Phase 5 — Fleet Simulation

Le système sera progressivement testé avec :

```text
1 taxi
  ↓
10 taxis
  ↓
100 taxis
  ↓
1 000 taxis
```

L'objectif est d'étudier les problématiques de :

* volume de données ;
* fréquence des événements ;
* latence ;
* consommation réseau ;
* stockage ;
* scalabilité ;
* traitement distribué.

---

# 📊 Exemple d'événement global

À terme, plusieurs sources pourront être corrélées autour d'un même véhicule :

```text
                    TAXI_001
                       │
        ┌──────────────┼──────────────┐
        │              │              │
      Vision          GPS            IMU
        │              │              │
        │              │              │
   Passenger       Position       Acceleration
   Track ID        Speed          Gyroscope
        │              │              │
        └──────────────┼──────────────┘
                       │
                     OBD-II
                       │
                       ▼
                 Event Stream
                       │
                       ▼
                  MQTT / Kafka
                       │
                       ▼
                    Flink
```

Cette approche permettra ensuite de construire des événements métier combinant **contexte passager, localisation, dynamique du véhicule et données mécaniques**.

---

# 🎓 Objectifs techniques

Ce projet sert également de laboratoire pour mettre en pratique plusieurs compétences :

* Computer Vision ;
* Deep Learning ;
* Object Detection ;
* Object Tracking ;
* Representation Learning ;
* Edge AI ;
* IoT ;
* Event-driven architecture ;
* Data Engineering ;
* Stream Processing ;
* systèmes distribués ;
* tests automatisés ;
* industrialisation logicielle.

---

# 📌 Projet en évolution

Le projet est volontairement construit de manière incrémentale.

La première version se concentre sur une problématique maîtrisable :

> **détecter et suivre les personnes présentes dans un véhicule à partir d'une vidéo.**

L'architecture est ensuite conçue pour accueillir progressivement :

> **Computer Vision → Embeddings → GPS → IMU → OBD-II → MQTT → Kafka → Flink → Fleet Monitoring**

L'objectif n'est donc pas uniquement de construire un modèle de Computer Vision, mais de développer progressivement un **système complet de télémétrie intelligente et de traitement de données distribuées**.
