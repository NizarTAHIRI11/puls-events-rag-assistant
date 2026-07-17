# Assistant de recommandation d'événements culturels - Système RAG

## Présentation

Ce projet a été réalisé dans le cadre du parcours **Data Engineer** d'OpenClassrooms.

L'objectif est de développer un **Proof of Concept (POC)** d'un assistant intelligent capable de recommander des événements culturels à partir des données publiques de **OpenAgenda**.

Le système repose sur une architecture **Retrieval-Augmented Generation (RAG)** combinant :

- OpenAgenda API pour la collecte des événements
- Mistral Embeddings pour la vectorisation des textes
- FAISS comme base de données vectorielle
- LangChain pour l'orchestration du pipeline RAG
- Mistral LLM pour la génération des réponses

---

# Objectifs

Le système permet de :

- récupérer automatiquement les événements depuis OpenAgenda ;
- nettoyer et filtrer les données ;
- récupérer et traiter les événements de la ou des villes sélectionnées (Paris et Amiens dans le cadre du POC) ;
- supprimer les doublons ;
- conserver uniquement les événements récents (moins d'un an) ainsi que les événements à venir ;
- construire une base vectorielle FAISS ;
- répondre aux questions des utilisateurs grâce à un pipeline RAG.

---

# Prérequis

- Python 3.12 ou supérieur
- Compte OpenAgenda
- Clé API OpenAgenda
- Clé API Mistral

---

# Architecture du projet

```
Mission/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── src/
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── build_vectorstore.py
│   ├── rag_chain.py
│   ├── chatbot.py
│   └── __init__.py
│
├── tests/
│   ├── test_collect_data.py
│   ├── test_preprocess.py
│   ├── test_build_vectorstore.py
│   └── test_rag_chain.py
│
├── vectorstore/
├── Presentation/
│
├── README.md
├── requirements.txt
├── .gitignore
└── .env
```

Le pipeline est composé de deux phases principales :

- une phase d'indexation exécutée à la demande pour construire la base vectorielle ;
- une phase de requête exécutée à chaque question de l'utilisateur.

---

# Description des principaux scripts

## collect_data.py

Récupère les agendas et les événements depuis l'API OpenAgenda.

Fonctionnalités :

- récupération des agendas ;
- pagination automatique ;
- récupération des événements ;
- suppression des doublons.

---

## preprocess.py

Nettoie les données avant leur indexation.

Traitements effectués :

- suppression des doublons ;
- filtrage géographique ;
- filtrage temporel (événements de moins d'un an et événements futurs) ;
- sauvegarde des fichiers CSV et Parquet.

---

## build_vectorstore.py

Construit la base vectorielle.

Étapes :

- chargement des événements nettoyés ;
- création des embeddings avec Mistral ;
- création de l'index FAISS ;
- sauvegarde de la base vectorielle.

---

## rag_chain.py

Construit le pipeline RAG.

Le pipeline :

1. recherche des événements similaires avec FAISS ;
2. construction du prompt avec LangChain ;
3. génération de la réponse avec Mistral.

---

## chatbot.py

Interface permettant d'interroger le système.

L'utilisateur pose une question en langage naturel et reçoit une réponse contextualisée.

---

# Technologies utilisées

- Python
- Pandas
- OpenAgenda API
- LangChain
- FAISS
- Mistral Embeddings
- Mistral LLM
- Pytest

---

# Installation

Créer un environnement virtuel :

```bash
python -m venv venv
```

## Activation

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

# Variables d'environnement

Créer un fichier `.env` contenant :

```text
OPEN_AGENDA_API=xxxxxxxxxxxxxxxx
MISTRAL_API_KEY=xxxxxxxxxxxxxxxx
```

⚠️ Le fichier `.env` contient des clés API sensibles. Il est ignoré par Git grâce au fichier `.gitignore` et ne doit jamais être partagé ni versionné.

---

# Exécution du projet

## 1. Collecte des événements

```bash
python -m src.collect_data
```

---

## 2. Prétraitement

```bash
python -m src.preprocess
```

---

## 3. Construction de la base vectorielle

```bash
python -m src.build_vectorstore
```

---

## 4. Lancement du chatbot

```bash
python -m src.chatbot
```

---

# Exemple d'utilisation

```text
Question

Quels événements sont prévus à Paris le 20 août 2026 ?

Réponse

Le Grand Jeu 2026

Lieu : Cité des sciences et de l'Industrie

Date : 20 août 2026

Description :
Activités ludiques, sportives et pédagogiques gratuites destinées aux enfants, adolescents et familles.
```

---

# Exécution des tests

Lancer tous les tests :

```bash
pytest -v
```

Les tests vérifient notamment :

- la récupération des données ;
- le nettoyage des événements ;
- la construction de la base vectorielle ;
- le fonctionnement du pipeline RAG.

---

# Données générées

Les données sont enregistrées dans le dossier `data`.

### raw/

- événements bruts au format CSV
- événements bruts au format Parquet

### processed/

- événements nettoyés au format CSV
- événements nettoyés au format Parquet

### evaluation/

- jeux de données utilisés pour l'évaluation du système.

---

# Résultats

Le Proof of Concept permet :

- la collecte automatique des événements OpenAgenda ;
- le nettoyage et la préparation des données ;
- la construction d'une base vectorielle FAISS ;
- la recherche sémantique des événements ;
- la génération de réponses contextualisées avec Mistral.

---

# Perspectives

Les principales améliorations envisagées sont :

- ajout de nouvelles villes et extension de la couverture géographique ;
- mise à jour automatique de la base vectorielle ;
- enrichissement du jeu de données d'évaluation ;
- ajout d'un historique conversationnel ;
- développement d'une interface web plus avancée.

---

# Auteur

Nizar Tahiri

Projet réalisé dans le cadre du parcours **Data Engineer** – OpenClassrooms.
