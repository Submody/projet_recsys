# 🏋️ FitMatch — Programmes Sportifs Personnalisés

## 📌 Présentation du Projet
FitMatch est un moteur de recommandation de programmes sportifs développé dans le cadre du **Sujet 11 — Programmes sportifs personnalisés**.

L'application prend en compte les contraintes physiques des utilisateurs (niveau de condition physique, matériel disponible, douleurs articulaires) et leurs objectifs pour recommander des exercices et générer un programme hebdomadaire sur-mesure.

### Key Features
- **5 Algorithmes & Modèle Hybride :** Baseline (Popularité bayésienne), Système expert par Règles/Profil, Filtrage par contenu (TF-IDF + Similarité Cosinus), Item-Item kNN, Factorisation Matricielle (ALS), et Modèle Hybride pondéré.
- **Filtres de sécurité & Démarrage à froid :** Prise en compte du matériel et des blessures, ainsi que la gestion des nouveaux profils et des nouveaux exercices.
- **Interface iOS Dark :** Démonstration interactive avec Streamlit (Glassmorphism, animations et graphiques Plotly).
- **Générateur hebdomadaire :** Export du programme d'entraînement au format `.txt`.

---

## 🛠️ Structure du Dépôt
```text
projet_recsys/
├── app.py              # Application principale Streamlit (Moteur, Données, EDA & UI)
├── FitMatch_EDA.ipynb   # Notebook récapitulatif (Analyse & Évaluation)
├── README.md            # Document d'instructions et de présentation
└── requirements.txt     # Dépendances Python