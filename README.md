# Projet Goutte d'Eau – MVP de risque de pluie (Bloc 2)

MVP Python : collecte météo (API Infoclimat) → base SQLite → préparation → modèle ML → API REST (FastAPI) → interface de test (Streamlit).
Région retenue : **Bretagne** (stations Rennes-Saint-Jacques 07130, Brest-Guipavas 07110, Lorient-Lann-Bihoué 07205).

> **Avertissement sur les données.** L'API Infoclimat exige un token personnel. Sans token, le projet fonctionne en **mode démo**
> sur des données **synthétiques** (`data_origin = synthetic_demo`), clairement signalées dans la base, l'API et l'interface.
> Les scores obtenus dans ce mode **ne mesurent aucune performance météorologique réelle**.
> Pour des résultats réels : renseigner `INFOCLIMAT_TOKEN` et lancer `python scripts/run_pipeline.py --mode real`.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # puis éditer si besoin (token, clé API)
```

## Utilisation
```bash
python scripts/run_pipeline.py --mode demo     # base + collecte démo + entraînement + évaluation (+ figures)
# python scripts/run_pipeline.py --mode real   # collecte réelle (INFOCLIMAT_TOKEN requis)
uvicorn src.api:app --port 8000                # API : http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py             # interface : http://localhost:8501
pytest -q                                      # tests
python scripts/make_diagrams.py                # régénère les diagrammes de docs/figures
```

## Exemple d'appel
`GET /predict?date=2025-01-01&station_id=07130` → probabilité, niveau de risque, mode (`next_day_forecast`, `historical_replay`, `climatology`), avertissements.

## Principe de la cible (important)
Cible = « le cumul de pluie du jour **D+1** (UTC) atteint ≥ 1 mm », prédite avec les seules observations jusqu'au jour **D**.
- Vraie prévision : **J+1 uniquement** (lendemain de la dernière observation en base).
- Dates déjà couvertes par la base : *rejeu historique* (estimation faite avec les observations de la veille).
- Au-delà de J+1 : **climatologie mensuelle** (pas une prévision).

## Structure
```
config/region.json   paramètres (région, stations, seuil, découpage)   src/            code (collecte, validation, base, ML, API)
app/                 interface Streamlit                               tests/          31 tests pytest
models/              modèle .joblib + metrics.json                     docs/figures/   diagrammes et graphiques
scripts/             pipeline, lancement, diagrammes                   database/       base SQLite (non versionnée)
```
Documentation détaillée : voir le rapport (`Rapport_Projet_Goutte_Eau_BLOC_2.docx/.pdf`) et `docs/`.

## Sécurité
Aucun secret dans le dépôt (`.env` ignoré). Requêtes SQL paramétrées, validation Pydantic, clé d'API optionnelle (`X-API-Key`).
Le modèle `.joblib` est un fichier *pickle* : ne charger que des modèles produits par ce projet.

## Limites connues
Données réelles non collectées dans l'environnement de développement (réseau et token) ; format JSON Infoclimat à valider avec un token ;
découpage test = 2ᵉ semestre uniquement ; pas de prévision > J+1 ; une seule région.
