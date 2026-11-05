# Architecture (résumé)
Voir `docs/figures/architecture_generale.png`, `diagramme_composants.png`, `modele_donnees.png`, `gantt.png`.

- **Collecte** (`data_collection.py`) : client HTTP avec timeout, tentatives exponentielles, gestion 401/403/429/5xx, découpage en périodes de 30 jours, pause « fair use ».
- **Validation** (`data_validation.py`) : dates ISO 8601 UTC, types, plages physiques, doublons ; les valeurs aberrantes deviennent NaN et sont comptées.
- **Stockage** (`database.py`) : SQLite, clé primaire (station, heure UTC) → idempotence ; origine des données obligatoire.
- **Modèle** (`preprocessing.py`, `train_model.py`) : agrégation journalière, variables à J, cible à J+1, découpage chronologique avec purge.
- **Service** (`predict.py`, `api.py`) : trois modes explicites (prévision J+1, rejeu historique, climatologie).
- **Interface** (`app/streamlit_app.py`) : client de l'API uniquement.

## Endpoints
| Méthode | Chemin | Rôle |
|---|---|---|
| GET | /health | état modèle + base (non protégé) |
| GET | /stations | stations de la région |
| GET | /predict?date=&station_id= | risque de pluie |
| GET | /history?station_id=&days= | probabilité estimée vs pluie observée |
| GET | /model | métadonnées et métriques du modèle |
Codes : 200, 401 (clé), 404 (station), 422 (paramètre), 503 (modèle absent).
