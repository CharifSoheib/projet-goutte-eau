import pytest

from src import database as db
from src.config import get_settings
from src.data_collection import collect_demo
from src.train_model import train


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    """Base + modele entraines sur un petit jeu SYNTHETIQUE, isoles dans un dossier temporaire."""
    import os
    d = tmp_path_factory.mktemp("goutte")
    os.environ["GOUTTE_DB_PATH"] = str(d / "test.db")
    os.environ["GOUTTE_MODEL_PATH"] = str(d / "model.joblib")
    os.environ["GOUTTE_METRICS_PATH"] = str(d / "metrics.json")
    os.environ["GOUTTE_FIGURES_DIR"] = str(d / "fig")
    os.environ.pop("GOUTTE_API_KEY", None)
    s = get_settings()
    conn = db.get_connection(s.db_path)
    db.init_db(conn)
    db.upsert_stations(conn, s.region["region"], s.stations)
    collect_demo(conn, s, "2023-01-01", "2024-06-30", seed=7)
    metrics = train(conn, s, "synthetic_demo", fast=True, make_figures=False)
    conn.close()
    return {"settings": s, "metrics": metrics, "last_day": "2024-06-30"}
