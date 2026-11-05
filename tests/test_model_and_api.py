from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.preprocessing import FEATURES


@pytest.fixture
def client(env):
    return TestClient(app)


def test_model_metrics_are_sane(env):
    m = env["metrics"]
    t = m["test"][m["selected_model"]]
    assert 0.0 <= t["brier"] <= 0.25 and t["roc_auc"] > 0.5
    assert set(m["features"]) == set(FEATURES)
    assert m["splits"]["train"]["end"] < m["splits"]["validation"]["start"] < m["splits"]["test"]["start"]
    assert m["data_origin"] == "synthetic_demo" and m["warning"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True, "database_ok": True,
                                                  "data_origin": "synthetic_demo"}


def test_stations(client):
    assert len(client.get("/stations").json()) == 3


def test_predict_historical_replay(client):
    r = client.get("/predict", params={"date": "2024-03-15", "station_id": "07130"})
    j = r.json()
    assert r.status_code == 200 and j["mode"] == "historical_replay"
    assert 0 <= j["probability"] <= 1 and j["features_date"] == "2024-03-14"
    assert j["rain_observed_mm"] is not None and any("SYNTHETIQUES" in w for w in j["warnings"])


def test_predict_next_day_forecast(client, env):
    d = date.fromisoformat(env["last_day"]) + timedelta(days=1)
    j = client.get("/predict", params={"date": d.isoformat(), "station_id": "07110"}).json()
    assert j["mode"] == "next_day_forecast" and j["features_date"] == env["last_day"] and j["horizon_days"] == 1


def test_predict_far_future_is_climatology(client, env):
    d = date.fromisoformat(env["last_day"]) + timedelta(days=10)
    j = client.get("/predict", params={"date": d.isoformat(), "station_id": "07205"}).json()
    assert j["mode"] == "climatology" and j["features_date"] is None and j["features_used"] is None


def test_predict_errors(client):
    assert client.get("/predict", params={"date": "2024-13-45", "station_id": "07130"}).status_code == 422
    assert client.get("/predict", params={"date": "1990-01-01", "station_id": "07130"}).status_code == 422
    assert client.get("/predict", params={"date": "2040-01-01", "station_id": "07130"}).status_code == 422
    assert client.get("/predict", params={"date": "2024-03-15", "station_id": "99999"}).status_code == 404
    assert client.get("/predict", params={"date": "2024-03-15", "station_id": "x'; DROP"}).status_code == 422
    assert client.get("/predict").status_code == 422


def test_history_and_model_info(client):
    h = client.get("/history", params={"station_id": "07130", "days": 14}).json()
    assert len(h["points"]) >= 10 and all(0 <= p["probability"] <= 1 for p in h["points"])
    assert client.get("/model").json()["algorithm"] in ("logistic", "hist_gb")


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("GOUTTE_API_KEY", "secret")
    assert client.get("/stations").status_code == 401
    assert client.get("/stations", headers={"X-API-Key": "mauvaise"}).status_code == 401
    assert client.get("/stations", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200  # supervision non protegee


def test_prediction_is_logged(env):
    from src import database as db
    c = db.get_connection(env["settings"].db_path)
    assert c.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0] >= 1
