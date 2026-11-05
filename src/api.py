"""API REST FastAPI - estimation du risque de pluie.

Lancement : uvicorn src.api:app --host 127.0.0.1 --port 8000
Documentation interactive : /docs (Swagger) et /redoc.
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
from datetime import date, timedelta

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from . import database as db
from .config import Settings, get_settings
from .predict import ModelNotAvailable, UnknownStation, load_bundle, predict_risk, replay_history
from .schemas import (HealthOut, HistoryPoint, HistoryResponse, ModelInfo, PredictionResponse, StationOut)

log = logging.getLogger("goutte.api")
MAX_FUTURE_DAYS = 366
MIN_DATE = date(2000, 1, 1)

app = FastAPI(
    title="Goutte d'Eau - API de risque de pluie (MVP)",
    version="0.1.0",
    description=("Estime la probabilite qu'une journee (UTC) cumule au moins le seuil de pluie defini, pour une "
                 "station de la region retenue. Prevision reelle uniquement a J+1 ; au-dela, climatologie."),
)


def settings_dep() -> Settings:
    return get_settings()


def conn_dep(settings: Settings = Depends(settings_dep)):
    conn = db.get_connection(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def auth_dep(x_api_key: str | None = Header(default=None), settings: Settings = Depends(settings_dep)) -> None:
    if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
        raise HTTPException(status_code=401, detail="Cle d'API manquante ou invalide (en-tete X-API-Key)")


def bundle_dep(settings: Settings = Depends(settings_dep)) -> dict:
    try:
        return load_bundle(settings)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health", response_model=HealthOut, tags=["supervision"])
def health(settings: Settings = Depends(settings_dep)) -> HealthOut:
    model_ok, origin = True, None
    try:
        origin = load_bundle(settings)["data_origin"]
    except ModelNotAvailable:
        model_ok = False
    try:
        c = db.get_connection(settings.db_path)
        c.execute("SELECT 1").fetchone()
        c.close()
        db_ok = settings.db_path.exists()
    except sqlite3.Error:
        db_ok = False
    return HealthOut(status="ok" if model_ok and db_ok else "degraded", model_loaded=model_ok,
                     database_ok=db_ok, data_origin=origin)


@app.get("/stations", response_model=list[StationOut], tags=["donnees"], dependencies=[Depends(auth_dep)])
def stations(settings: Settings = Depends(settings_dep)):
    region = settings.region["region"]
    return [StationOut(region=region, **{k: s[k] for k in ("station_id", "name", "latitude", "longitude")})
            for s in settings.stations]


@app.get("/predict", response_model=PredictionResponse, tags=["prediction"], dependencies=[Depends(auth_dep)],
         responses={404: {"description": "Station inconnue"}, 422: {"description": "Parametre invalide"},
                    503: {"description": "Modele indisponible"}})
def predict(date_: date = Query(..., alias="date", description="Date cible (AAAA-MM-JJ)"),
            station_id: str = Query(..., min_length=3, max_length=10, pattern=r"^[A-Za-z0-9]+$"),
            settings: Settings = Depends(settings_dep), conn=Depends(conn_dep), bundle: dict = Depends(bundle_dep)):
    if date_ < MIN_DATE or date_ > date.today() + timedelta(days=MAX_FUTURE_DAYS):
        raise HTTPException(422, f"Date hors plage ({MIN_DATE} .. aujourd'hui + {MAX_FUTURE_DAYS} jours)")
    try:
        res = predict_risk(conn, settings, bundle, station_id, date_)
    except UnknownStation:
        raise HTTPException(404, f"Station inconnue : {station_id}")
    db.log_prediction(conn, station_id, date_, res["probability"], res["mode"], res["model_id"])
    return res


@app.get("/history", response_model=HistoryResponse, tags=["prediction"], dependencies=[Depends(auth_dep)])
def history(station_id: str = Query(..., min_length=3, max_length=10, pattern=r"^[A-Za-z0-9]+$"),
            days: int = Query(30, ge=7, le=120), settings: Settings = Depends(settings_dep),
            conn=Depends(conn_dep), bundle: dict = Depends(bundle_dep)):
    try:
        pts = replay_history(conn, settings, bundle, station_id, days)
    except UnknownStation:
        raise HTTPException(404, f"Station inconnue : {station_id}")
    return HistoryResponse(station_id=station_id, data_origin=bundle["data_origin"],
                           points=[HistoryPoint(**p) for p in pts])


@app.get("/model", response_model=ModelInfo, tags=["supervision"], dependencies=[Depends(auth_dep)])
def model_info(settings: Settings = Depends(settings_dep), bundle: dict = Depends(bundle_dep)):
    import json
    metrics = None
    if settings.metrics_path.exists():
        m = json.loads(settings.metrics_path.read_text(encoding="utf-8"))
        metrics = m["test"].get(m["selected_model"])
    return ModelInfo(model_id=bundle["model_id"], algorithm=bundle["algorithm"], data_origin=bundle["data_origin"],
                     trained_at=bundle["trained_at"], rain_threshold_mm=bundle["rain_threshold_mm"],
                     decision_threshold=bundle["threshold"], test_metrics=metrics,
                     warning=("Modele entraine sur donnees synthetiques de demonstration"
                              if bundle["data_origin"] == "synthetic_demo" else None))
