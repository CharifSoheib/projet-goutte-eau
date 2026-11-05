"""Module de prediction reutilisable (API, interface, scripts).

Trois modes, volontairement distincts :
- next_day_forecast : target_date == dernier jour observe + 1 -> vraie prevision a J+1.
- historical_replay : target_date deja couverte par les donnees -> estimation faite avec les seules
  observations de la veille (rejeu ; ce n'est pas une prevision future).
- climatology       : horizon > 1 jour ou donnees insuffisantes -> frequence mensuelle historique.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import joblib
import pandas as pd

from . import database as db
from .config import Settings
from .preprocessing import FEATURES, build_features, hourly_to_daily

MIN_FEATURE_COMPLETENESS = 0.7


class ModelNotAvailable(RuntimeError):
    pass


class UnknownStation(KeyError):
    pass


@lru_cache(maxsize=4)
def _load_bundle_cached(path: str, mtime: float) -> dict:
    return joblib.load(path)


def load_bundle(settings: Settings) -> dict:
    p = settings.model_path
    if not p.exists():
        raise ModelNotAvailable(f"Modele introuvable : {p}. Lancer scripts/run_pipeline.py")
    return _load_bundle_cached(str(p), p.stat().st_mtime)  # joblib = pickle : charger uniquement un modele de confiance


def risk_level(prob: float, settings: Settings) -> str:
    lv = settings.region["risk_levels"]
    return "faible" if prob < lv["low_below"] else ("eleve" if prob >= lv["high_from"] else "modere")


def station_features(conn, settings: Settings, station_id: str, origin: str) -> pd.DataFrame:
    obs = db.load_observations(conn, origin=origin, station_ids=[station_id])
    if obs.empty:
        return pd.DataFrame()
    daily = hourly_to_daily(obs, settings.min_rain_hours)
    f = build_features(daily, settings.rain_threshold_mm)
    return f.set_index("feature_date")


def get_station(settings: Settings, station_id: str) -> dict:
    for s in settings.stations:
        if s["station_id"] == station_id:
            return s
    raise UnknownStation(station_id)


def predict_risk(conn, settings: Settings, bundle: dict, station_id: str, target_date: date) -> dict:
    station = get_station(settings, station_id)
    origin = bundle["data_origin"]
    feats = station_features(conn, settings, station_id, origin)
    warnings: list[str] = []
    if origin == "synthetic_demo":
        warnings.append("Donnees de demonstration SYNTHETIQUES : resultat sans valeur meteorologique reelle.")

    prev = pd.Timestamp(target_date) - pd.Timedelta(days=1)
    last_day = feats.index[feats["has_obs"]].max() if len(feats) and feats["has_obs"].any() else None
    td = pd.Timestamp(target_date)
    mode, prob, used, features_date, observed = "climatology", None, None, None, None

    usable = (
        last_day is not None and prev in feats.index and td <= last_day + pd.Timedelta(days=1)
        and feats.loc[prev, FEATURES].notna().mean() >= MIN_FEATURE_COMPLETENESS
    )
    if usable:
        row = feats.loc[[prev], FEATURES]
        prob = float(bundle["pipeline"].predict_proba(row)[0, 1])
        features_date = prev.date()
        used = {k: round(float(v), 2) for k, v in row.iloc[0].items() if pd.notna(v)
                and k not in ("doy_sin", "doy_cos")}
        if td <= last_day:
            mode = "historical_replay"
            warnings.append("Rejeu historique : estimation calculee avec les seules observations de la veille ; "
                            "ce n'est pas une prevision future.")
            if td in feats.index and pd.notna(feats.loc[td, "rain_today_mm"]):
                observed = float(feats.loc[td, "rain_today_mm"])
        else:
            mode = "next_day_forecast"
            warnings.append(f"Prevision a 1 jour fondee sur les observations jusqu'au {last_day.date().isoformat()}.")
    else:
        prob = float(bundle["climatology"][td.month])
        warnings.append("Horizon superieur a 1 jour ou observations insuffisantes : le MVP ne fournit que la "
                        "frequence climatologique mensuelle de jours pluvieux, pas une prevision.")
    return {
        "station_id": station_id, "station_name": station["name"], "target_date": target_date,
        "probability": round(prob, 4), "risk_level": risk_level(prob, settings),
        "rain_expected": bool(prob >= bundle["threshold"]), "rain_threshold_mm": bundle["rain_threshold_mm"],
        "mode": mode,
        "horizon_days": 1 if mode != "climatology" else (int((td - last_day).days) if last_day is not None and td > last_day else None),
        "features_date": features_date, "rain_observed_mm": observed, "data_origin": origin,
        "model_id": bundle["model_id"], "features_used": used, "warnings": warnings,
    }


def replay_history(conn, settings: Settings, bundle: dict, station_id: str, n_days: int = 30) -> list[dict]:
    """Probabilite estimee (a partir de la veille) vs pluie observee, sur les n derniers jours."""
    get_station(settings, station_id)
    feats = station_features(conn, settings, station_id, bundle["data_origin"])
    if feats.empty:
        return []
    ok = feats[FEATURES].notna().mean(axis=1) >= MIN_FEATURE_COMPLETENESS
    probs = pd.Series(index=feats.index, dtype=float)
    if ok.any():
        probs[ok] = bundle["pipeline"].predict_proba(feats.loc[ok, FEATURES])[:, 1]
    out = []
    for fd in feats.index[feats["has_obs"]][-n_days - 1:-1]:  # features du jour fd -> cible fd+1
        td = fd + pd.Timedelta(days=1)
        if pd.isna(probs.loc[fd]):
            continue
        obs = feats.loc[fd, "rain_next_mm"]
        out.append({"date": td.date(), "probability": round(float(probs.loc[fd]), 4),
                    "rain_observed_mm": None if pd.isna(obs) else float(obs),
                    "rain_observed": None if pd.isna(obs) else bool(obs >= bundle["rain_threshold_mm"])})
    return out
