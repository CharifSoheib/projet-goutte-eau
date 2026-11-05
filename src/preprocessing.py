"""Preparation : agregation horaire -> journaliere, variables explicatives, cible, decoupage temporel.

Principe anti-fuite : la ligne datee D ne contient que des informations observees jusqu'a la fin
du jour D (UTC). La cible porte sur le jour D+1 (pluie journaliere >= seuil).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "temp_mean", "temp_range", "pressure_mean", "pressure_delta_24h", "pressure_delta_72h",
    "humidity_mean", "wind_mean", "wind_gust_max", "rain_today_mm", "rain_3d_mm",
    "doy_sin", "doy_cos",
]


def hourly_to_daily(obs: pd.DataFrame, min_rain_hours: int = 18) -> pd.DataFrame:
    """Agrege les observations horaires par station et jour UTC.

    Le cumul de pluie n'est valide que si au moins `min_rain_hours` valeurs horaires sont presentes ;
    sinon il est mis a NaN (on ne suppose pas qu'une heure manquante = pas de pluie).
    """
    if obs.empty:
        return pd.DataFrame()
    df = obs.copy()
    df["date"] = pd.to_datetime(df["obs_time_utc"], utc=True).dt.tz_localize(None).dt.normalize()
    daily = df.groupby(["station_id", "date"]).agg(
        temp_mean=("temperature_c", "mean"), temp_min=("temperature_c", "min"),
        temp_max=("temperature_c", "max"), pressure_mean=("pressure_hpa", "mean"),
        humidity_mean=("humidity_pct", "mean"), wind_mean=("wind_mean_kmh", "mean"),
        wind_gust_max=("wind_gust_kmh", "max"), rain_sum=("rain_1h_mm", "sum"),
        rain_hours=("rain_1h_mm", "count"), n_obs=("obs_time_utc", "count"),
    ).reset_index()
    daily.loc[daily["rain_hours"] < min_rain_hours, "rain_sum"] = np.nan
    return daily


def _station_features(d: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = d.set_index("date").sort_index().asfreq("D")  # jours absents -> NaN explicites
    f = pd.DataFrame(index=d.index)
    f["temp_mean"] = d["temp_mean"]
    f["temp_range"] = d["temp_max"] - d["temp_min"]
    f["pressure_mean"] = d["pressure_mean"]
    f["pressure_delta_24h"] = d["pressure_mean"] - d["pressure_mean"].shift(1)
    f["pressure_delta_72h"] = d["pressure_mean"] - d["pressure_mean"].shift(3)
    f["humidity_mean"] = d["humidity_mean"]
    f["wind_mean"] = d["wind_mean"]
    f["wind_gust_max"] = d["wind_gust_max"]
    f["rain_today_mm"] = d["rain_sum"]
    f["rain_3d_mm"] = d["rain_sum"].rolling(3, min_periods=3).sum()
    doy = f.index.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    nxt = d["rain_sum"].shift(-1)  # pluie du jour D+1 : sert UNIQUEMENT a construire la cible
    f["rain_next_mm"] = nxt
    f["target"] = (nxt >= threshold).astype("float").where(nxt.notna())
    f["feature_date"] = f.index
    f["target_date"] = f.index + pd.Timedelta(days=1)
    f["has_obs"] = d["n_obs"].fillna(0) > 0
    return f


def build_features(daily: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Construit la table (station, feature_date) -> variables + cible (peut etre NaN)."""
    parts = []
    for st, d in daily.groupby("station_id"):
        f = _station_features(d, threshold)
        f.insert(0, "station_id", st)
        parts.append(f.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def make_splits(feats: pd.DataFrame, train: float = 0.70, validation: float = 0.15,
                purge_days: int = 1) -> dict[str, pd.Series]:
    """Decoupage chronologique (sur les dates, toutes stations confondues) avec purge d'un jour.

    La purge evite que la cible du dernier jour d'un ensemble (jour D+1) recouvre le premier jour
    de l'ensemble suivant.
    """
    labeled = feats[feats["target"].notna()]
    dates = np.sort(labeled["feature_date"].unique())
    n = len(dates)
    i1, i2 = int(n * train), int(n * (train + validation))
    gap = pd.Timedelta(days=purge_days)
    d_val_start, d_test_start = dates[i1] + gap, dates[i2] + gap
    fd = feats["feature_date"]
    ok = feats["target"].notna()
    return {
        "train": ok & (fd < dates[i1]),
        "validation": ok & (fd >= d_val_start) & (fd < dates[i2]),
        "test": ok & (fd >= d_test_start),
    }
