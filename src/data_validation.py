"""Validation et nettoyage des observations (dates, types, unites, plages, doublons)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS_REQUIRED

# Plages physiquement plausibles (unites : degC, hPa, %, km/h, mm)
VALID_RANGES = {
    "temperature_c": (-40.0, 50.0),
    "pressure_hpa": (900.0, 1090.0),
    "humidity_pct": (0.0, 100.0),
    "wind_mean_kmh": (0.0, 250.0),
    "wind_gust_kmh": (0.0, 350.0),
    "rain_1h_mm": (0.0, 100.0),
}


class ValidationError(ValueError):
    """Schema invalide (colonnes manquantes)."""


def validate_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Retourne (dataframe nettoye, rapport de qualite).

    Regles : horodatage UTC parsable (sinon ligne rejetee) ; station renseignee ;
    valeurs numeriques hors plage -> NaN (comptees) ; doublons (station, heure) supprimes.
    Les valeurs manquantes sont conservees (NaN) et quantifiees.
    """
    missing = [c for c in FEATURE_COLUMNS_REQUIRED if c not in df.columns]
    if missing:
        raise ValidationError(f"Colonnes manquantes : {missing}")

    report: dict = {"rows_in": int(len(df))}
    out = df[FEATURE_COLUMNS_REQUIRED].copy()

    out["station_id"] = out["station_id"].astype("string").str.strip()
    no_station = out["station_id"].isna() | (out["station_id"] == "")
    ts = pd.to_datetime(out["obs_time_utc"], errors="coerce", utc=True, format="ISO8601")
    bad_dates = ts.isna()
    report["rows_without_station"] = int(no_station.sum())
    report["rows_invalid_date"] = int(bad_dates.sum())
    keep = ~(no_station | bad_dates)
    out = out.loc[keep].copy()
    out["obs_time_utc"] = ts.loc[keep].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_of_range: dict[str, int] = {}
    for col, (lo, hi) in VALID_RANGES.items():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
        out_of_range[col] = int(bad.sum())
        out.loc[bad, col] = np.nan
    report["values_out_of_range"] = out_of_range

    before = len(out)
    out = out.drop_duplicates(subset=["station_id", "obs_time_utc"], keep="last")
    report["duplicates_removed"] = int(before - len(out))

    out = out.sort_values(["station_id", "obs_time_utc"]).reset_index(drop=True)
    out["station_id"] = out["station_id"].astype(str)
    report["rows_out"] = int(len(out))
    report["missing_rate"] = {c: round(float(out[c].isna().mean()), 4) if len(out) else None
                              for c in VALID_RANGES}
    return out, report
