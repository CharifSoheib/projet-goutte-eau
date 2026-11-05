"""Generateur de donnees METEOROLOGIQUES SYNTHETIQUES (mode demonstration).

ATTENTION : ces donnees ne sont PAS des observations reelles. Elles servent uniquement
a rendre le MVP reproductible hors ligne. Elles sont enregistrees avec
data_origin = 'synthetic_demo' et ne doivent jamais etre presentees comme reelles.
Les relations pluie <-> pression/humidite sont posees par construction : les scores
obtenus sur ces donnees ne renseignent en rien la performance sur des donnees reelles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _ar1(rng, n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return x


def generate_synthetic_observations(stations: list[dict], start: str, end: str, seed: int = 42,
                                    add_quality_issues: bool = True) -> pd.DataFrame:
    """Retourne un DataFrame horaire au schema standard (voir config.FEATURE_COLUMNS_REQUIRED)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n = len(days)
    doy = days.dayofyear.to_numpy()
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25)  # +1 mi-janvier, -1 mi-juillet
    common_anom = _ar1(rng, n, 0.75, 6.0)  # anomalie de pression commune a la region
    frames = []
    for k, st in enumerate(stations):
        anom = common_anom + _ar1(rng, n, 0.5, 1.5)
        base_h = 75 + 5 * winter - 0.6 * anom + rng.normal(0, 5, n)
        temp_mean = 11 - 6.5 * winter + 0.5 * k + _ar1(rng, n, 0.7, 2.0)
        P = 1015 + anom
        H = base_h.copy()
        rain = np.zeros(n, dtype=bool)
        amount = np.zeros(n)
        for t in range(2, n):
            logit = (-1.35 + 0.6 * winter[t] + 0.09 * (-(P[t - 1] - P[t - 2]))
                     + 0.045 * (H[t - 1] - 75) + 0.7 * rain[t - 1] - 0.03 * anom[t - 1])
            rain[t] = rng.random() < _sigmoid(logit)
            if rain[t]:
                amount[t] = 1.0 + rng.gamma(0.9, 4.0)
            elif rng.random() < 0.08:
                amount[t] = rng.uniform(0.1, 0.9)  # bruine sous le seuil de 1 mm
            P[t] = 1015 + anom[t] - 2.0 * rain[t]
            H[t] = np.clip(base_h[t] + 8 * rain[t], 35, 100)
        wind_day = 12 + rng.gamma(2.0, 2.0, n) + 5 * rain
        hours = np.arange(24)
        rows = []
        for t, day in enumerate(days):
            ts = [(day + pd.Timedelta(hours=int(h))).strftime("%Y-%m-%dT%H:%M:%SZ") for h in hours]
            temp = temp_mean[t] + 3 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.4, 24)
            pres = P[t] + 0.5 * np.sin(2 * np.pi * (hours - 4) / 12) + rng.normal(0, 0.2, 24)
            hum = np.clip(H[t] - 8 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 2, 24), 20, 100)
            wmean = np.clip(wind_day[t] + rng.normal(0, 2.5, 24), 0, None)
            gust = wmean * 1.5 + np.abs(rng.normal(0, 3, 24))
            r1h = np.zeros(24)
            if amount[t] > 0:
                dur = int(rng.integers(2, 10)) if amount[t] >= 1 else int(rng.integers(1, 3))
                start_h = int(rng.integers(0, 24 - dur + 1))
                w = rng.dirichlet(np.ones(dur))
                r1h[start_h:start_h + dur] = np.round(amount[t] * w, 1)
            rows.append(pd.DataFrame({
                "station_id": st["station_id"], "obs_time_utc": ts,
                "temperature_c": np.round(temp, 1), "pressure_hpa": np.round(pres, 1),
                "humidity_pct": np.round(hum, 0), "wind_mean_kmh": np.round(wmean, 1),
                "wind_gust_kmh": np.round(gust, 1), "rain_1h_mm": r1h,
            }))
        frames.append(pd.concat(rows, ignore_index=True))
    df = pd.concat(frames, ignore_index=True)

    if add_quality_issues:  # defauts volontaires pour exercer la validation
        cols = ["temperature_c", "pressure_hpa", "humidity_pct", "wind_mean_kmh", "wind_gust_kmh"]
        for c in cols:
            idx = rng.choice(len(df), size=int(0.015 * len(df)), replace=False)
            df.loc[idx, c] = np.nan
        idx = rng.choice(len(df), size=max(1, int(0.0003 * len(df))), replace=False)
        df.loc[idx, "temperature_c"] = 99.9
        idx = rng.choice(len(df), size=max(1, int(0.001 * len(df))), replace=False)
        df = pd.concat([df, df.loc[idx]], ignore_index=True)  # doublons
    return df
