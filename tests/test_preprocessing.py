import numpy as np
import pandas as pd

from src.preprocessing import FEATURES, build_features, hourly_to_daily, make_splits


def _hourly(days=40, rain_by_day=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d, day in enumerate(pd.date_range("2024-01-01", periods=days)):
        for h in range(24):
            rows.append({"station_id": "07130", "obs_time_utc": (day + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "temperature_c": 8 + rng.normal(), "pressure_hpa": 1010 + rng.normal(),
                         "humidity_pct": 80.0, "wind_mean_kmh": 10.0, "wind_gust_kmh": 20.0,
                         "rain_1h_mm": (rain_by_day or {}).get(d, 0.0) if h == 5 else 0.0})
    return pd.DataFrame(rows)


def test_daily_aggregation_and_rain_validity():
    obs = _hourly(3, {1: 4.0})
    obs.loc[(obs.obs_time_utc.str.startswith("2024-01-03")) & (obs.index % 24 < 10), "rain_1h_mm"] = np.nan
    daily = hourly_to_daily(obs, 18)
    assert daily.loc[1, "rain_sum"] == 4.0 and daily.loc[0, "rain_sum"] == 0.0
    assert np.isnan(daily.loc[2, "rain_sum"])  # trop d'heures manquantes -> cumul inconnu


def test_target_is_next_day_rain():
    feats = build_features(hourly_to_daily(_hourly(10, {4: 5.0})), 1.0).set_index("feature_date")
    assert feats.loc["2024-01-04", "target"] == 1.0      # jour 3 (index) -> pluie le 5 janvier (d=4)
    assert feats.loc["2024-01-05", "target"] == 0.0
    assert feats.loc["2024-01-10", "target"] != feats.loc["2024-01-10", "target"]  # dernier jour : cible inconnue


def test_no_leakage_features_ignore_future():
    """Modifier toutes les observations apres le jour D ne doit pas changer les variables du jour D."""
    obs = _hourly(30, {10: 3.0, 12: 2.0})  # pas de pluie le 12 janvier (d=11)
    D = pd.Timestamp("2024-01-11")
    base = build_features(hourly_to_daily(obs), 1.0).set_index("feature_date")
    mod = obs.copy()
    future = pd.to_datetime(mod["obs_time_utc"]).dt.tz_localize(None) >= D + pd.Timedelta(days=1)
    mod.loc[future, ["temperature_c", "pressure_hpa", "humidity_pct", "rain_1h_mm"]] = [30.0, 1060.0, 10.0, 50.0]
    alt = build_features(hourly_to_daily(mod), 1.0).set_index("feature_date")
    pd.testing.assert_series_equal(base.loc[D, FEATURES], alt.loc[D, FEATURES], check_names=False)
    assert base.loc[D, "target"] != alt.loc[D, "target"]  # seule la cible change


def test_chronological_split_with_purge():
    feats = build_features(hourly_to_daily(_hourly(200, {i: 3.0 for i in range(0, 200, 4)})), 1.0)
    m = make_splits(feats, 0.7, 0.15, 1)
    tr, va, te = (feats[m[k]]["feature_date"] for k in ("train", "validation", "test"))
    assert tr.max() < va.min() - pd.Timedelta(days=1) + pd.Timedelta(seconds=1)
    assert va.max() < te.min() - pd.Timedelta(days=1) + pd.Timedelta(seconds=1)
    assert not (m["train"] & m["validation"]).any() and not (m["validation"] & m["test"]).any()
