import sqlite3

import pandas as pd
import pytest

from src import database as db


@pytest.fixture
def conn(tmp_path):
    c = db.get_connection(tmp_path / "t.db")
    db.init_db(c)
    db.init_db(c)  # idempotent
    db.upsert_stations(c, "Bretagne", [{"station_id": "07130", "name": "R", "latitude": 48, "longitude": -1.7, "elevation_m": 30}])
    return c


def _obs(n=3):
    return pd.DataFrame({"station_id": "07130",
                         "obs_time_utc": [f"2024-01-01T0{i}:00:00Z" for i in range(n)],
                         "temperature_c": 5.0, "pressure_hpa": 1010.0, "humidity_pct": 80.0,
                         "wind_mean_kmh": 10.0, "wind_gust_kmh": 20.0, "rain_1h_mm": [0.0, None, 1.2][:n]})


def test_insert_is_idempotent(conn):
    assert db.insert_observations(conn, _obs(), "synthetic_demo") == 3
    assert db.insert_observations(conn, _obs(), "synthetic_demo") == 0
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3


def test_null_preserved(conn):
    db.insert_observations(conn, _obs(), "synthetic_demo")
    assert conn.execute("SELECT rain_1h_mm FROM observations WHERE obs_time_utc='2024-01-01T01:00:00Z'").fetchone()[0] is None


def test_constraints(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_observations(conn, _obs(), "origine_inconnue")
    bad = _obs().assign(station_id="99999", obs_time_utc="2025-01-01T00:00:00Z")
    with pytest.raises(sqlite3.IntegrityError):  # cle etrangere vers stations
        db.insert_observations(conn, bad, "synthetic_demo")


def test_load_filters_by_origin(conn):
    db.insert_observations(conn, _obs(), "synthetic_demo")
    assert len(db.load_observations(conn, origin="infoclimat")) == 0
    assert len(db.load_observations(conn, origin="synthetic_demo")) == 3
