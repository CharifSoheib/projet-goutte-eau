"""Acces a la base SQLite : schema, insertion sans doublon, lecture."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    station_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    region       TEXT NOT NULL,
    latitude     REAL NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude    REAL NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    elevation_m  REAL
);
CREATE TABLE IF NOT EXISTS observations (
    station_id     TEXT NOT NULL REFERENCES stations(station_id),
    obs_time_utc   TEXT NOT NULL,
    temperature_c  REAL,
    pressure_hpa   REAL,
    humidity_pct   REAL,
    wind_mean_kmh  REAL,
    wind_gust_kmh  REAL,
    rain_1h_mm     REAL CHECK (rain_1h_mm IS NULL OR rain_1h_mm >= 0),
    data_origin    TEXT NOT NULL CHECK (data_origin IN ('infoclimat','synthetic_demo')),
    ingested_at    TEXT NOT NULL,
    PRIMARY KEY (station_id, obs_time_utc)
);
CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(obs_time_utc);
CREATE INDEX IF NOT EXISTS idx_obs_origin ON observations(data_origin);
CREATE TABLE IF NOT EXISTS collection_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    data_origin   TEXT NOT NULL,
    period_start  TEXT,
    period_end    TEXT,
    rows_received INTEGER,
    rows_valid    INTEGER,
    rows_inserted INTEGER,
    status        TEXT NOT NULL,
    message       TEXT
);
CREATE TABLE IF NOT EXISTS model_runs (
    model_id     TEXT PRIMARY KEY,
    trained_at   TEXT NOT NULL,
    data_origin  TEXT NOT NULL,
    algorithm    TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    model_path   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    requested_at TEXT NOT NULL,
    station_id   TEXT NOT NULL,
    target_date  TEXT NOT NULL,
    probability  REAL NOT NULL,
    mode         TEXT NOT NULL,
    model_id     TEXT
);
"""

OBS_COLUMNS = ["station_id", "obs_time_utc", "temperature_c", "pressure_hpa",
               "humidity_pct", "wind_mean_kmh", "wind_gust_kmh", "rain_1h_mm"]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_stations(conn: sqlite3.Connection, region: str, stations: list[dict]) -> None:
    conn.executemany(
        """INSERT INTO stations(station_id,name,region,latitude,longitude,elevation_m)
           VALUES(:station_id,:name,:region,:latitude,:longitude,:elevation_m)
           ON CONFLICT(station_id) DO UPDATE SET name=excluded.name, region=excluded.region,
           latitude=excluded.latitude, longitude=excluded.longitude, elevation_m=excluded.elevation_m""",
        [{**s, "region": region} for s in stations],
    )
    conn.commit()


def insert_observations(conn: sqlite3.Connection, df: pd.DataFrame, origin: str) -> int:
    """Insere les observations ; les doublons (station, horodatage) sont ignores.

    Retourne le nombre de lignes reellement inserees.
    """
    if df.empty:
        return 0
    sub = df[OBS_COLUMNS]
    rows = sub.astype(object).where(sub.notna(), None)
    ts = now_utc()
    before = conn.total_changes
    conn.executemany(
        """INSERT INTO observations
           (station_id,obs_time_utc,temperature_c,pressure_hpa,humidity_pct,
            wind_mean_kmh,wind_gust_kmh,rain_1h_mm,data_origin,ingested_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(station_id, obs_time_utc) DO NOTHING""",
        [tuple(r) + (origin, ts) for r in rows.itertuples(index=False, name=None)],
    )
    conn.commit()
    return conn.total_changes - before


def load_observations(conn: sqlite3.Connection, origin: str | None = None,
                      station_ids: list[str] | None = None) -> pd.DataFrame:
    sql = f"SELECT {','.join(OBS_COLUMNS)} FROM observations WHERE 1=1"
    params: list = []
    if origin:
        sql += " AND data_origin = ?"
        params.append(origin)
    if station_ids:
        sql += f" AND station_id IN ({','.join('?' * len(station_ids))})"
        params += station_ids
    sql += " ORDER BY station_id, obs_time_utc"
    return pd.read_sql_query(sql, conn, params=params)


def list_origins(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT data_origin, COUNT(*) n FROM observations GROUP BY data_origin").fetchall()
    return {r["data_origin"]: r["n"] for r in rows}


def log_collection_run(conn, **kw) -> int:
    cur = conn.execute(
        """INSERT INTO collection_runs(started_at,finished_at,data_origin,period_start,period_end,
           rows_received,rows_valid,rows_inserted,status,message)
           VALUES(:started_at,:finished_at,:data_origin,:period_start,:period_end,
           :rows_received,:rows_valid,:rows_inserted,:status,:message)""", kw)
    conn.commit()
    return cur.lastrowid


def log_model_run(conn, model_id, data_origin, algorithm, metrics: dict, model_path) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO model_runs VALUES(?,?,?,?,?,?)",
        (model_id, now_utc(), data_origin, algorithm, json.dumps(metrics, default=str), str(model_path)),
    )
    conn.commit()


def log_prediction(conn, station_id, target_date, probability, mode, model_id) -> None:
    conn.execute(
        "INSERT INTO prediction_log(requested_at,station_id,target_date,probability,mode,model_id) VALUES(?,?,?,?,?,?)",
        (now_utc(), station_id, str(target_date), float(probability), mode, model_id),
    )
    conn.commit()
