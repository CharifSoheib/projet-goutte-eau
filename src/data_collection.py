"""Collecte des donnees meteorologiques (API Infoclimat / StatIC + SYNOP) et mode demonstration.

Source reelle : https://www.infoclimat.fr/opendata/?method=get&format=json&stations[]=<id>
                &start=YYYY-MM-DD&end=YYYY-MM-DD&token=<TOKEN>
Contraintes documentees : token personnel obligatoire ; volume limite par requete (decoupage
en periodes de 30 jours) ; pas de limite stricte de debit mais "fair use" demande (pause entre requetes).

Usage : python -m src.data_collection --mode real|demo [--start YYYY-MM-DD --end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from .config import Settings, get_settings
from .data_validation import validate_observations
from . import database as db
from .synthetic_data import generate_synthetic_observations

log = logging.getLogger("goutte.collect")

# Correspondance champs API -> schema interne (unites : degC, hPa, %, km/h, mm)
FIELD_MAP = {
    "temperature": "temperature_c", "pression": "pressure_hpa", "humidite": "humidity_pct",
    "vent_moyen": "wind_mean_kmh", "vent_rafales": "wind_gust_kmh", "pluie_1h": "rain_1h_mm",
}


class CollectionError(RuntimeError):
    """Erreur non recuperable lors de la collecte."""


class MissingTokenError(CollectionError):
    pass


def parse_infoclimat_payload(payload) -> pd.DataFrame:
    """Transforme la reponse JSON en DataFrame au schema interne.

    Accepte 'hourly' sous forme de dict {station: [mesures]} ou de liste de mesures.
    Une reponse sans donnees horaires donne un DataFrame vide ; une structure
    inattendue leve CollectionError.
    """
    if not isinstance(payload, dict):
        raise CollectionError("Reponse JSON inattendue (objet attendu)")
    hourly = payload.get("hourly")
    if hourly is None:
        return pd.DataFrame(columns=["station_id", "obs_time_utc", *FIELD_MAP.values()])
    records: list[dict] = []
    if isinstance(hourly, dict):
        for st, items in hourly.items():
            for it in items or []:
                records.append({"id_station": it.get("id_station", st), **it})
    elif isinstance(hourly, list):
        records = [dict(it) for it in hourly]
    else:
        raise CollectionError("Champ 'hourly' de type inattendu")
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["station_id", "obs_time_utc", *FIELD_MAP.values()])
    if "dh_utc" not in df.columns or "id_station" not in df.columns:
        raise CollectionError("Champs 'dh_utc' / 'id_station' absents de la reponse")
    out = pd.DataFrame({"station_id": df["id_station"].astype(str),
                        "obs_time_utc": df["dh_utc"]})
    for src, dst in FIELD_MAP.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else float("nan")
    return out


class InfoclimatClient:
    def __init__(self, token: str | None, base_url: str, timeout: float = 30.0, max_retries: int = 4,
                 backoff: float = 2.0, min_interval: float = 1.0, chunk_days: int = 30,
                 client: httpx.Client | None = None):
        if not token:
            raise MissingTokenError("INFOCLIMAT_TOKEN absent : definir la variable d'environnement (voir .env.example)")
        self.token, self.base_url = token, base_url
        self.max_retries, self.backoff, self.min_interval = max_retries, backoff, min_interval
        self.chunk_days = chunk_days
        self.client = client or httpx.Client(timeout=timeout)

    def _get(self, params: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.get(self.base_url, params=params)
                if resp.status_code in (401, 403):
                    raise CollectionError(f"Acces refuse (HTTP {resp.status_code}) : token invalide ou adresse IP non autorisee")
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = float(resp.headers.get("Retry-After", self.backoff * (2 ** attempt)))
                    last_exc = CollectionError(f"HTTP {resp.status_code}")
                    log.warning("HTTP %s, nouvel essai dans %.1fs", resp.status_code, wait)
                    time.sleep(min(wait, 60))
                    continue
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError as exc:
                    raise CollectionError("Reponse non JSON") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                log.warning("Erreur reseau (%s), essai %d/%d", exc.__class__.__name__, attempt + 1, self.max_retries + 1)
                time.sleep(self.backoff * (2 ** attempt))
        raise CollectionError(f"Echec apres {self.max_retries + 1} tentatives : {last_exc}")

    def fetch(self, station_ids: list[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=self.chunk_days - 1), end)
            params = {"method": "get", "format": "json", "stations[]": station_ids,
                      "start": cur.isoformat(), "end": chunk_end.isoformat(), "token": self.token}
            frames.append(parse_infoclimat_payload(self._get(params)))
            cur = chunk_end + timedelta(days=1)
            if cur <= end:
                time.sleep(self.min_interval)  # fair use
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _record(conn, origin, started, start, end, received, valid, inserted, status, message=""):
    db.log_collection_run(conn, started_at=started, finished_at=db.now_utc(), data_origin=origin,
                          period_start=str(start), period_end=str(end), rows_received=received,
                          rows_valid=valid, rows_inserted=inserted, status=status, message=message)


def collect_real(conn, settings: Settings, start: date, end: date,
                 client: InfoclimatClient | None = None) -> dict:
    started = db.now_utc()
    try:
        client = client or InfoclimatClient(settings.infoclimat_token, settings.infoclimat_url)
        raw = client.fetch([s["station_id"] for s in settings.stations], start, end)
        clean, report = validate_observations(raw) if len(raw) else (raw, {"rows_in": 0, "rows_out": 0})
        inserted = db.insert_observations(conn, clean, "infoclimat") if len(clean) else 0
        _record(conn, "infoclimat", started, start, end, len(raw), len(clean), inserted, "ok")
        return {**report, "rows_inserted": inserted, "origin": "infoclimat"}
    except CollectionError as exc:
        _record(conn, "infoclimat", started, start, end, 0, 0, 0, "error", str(exc))
        raise


def collect_demo(conn, settings: Settings, start: str | None = None, end: str | None = None,
                 seed: int | None = None) -> dict:
    demo = settings.region["demo"]
    start, end, seed = start or demo["start"], end or demo["end"], seed if seed is not None else demo["seed"]
    started = db.now_utc()
    raw = generate_synthetic_observations(settings.stations, start, end, seed)
    clean, report = validate_observations(raw)
    inserted = db.insert_observations(conn, clean, "synthetic_demo")
    _record(conn, "synthetic_demo", started, start, end, len(raw), len(clean), inserted, "ok",
            "DONNEES SYNTHETIQUES - non meteorologiques reelles")
    return {**report, "rows_inserted": inserted, "origin": "synthetic_demo"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["real", "demo"], default="demo")
    p.add_argument("--start")
    p.add_argument("--end")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    conn = db.get_connection(settings.db_path)
    db.init_db(conn)
    db.upsert_stations(conn, settings.region["region"], settings.stations)
    if a.mode == "real":
        end = date.fromisoformat(a.end) if a.end else datetime.now(timezone.utc).date() - timedelta(days=1)
        start = date.fromisoformat(a.start) if a.start else end - timedelta(days=3 * 365)
        result = collect_real(conn, settings, start, end)
    else:
        result = collect_demo(conn, settings, a.start, a.end)
    log.info("Resultat : %s", result)


if __name__ == "__main__":
    main()
