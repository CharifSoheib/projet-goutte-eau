"""Pipeline complet : init base -> collecte (demo ou reelle) -> entrainement -> evaluation.

Usage : python scripts/run_pipeline.py --mode demo|real [--fast]
"""
import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import database as db  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.data_collection import collect_demo, collect_real  # noqa: E402
from src.train_model import train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "real"], default="demo")
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    conn = db.get_connection(s.db_path)
    db.init_db(conn)
    db.upsert_stations(conn, s.region["region"], s.stations)
    if a.mode == "real":
        end = datetime.now(timezone.utc).date() - timedelta(days=1)
        rep = collect_real(conn, s, end - timedelta(days=3 * 365), end)
        origin = "infoclimat"
    else:
        rep = collect_demo(conn, s)
        origin = "synthetic_demo"
    print("Collecte :", rep)
    m = train(conn, s, origin, fast=a.fast)
    print("Modele :", m["selected_model"], "| test :", m["test"][m["selected_model"]])


if __name__ == "__main__":
    main()
