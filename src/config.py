"""Configuration centralisee : chemins, variables d'environnement, parametres metier."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS_REQUIRED = [
    "station_id", "obs_time_utc", "temperature_c", "pressure_hpa",
    "humidity_pct", "wind_mean_kmh", "wind_gust_kmh", "rain_1h_mm",
]
DATA_ORIGINS = ("infoclimat", "synthetic_demo")


def _load_dotenv() -> None:
    """Lecture minimale d'un fichier .env (sans dependance externe)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model_path: Path
    metrics_path: Path
    figures_dir: Path
    infoclimat_token: str | None
    infoclimat_url: str
    api_key: str | None
    region: dict

    @property
    def stations(self) -> list[dict]:
        return self.region["stations"]

    @property
    def rain_threshold_mm(self) -> float:
        return float(self.region["target"]["rain_threshold_mm"])

    @property
    def min_rain_hours(self) -> int:
        return int(self.region["target"]["min_rain_hours_per_day"])


def get_settings() -> Settings:
    """Relit l'environnement a chaque appel (pratique pour les tests)."""
    _load_dotenv()
    region = json.loads((ROOT / "config" / "region.json").read_text(encoding="utf-8"))
    return Settings(
        db_path=_resolve(os.getenv("GOUTTE_DB_PATH", "database/goutte_eau.db")),
        model_path=_resolve(os.getenv("GOUTTE_MODEL_PATH", "models/rain_model.joblib")),
        metrics_path=_resolve(os.getenv("GOUTTE_METRICS_PATH", "models/metrics.json")),
        figures_dir=_resolve(os.getenv("GOUTTE_FIGURES_DIR", "docs/figures")),
        infoclimat_token=os.getenv("INFOCLIMAT_TOKEN") or None,
        infoclimat_url=os.getenv("INFOCLIMAT_URL", "https://www.infoclimat.fr/opendata/"),
        api_key=os.getenv("GOUTTE_API_KEY") or None,
        region=region,
    )
