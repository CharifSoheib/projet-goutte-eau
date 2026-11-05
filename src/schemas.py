"""Schemas Pydantic de l'API."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class StationOut(BaseModel):
    station_id: str
    name: str
    region: str
    latitude: float
    longitude: float


class PredictionResponse(BaseModel):
    station_id: str
    station_name: str
    target_date: date
    probability: float = Field(ge=0, le=1, description="Probabilite de pluie >= seuil (journee UTC)")
    risk_level: Literal["faible", "modere", "eleve"]
    rain_expected: bool = Field(description="Probabilite >= seuil de decision du modele")
    rain_threshold_mm: float
    mode: Literal["next_day_forecast", "historical_replay", "climatology"]
    horizon_days: Optional[int] = None
    features_date: Optional[date] = None
    rain_observed_mm: Optional[float] = None
    data_origin: str
    model_id: Optional[str] = None
    features_used: Optional[dict[str, float]] = None
    warnings: list[str] = []


class HistoryPoint(BaseModel):
    date: date
    probability: float
    rain_observed_mm: Optional[float] = None
    rain_observed: Optional[bool] = None


class HistoryResponse(BaseModel):
    station_id: str
    data_origin: str
    points: list[HistoryPoint]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    database_ok: bool
    data_origin: Optional[str] = None


class ModelInfo(BaseModel):
    model_id: str
    algorithm: str
    data_origin: str
    trained_at: str
    rain_threshold_mm: float
    decision_threshold: float
    test_metrics: Optional[dict] = None
    warning: Optional[str] = None


class ErrorOut(BaseModel):
    detail: str
