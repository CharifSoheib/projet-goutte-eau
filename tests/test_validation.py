import numpy as np
import pandas as pd
import pytest

from src.data_validation import ValidationError, validate_observations

COLS = ["station_id", "obs_time_utc", "temperature_c", "pressure_hpa", "humidity_pct",
        "wind_mean_kmh", "wind_gust_kmh", "rain_1h_mm"]


def _df(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_out_of_range_becomes_nan_and_is_counted():
    df = _df([["07130", "2024-01-01T00:00:00Z", 99.0, 1010, 80, 10, 20, 0.0],
              ["07130", "2024-01-01T01:00:00Z", 10.0, 1010, 120, 10, 20, -1.0]])
    out, rep = validate_observations(df)
    assert np.isnan(out.loc[0, "temperature_c"]) and np.isnan(out.loc[1, "humidity_pct"])
    assert np.isnan(out.loc[1, "rain_1h_mm"])
    assert rep["values_out_of_range"]["temperature_c"] == 1 and rep["values_out_of_range"]["rain_1h_mm"] == 1


def test_duplicates_and_bad_dates_removed():
    df = _df([["07130", "2024-01-01T00:00:00Z", 10, 1010, 80, 10, 20, 0.0],
              ["07130", "2024-01-01 00:00:00", 11, 1010, 80, 10, 20, 0.0],  # meme instant, autre format
              ["07130", "pas-une-date", 10, 1010, 80, 10, 20, 0.0]])
    out, rep = validate_observations(df)
    assert len(out) == 1 and rep["duplicates_removed"] == 1 and rep["rows_invalid_date"] == 1


def test_missing_column_raises():
    with pytest.raises(ValidationError):
        validate_observations(pd.DataFrame({"station_id": ["x"]}))


def test_strings_are_coerced_to_numbers():
    df = _df([["07130", "2024-01-01T00:00:00Z", "12.5", "1010", "80", "10", "20", "0.2"]])
    out, _ = validate_observations(df)
    assert out.loc[0, "temperature_c"] == 12.5
