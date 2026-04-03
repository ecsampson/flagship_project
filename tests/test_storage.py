import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.storage import (
    store_noaa_data,
    store_extreme_weather,
    store_weather_signals,
    upload_to_s3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEATHER_RECORDS = [
    {"date": "2024-01-15", "type": "TMAX", "value": 32.0},
    {"date": "2024-01-15", "type": "PRCP", "value": 5.0},
]

EXTREME_RECORDS = [
    {"date": "2024-07-01", "type": "TMAX", "value": 40.0, "threshold": 35.0, "condition": "above"},
]


# ---------------------------------------------------------------------------
# store_noaa_data
# ---------------------------------------------------------------------------

def test_store_noaa_data_creates_csv_with_correct_rows(tmp_path):
    """
    The file should contain exactly as many data rows as records passed in.
    tmp_path is a pytest built-in that provides a fresh temporary directory
    per test — no cleanup needed and no risk of tests interfering with each other.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_csv(file_path)
    assert len(df) == 2


def test_store_noaa_data_writes_correct_columns(tmp_path):
    """
    Downstream consumers of the CSV depend on specific column names.
    A column rename in the dict keys would silently break them.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_csv(file_path)
    assert set(df.columns) == {"date", "type", "value"}


def test_store_noaa_data_appends_on_second_call(tmp_path):
    """
    The pipeline is designed to run repeatedly and accumulate records.
    A second call must append rather than overwrite — losing prior data
    would corrupt the historical dataset silently.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_csv(file_path)
    assert len(df) == 4


def test_store_noaa_data_header_written_once_on_append(tmp_path):
    """
    Appending must not repeat the header row. If the header appears twice,
    pandas will read it as a data row on the next load, corrupting types.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_csv(file_path)
    # If the header were duplicated, 'date' would appear as a value in the
    # date column and len(df) would be 5 instead of 4.
    assert "date" not in df["date"].values


def test_store_noaa_data_preserves_values(tmp_path):
    """
    The values written to disk must match the input exactly. Floating-point
    rounding or type coercion during CSV serialization would silently corrupt
    downstream analytics.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_csv(file_path)
    assert df.iloc[0]["date"] == "2024-01-15"
    assert df.iloc[0]["type"] == "TMAX"
    assert df.iloc[0]["value"] == 32.0


# ---------------------------------------------------------------------------
# store_extreme_weather
# ---------------------------------------------------------------------------

def test_store_extreme_weather_creates_csv_with_correct_rows(tmp_path):
    """Same append-mode contract as store_noaa_data, verified for the extreme
    events path which writes additional columns (threshold, condition)."""
    file_path = tmp_path / "extreme.csv"
    store_extreme_weather(EXTREME_RECORDS, file_path)

    df = pd.read_csv(file_path)
    assert len(df) == 1


def test_store_extreme_weather_writes_all_columns(tmp_path):
    """
    Extreme event records carry extra metadata (threshold, condition) that
    doesn't exist in the raw weather records. These must survive the write
    round-trip intact so the CSV is usable for alerting and reporting.
    """
    file_path = tmp_path / "extreme.csv"
    store_extreme_weather(EXTREME_RECORDS, file_path)

    df = pd.read_csv(file_path)
    assert set(df.columns) == {"date", "type", "value", "threshold", "condition"}


def test_store_extreme_weather_appends_on_second_call(tmp_path):
    """Same append behavior requirement as the raw weather store."""
    file_path = tmp_path / "extreme.csv"
    store_extreme_weather(EXTREME_RECORDS, file_path)
    store_extreme_weather(EXTREME_RECORDS, file_path)

    df = pd.read_csv(file_path)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# store_weather_signals
# ---------------------------------------------------------------------------

def test_store_weather_signals_creates_csv(tmp_path):
    """
    store_weather_signals accepts a DataFrame directly (not a list of dicts)
    because weather signals are already built as a DataFrame by the feature
    engineering step. Verify the round-trip works end-to-end.
    """
    file_path = tmp_path / "signals.csv"
    signals_df = pd.DataFrame([
        {"date": "2024-01-15", "tmax_7d_avg": 28.5, "prcp_7d_total": 12.0},
    ])
    store_weather_signals(signals_df, file_path)

    df = pd.read_csv(file_path)
    assert len(df) == 1
    assert "tmax_7d_avg" in df.columns


def test_store_weather_signals_appends_on_second_call(tmp_path):
    """Same append-only contract verified for the signals path."""
    file_path = tmp_path / "signals.csv"
    signals_df = pd.DataFrame([{"date": "2024-01-15", "tmax_7d_avg": 28.5}])

    store_weather_signals(signals_df, file_path)
    store_weather_signals(signals_df, file_path)

    df = pd.read_csv(file_path)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# upload_to_s3
# ---------------------------------------------------------------------------

def test_upload_to_s3_raises_on_invalid_data_type(tmp_path):
    """
    upload_to_s3 guards against typos in data_type before touching AWS.
    Without this check a bad string would create a file under an unexpected
    S3 prefix and be silently lost. Verify the ValueError fires before any
    network call is made.
    """
    fake_file = tmp_path / "data.csv"
    fake_file.write_text("col\nval")

    with pytest.raises(ValueError, match="data_type must be"):
        upload_to_s3(fake_file, "wrong_type", "my-bucket")


@pytest.mark.parametrize("data_type", ["raw", "extreme", "features", "dimensions", "facts"])
def test_upload_to_s3_accepts_all_valid_data_types(tmp_path, data_type):
    """
    Every valid data_type should reach the boto3 call without raising.
    Using parametrize means each type is an independent test case — if one
    breaks, the others still run.
    """
    fake_file = tmp_path / "data.csv"
    fake_file.write_text("col\nval")

    with patch("src.storage.boto3.client") as mock_boto3_client:
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        upload_to_s3(fake_file, data_type, "my-bucket")

        mock_s3.upload_file.assert_called_once()


def test_upload_to_s3_constructs_correct_s3_key(tmp_path):
    """
    The S3 key must be '{data_type}/{filename}'. A wrong key puts the file
    in the wrong prefix — downstream jobs reading from s3://bucket/raw/
    would silently miss it.
    """
    fake_file = tmp_path / "noaa_weather_data.csv"
    fake_file.write_text("col\nval")

    with patch("src.storage.boto3.client") as mock_boto3_client:
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        upload_to_s3(fake_file, "raw", "my-bucket")

        _, call_args, _ = mock_s3.upload_file.mock_calls[0]
        s3_key = call_args[2]
        assert s3_key == "raw/noaa_weather_data.csv"


def test_upload_to_s3_uses_correct_bucket(tmp_path):
    """
    The bucket name passed in must be forwarded to boto3 unchanged.
    A hardcoded bucket name in the function would deploy to the wrong
    environment when the bucket name is overridden via config.
    """
    fake_file = tmp_path / "data.csv"
    fake_file.write_text("col\nval")

    with patch("src.storage.boto3.client") as mock_boto3_client:
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        upload_to_s3(fake_file, "raw", "production-bucket")

        _, call_args, _ = mock_s3.upload_file.mock_calls[0]
        bucket = call_args[1]
        assert bucket == "production-bucket"


# ---------------------------------------------------------------------------
# Parquet output — store_noaa_data
# ---------------------------------------------------------------------------

def test_store_noaa_data_creates_parquet_alongside_csv(tmp_path):
    """
    A parquet file must be written next to the CSV on every call.
    If parquet is missing, downstream consumers expecting columnar format
    would fail with a FileNotFoundError — with no indication the CSV succeeded.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)

    assert file_path.with_suffix(".parquet").exists()


def test_store_noaa_data_parquet_has_correct_row_count(tmp_path):
    """
    The parquet file must contain the same number of rows as the input.
    A row count mismatch between the CSV and parquet would corrupt any
    pipeline that reads one format for processing and the other for auditing.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert len(df) == len(WEATHER_RECORDS)


def test_store_noaa_data_parquet_has_correct_columns(tmp_path):
    """Parquet output must have the same column schema as the CSV."""
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert set(df.columns) == {"date", "type", "value"}


def test_store_noaa_data_parquet_preserves_values(tmp_path):
    """
    Values written to parquet must match the input exactly. Parquet uses
    typed encoding — a float stored as an integer or an object stored as
    a string would silently corrupt downstream analytics.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert df.iloc[0]["value"] == 32.0
    assert df.iloc[0]["type"] == "TMAX"


def test_store_noaa_data_parquet_overwrites_not_appends(tmp_path):
    """
    Parquet has no append mode — each write must produce a complete,
    self-contained snapshot. A second call must not double the row count.
    The CSV counterpart appends; this asymmetry is intentional and must
    not be accidentally 'fixed' by adding rows to the parquet file.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)
    store_noaa_data(WEATHER_RECORDS, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert len(df) == len(WEATHER_RECORDS)  # not doubled


def test_store_noaa_data_csv_still_appends_when_parquet_overwrites(tmp_path):
    """
    The parquet overwrite must not affect CSV append behavior. Both behaviors
    must coexist — breaking CSV append would drop historical records.
    """
    file_path = tmp_path / "weather.csv"
    store_noaa_data(WEATHER_RECORDS, file_path)
    store_noaa_data(WEATHER_RECORDS, file_path)

    csv_df = pd.read_csv(file_path)
    parquet_df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert len(csv_df) == len(WEATHER_RECORDS) * 2  # appended
    assert len(parquet_df) == len(WEATHER_RECORDS)   # overwritten


# ---------------------------------------------------------------------------
# Parquet output — store_extreme_weather
# ---------------------------------------------------------------------------

def test_store_extreme_weather_creates_parquet_alongside_csv(tmp_path):
    """Parquet file must exist after a call to store_extreme_weather."""
    file_path = tmp_path / "extreme.csv"
    store_extreme_weather(EXTREME_RECORDS, file_path)

    assert file_path.with_suffix(".parquet").exists()


def test_store_extreme_weather_parquet_has_all_columns(tmp_path):
    """
    The threshold and condition columns must survive the parquet write.
    They are the reason extreme events have a separate output from raw weather —
    losing them in parquet format would make the parquet file useless for alerting.
    """
    file_path = tmp_path / "extreme.csv"
    store_extreme_weather(EXTREME_RECORDS, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert set(df.columns) == {"date", "type", "value", "threshold", "condition"}


def test_store_extreme_weather_parquet_overwrites_not_appends(tmp_path):
    """Parquet must overwrite on the second call, same as store_noaa_data."""
    file_path = tmp_path / "extreme.csv"
    store_extreme_weather(EXTREME_RECORDS, file_path)
    store_extreme_weather(EXTREME_RECORDS, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert len(df) == len(EXTREME_RECORDS)  # not doubled


# ---------------------------------------------------------------------------
# Parquet output — store_weather_signals
# ---------------------------------------------------------------------------

def test_store_weather_signals_creates_parquet_alongside_csv(tmp_path):
    """Parquet file must exist after a call to store_weather_signals."""
    file_path = tmp_path / "signals.csv"
    signals_df = pd.DataFrame([{"date": "2024-01-15", "tmax_7d_avg": 28.5}])
    store_weather_signals(signals_df, file_path)

    assert file_path.with_suffix(".parquet").exists()


def test_store_weather_signals_parquet_has_correct_row_count(tmp_path):
    """Parquet row count must match the input DataFrame."""
    file_path = tmp_path / "signals.csv"
    signals_df = pd.DataFrame([
        {"date": "2024-01-15", "tmax_7d_avg": 28.5},
        {"date": "2024-01-16", "tmax_7d_avg": 29.0},
    ])
    store_weather_signals(signals_df, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert len(df) == 2


def test_store_weather_signals_parquet_overwrites_not_appends(tmp_path):
    """Parquet must overwrite on the second call."""
    file_path = tmp_path / "signals.csv"
    signals_df = pd.DataFrame([{"date": "2024-01-15", "tmax_7d_avg": 28.5}])
    store_weather_signals(signals_df, file_path)
    store_weather_signals(signals_df, file_path)

    df = pd.read_parquet(file_path.with_suffix(".parquet"))
    assert len(df) == 1  # not doubled


# ---------------------------------------------------------------------------
# Parquet upload to S3
# ---------------------------------------------------------------------------

def test_upload_to_s3_constructs_correct_s3_key_for_parquet(tmp_path):
    """
    The S3 key for a parquet file must be '{data_type}/{filename}.parquet'.
    upload_to_s3 derives the key from file_path.name — it must not strip or
    change the extension, or the parquet file would land at the wrong S3 path.
    """
    parquet_file = tmp_path / "USC00213567_weather_data.parquet"
    pd.DataFrame([{"col": "val"}]).to_parquet(parquet_file, index=False)

    with patch("src.storage.boto3.client") as mock_boto3_client:
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        upload_to_s3(parquet_file, "raw", "my-bucket")

        _, call_args, _ = mock_s3.upload_file.mock_calls[0]
        assert call_args[2] == "raw/USC00213567_weather_data.parquet"
