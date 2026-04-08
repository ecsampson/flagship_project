import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.storage import (
    store_noaa_data,
    store_extreme_weather,
    store_weather_signals,
    upload_to_s3,
    upload_parquet_append_to_s3,
    get_last_run_date,
    save_last_run_date,
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
        assert s3_key == "raw/csv/noaa_weather_data.csv"


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
        assert call_args[2] == "raw/parquet/USC00213567_weather_data.parquet"


# ---------------------------------------------------------------------------
# Helpers shared by the S3 append/dedup tests
# ---------------------------------------------------------------------------

def _make_parquet_bytes(records):
    """Serialize a list of dicts to an in-memory Parquet buffer."""
    buf = io.BytesIO()
    pd.DataFrame(records).to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    return buf.getvalue()


def _s3_client_mock(existing_parquet_bytes=None, get_error_code=None):
    """
    Return a (mock_boto3_client patcher, mock_s3) pair ready for use in a
    with-statement. Configures get_object to either return parquet bytes,
    raise a ClientError, or raise NoSuchKey depending on the arguments.
    """
    mock_s3 = MagicMock()

    if get_error_code is not None:
        error_response = {"Error": {"Code": get_error_code, "Message": "test"}}
        mock_s3.get_object.side_effect = ClientError(error_response, "GetObject")
    elif existing_parquet_bytes is not None:
        mock_body = MagicMock()
        mock_body.read.return_value = existing_parquet_bytes
        mock_s3.get_object.return_value = {"Body": mock_body}

    return mock_s3


# ---------------------------------------------------------------------------
# upload_parquet_append_to_s3
# ---------------------------------------------------------------------------

class TestUploadParquetAppendToS3:
    """
    upload_parquet_append_to_s3 reads existing Parquet from S3 (if present),
    concatenates new rows, deduplicates, and writes the result back.
    These tests verify the three code paths: new key, existing key, and empty input.
    """

    def test_new_key_writes_input_df_directly(self):
        """
        When the S3 key does not exist (NoSuchKey), there is no prior data to
        merge with. The function must write the new DataFrame as-is. Writing
        more or fewer rows than the input would corrupt a fresh data lake object.
        """
        new_rows = [
            {"date": "2024-01-01", "type": "TMAX", "value": 32.0},
            {"date": "2024-01-02", "type": "TMAX", "value": 33.0},
        ]
        df = pd.DataFrame(new_rows)
        mock_s3 = _s3_client_mock(get_error_code="NoSuchKey")

        with patch("src.storage.boto3.client", return_value=mock_s3):
            upload_parquet_append_to_s3(df, "test-bucket", "raw/parquet/weather.parquet", ["date", "type"])

        put_call = mock_s3.put_object.call_args
        written = pd.read_parquet(io.BytesIO(put_call.kwargs["Body"]))
        assert len(written) == 2

    def test_existing_key_appends_non_overlapping_rows(self):
        """
        When the S3 key exists and the new rows have different natural keys than
        the existing rows, all rows must appear in the output. A lost row here
        means historical records are silently dropped on the next pipeline run.
        """
        existing_bytes = _make_parquet_bytes([
            {"date": "2024-01-01", "type": "TMAX", "value": 30.0},
            {"date": "2024-01-02", "type": "TMAX", "value": 31.0},
        ])
        new_df = pd.DataFrame([{"date": "2024-01-03", "type": "TMAX", "value": 32.0}])
        mock_s3 = _s3_client_mock(existing_parquet_bytes=existing_bytes)

        with patch("src.storage.boto3.client", return_value=mock_s3):
            upload_parquet_append_to_s3(new_df, "test-bucket", "raw/parquet/weather.parquet", ["date", "type"])

        put_call = mock_s3.put_object.call_args
        written = pd.read_parquet(io.BytesIO(put_call.kwargs["Body"]))
        assert len(written) == 3

    def test_dedup_new_row_wins_on_same_key(self):
        """
        When the same natural key exists in both old and new data, the new row
        must replace the old one (keep='last'). The old value surviving would mean
        a correction run never updates the stored value — silently stale data.
        """
        existing_bytes = _make_parquet_bytes([
            {"date": "2024-01-01", "type": "TMAX", "value": 30.0},
        ])
        new_df = pd.DataFrame([{"date": "2024-01-01", "type": "TMAX", "value": 99.0}])
        mock_s3 = _s3_client_mock(existing_parquet_bytes=existing_bytes)

        with patch("src.storage.boto3.client", return_value=mock_s3):
            upload_parquet_append_to_s3(new_df, "test-bucket", "raw/parquet/weather.parquet", ["date", "type"])

        put_call = mock_s3.put_object.call_args
        written = pd.read_parquet(io.BytesIO(put_call.kwargs["Body"]))
        assert len(written) == 1
        assert written.iloc[0]["value"] == 99.0

    def test_dedup_does_not_drop_distinct_keys(self):
        """
        Deduplication must only remove rows whose natural keys collide. Rows with
        different keys in the same batch must all survive. An over-aggressive dedup
        (e.g. deduping on the wrong column) would silently lose valid records.
        """
        existing_bytes = _make_parquet_bytes([
            {"date": "2024-01-01", "type": "TMAX", "value": 30.0},
        ])
        new_df = pd.DataFrame([
            {"date": "2024-01-01", "type": "TMAX", "value": 30.0},  # duplicate key
            {"date": "2024-01-01", "type": "TMIN", "value": 10.0},  # different type — keep
            {"date": "2024-01-02", "type": "TMAX", "value": 31.0},  # different date — keep
        ])
        mock_s3 = _s3_client_mock(existing_parquet_bytes=existing_bytes)

        with patch("src.storage.boto3.client", return_value=mock_s3):
            upload_parquet_append_to_s3(new_df, "test-bucket", "raw/parquet/weather.parquet", ["date", "type"])

        put_call = mock_s3.put_object.call_args
        written = pd.read_parquet(io.BytesIO(put_call.kwargs["Body"]))
        assert len(written) == 3  # 1 existing deduped + 2 new distinct rows

    def test_empty_dataframe_makes_no_s3_calls(self):
        """
        An empty DataFrame must return early without touching S3. Uploading an
        empty Parquet would overwrite a healthy S3 object with zero rows, silently
        destroying the historical dataset for that key.
        """
        mock_s3 = MagicMock()

        with patch("src.storage.boto3.client", return_value=mock_s3):
            upload_parquet_append_to_s3(
                pd.DataFrame(), "test-bucket", "raw/parquet/weather.parquet", ["date", "type"]
            )

        mock_s3.get_object.assert_not_called()
        mock_s3.put_object.assert_not_called()

    def test_non_nosuchkey_error_reraises(self):
        """
        Only NoSuchKey is a recoverable condition (first run with no prior data).
        Any other S3 error (permissions, network, etc.) must propagate so the
        pipeline fails loudly rather than silently writing incomplete data.
        """
        mock_s3 = _s3_client_mock(get_error_code="AccessDenied")
        df = pd.DataFrame([{"date": "2024-01-01", "type": "TMAX", "value": 32.0}])

        with patch("src.storage.boto3.client", return_value=mock_s3):
            with pytest.raises(ClientError) as exc_info:
                upload_parquet_append_to_s3(df, "test-bucket", "raw/parquet/weather.parquet", ["date", "type"])

        assert exc_info.value.response["Error"]["Code"] == "AccessDenied"

    def test_put_object_called_with_correct_bucket_and_key(self):
        """
        The write-back must target the exact bucket and S3 key that were passed
        in. A hardcoded or misrouted key would silently deposit data at the wrong
        path and leave the intended path stale.
        """
        mock_s3 = _s3_client_mock(get_error_code="NoSuchKey")
        df = pd.DataFrame([{"date": "2024-01-01", "type": "TMAX", "value": 32.0}])

        with patch("src.storage.boto3.client", return_value=mock_s3):
            upload_parquet_append_to_s3(df, "my-bucket", "facts/parquet/fact.parquet", ["date", "type"])

        put_call = mock_s3.put_object.call_args
        assert put_call.kwargs["Bucket"] == "my-bucket"
        assert put_call.kwargs["Key"] == "facts/parquet/fact.parquet"


# ---------------------------------------------------------------------------
# get_last_run_date
# ---------------------------------------------------------------------------

class TestGetLastRunDate:
    """
    get_last_run_date reads a JSON cursor from S3 so the Lambda knows where
    to resume. These tests verify the happy path, the first-run fallback, and
    that unexpected errors are not silently swallowed.
    """

    def _make_mock_s3(self, date_str=None, error_code=None):
        mock_s3 = MagicMock()
        if error_code:
            err = {"Error": {"Code": error_code, "Message": "test"}}
            mock_s3.get_object.side_effect = ClientError(err, "GetObject")
        else:
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({"last_run_date": date_str}).encode()
            mock_s3.get_object.return_value = {"Body": mock_body}
        return mock_s3

    def test_returns_date_from_s3_json(self):
        """
        The primary use case: S3 has a cursor from a prior run. The function
        must return the stored date string exactly so the next pipeline window
        starts where the last one ended.
        """
        mock_s3 = self._make_mock_s3(date_str="2024-06-15")
        with patch("src.storage.boto3.client", return_value=mock_s3):
            result = get_last_run_date("test-bucket")
        assert result == "2024-06-15"

    def test_nosuchkey_returns_full_backfill_start_date(self):
        """
        On the very first run no cursor exists in S3. The function must return
        the earliest available NOAA date ("1938-01-01") so the backfill starts
        from the beginning. Any other fallback would silently skip early history.
        """
        mock_s3 = self._make_mock_s3(error_code="NoSuchKey")
        with patch("src.storage.boto3.client", return_value=mock_s3):
            result = get_last_run_date("test-bucket")
        assert result == "1938-01-01"

    def test_other_s3_error_reraises(self):
        """
        A permissions error or network failure must not be mistaken for a
        first-run condition and silently return the fallback date. Doing so
        would restart a completed backfill from 1938 — destroying the cursor.
        """
        mock_s3 = self._make_mock_s3(error_code="AccessDenied")
        with patch("src.storage.boto3.client", return_value=mock_s3):
            with pytest.raises(ClientError) as exc_info:
                get_last_run_date("test-bucket")
        assert exc_info.value.response["Error"]["Code"] == "AccessDenied"

    def test_reads_from_correct_s3_key(self):
        """
        The cursor lives at a fixed path. Reading from any other key would
        return stale or missing data and silently reset the pipeline window.
        """
        mock_s3 = self._make_mock_s3(date_str="2024-06-15")
        with patch("src.storage.boto3.client", return_value=mock_s3):
            get_last_run_date("my-bucket")
        mock_s3.get_object.assert_called_once_with(
            Bucket="my-bucket", Key="metadata/last_run_date.json"
        )


# ---------------------------------------------------------------------------
# save_last_run_date
# ---------------------------------------------------------------------------

class TestSaveLastRunDate:
    """
    save_last_run_date persists the successfully processed end date so the
    next Lambda invocation can resume from the right window.
    """

    def _run_save(self, date_str="2024-06-15", bucket="test-bucket"):
        mock_s3 = MagicMock()
        with patch("src.storage.boto3.client", return_value=mock_s3):
            save_last_run_date(date_str, bucket)
        return mock_s3.put_object.call_args

    def test_writes_correct_json_body(self):
        """
        The body must be valid JSON containing last_run_date. A malformed body
        would crash get_last_run_date on the next invocation, stalling the pipeline.
        """
        put_kwargs = self._run_save("2024-06-15").kwargs
        body = json.loads(put_kwargs["Body"])
        assert body == {"last_run_date": "2024-06-15"}

    def test_writes_to_correct_s3_key(self):
        """
        The cursor must be saved to the same key that get_last_run_date reads from.
        A key mismatch means get_last_run_date never sees the saved cursor.
        """
        put_kwargs = self._run_save().kwargs
        assert put_kwargs["Key"] == "metadata/last_run_date.json"

    def test_writes_to_correct_bucket(self):
        """
        The bucket passed in must be forwarded to S3 unchanged. Writing to a
        hardcoded bucket name would silently write the cursor to the wrong
        environment's bucket.
        """
        put_kwargs = self._run_save(bucket="production-bucket").kwargs
        assert put_kwargs["Bucket"] == "production-bucket"

    def test_sets_content_type_json(self):
        """
        ContentType must be application/json so AWS and downstream tooling
        correctly interpret the object without needing to inspect the content.
        """
        put_kwargs = self._run_save().kwargs
        assert put_kwargs["ContentType"] == "application/json"
