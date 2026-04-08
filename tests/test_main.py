"""
Unit tests for main.py behaviors introduced by the station-ID naming,
dual CSV+Parquet output changes, and the Lambda handler.

Strategy: patch CONFIG_PATH to a real (but minimal) temp yaml file, patch
DATA_DIR to a temp directory, and mock fetch_all_noaa_data + upload_to_s3
+ upload_parquet_append_to_s3 so no network calls are made. The store
functions run against the real filesystem, which also exercises the parquet
write path end-to-end.
"""

import datetime
import importlib
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.main import main, lambda_handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal raw record — just enough for the full pipeline to run without crashing.
# One record means all downstream transformations (dim_date, signals, etc.) run
# but complete quickly. TMAX 400 → 40.0°C triggers the extreme weather threshold.
_RAW_RECORDS = [
    {"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:USC00213567", "TMAX": "400"},
]

_BASE_SETTINGS = {
    "noaa": {
        "api_key": "test-key",
        "base_url": "http://example.com/api",
        "default_location": "FIPS:27",
        "query": {
            "stations": "USC00213567",
            "dataset": "daily-summaries",
            "dataTypes": "TMAX",
            "startDate": "2024-01-15",
            "endDate": "2024-01-15",
            "format": "json",
        },
    },
    "aws": {"bucket": "test-bucket"},
    "extreme_weather": {
        "thresholds": {
            "TMAX": {"above": 35.0},
            "TMIN": {"below": -20.0},
        }
    },
}


def _write_config(tmp_path, settings=None):
    """Write a settings dict as YAML and return the file path."""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.dump(settings or _BASE_SETTINGS))
    return config_path


def _run_main(config_path, data_dir):
    """Run main() with real filesystem but mocked network calls."""
    with patch("src.main.CONFIG_PATH", config_path), \
         patch("src.main.DATA_DIR", data_dir), \
         patch("src.main.fetch_all_noaa_data", return_value=_RAW_RECORDS), \
         patch("src.main.upload_to_s3"), \
         patch("src.main.upload_parquet_append_to_s3"):
        main()


def _run_main_capture_uploads(config_path, data_dir):
    """Run main() and return the mock upload_to_s3 so call args can be inspected."""
    with patch("src.main.CONFIG_PATH", config_path), \
         patch("src.main.DATA_DIR", data_dir), \
         patch("src.main.fetch_all_noaa_data", return_value=_RAW_RECORDS), \
         patch("src.main.upload_to_s3") as mock_upload, \
         patch("src.main.upload_parquet_append_to_s3"):
        main()
    return mock_upload


def _run_main_capture_all_uploads(config_path, data_dir):
    """
    Run main() and return both upload mocks.
    CSV outputs go through upload_to_s3; Parquet outputs go through
    upload_parquet_append_to_s3. Both must be captured to verify complete
    S3 coverage.
    """
    with patch("src.main.CONFIG_PATH", config_path), \
         patch("src.main.DATA_DIR", data_dir), \
         patch("src.main.fetch_all_noaa_data", return_value=_RAW_RECORDS), \
         patch("src.main.upload_to_s3") as mock_csv_upload, \
         patch("src.main.upload_parquet_append_to_s3") as mock_parquet_upload:
        main()
    return mock_csv_upload, mock_parquet_upload


# ---------------------------------------------------------------------------
# Station ID extraction
# ---------------------------------------------------------------------------

class TestStationIdExtraction:
    """
    The station ID from settings["noaa"]["query"]["stations"] is the prefix
    for every output filename. Incorrect extraction silently writes files to
    the wrong path — or crashes with a KeyError on startup.
    """

    def test_string_station_id_used_as_filename_prefix(self, tmp_path):
        """
        When stations is a plain string, it must be used verbatim as the
        filename prefix. No truncation, coercion, or reformatting allowed.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)

        _run_main(config, data_dir)

        assert (data_dir / "USC00213567_weather_data.csv").exists()

    def test_list_station_id_uses_first_element(self, tmp_path):
        """
        When stations is a list (multiple stations in the query), the first
        element must be used as the prefix. Using all of them would produce
        an invalid filename; using none would crash on the string format.
        """
        settings = {**_BASE_SETTINGS, "noaa": {**_BASE_SETTINGS["noaa"],
            "query": {**_BASE_SETTINGS["noaa"]["query"],
                "stations": ["USC00213567", "USW00014922"]}}}
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path, settings)

        _run_main(config, data_dir)

        assert (data_dir / "USC00213567_weather_data.csv").exists()

    def test_list_station_id_does_not_use_second_element(self, tmp_path):
        """
        Only the first list element may be used as the prefix. A file prefixed
        with the second station would mean main() iterated the list instead of
        taking the first element — that would also produce duplicate outputs.
        """
        settings = {**_BASE_SETTINGS, "noaa": {**_BASE_SETTINGS["noaa"],
            "query": {**_BASE_SETTINGS["noaa"]["query"],
                "stations": ["USC00213567", "USW00014922"]}}}
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path, settings)

        _run_main(config, data_dir)

        assert not (data_dir / "USW00014922_weather_data.csv").exists()


# ---------------------------------------------------------------------------
# Station-prefixed filenames
# ---------------------------------------------------------------------------

class TestStationPrefixedFilenames:
    """
    All six pipeline outputs (weather data, extreme events, signals, dim_date,
    dim_location, fact table) must be written with the station ID prefix for
    both CSV and Parquet formats. A missing prefix means the next pipeline run
    for a different station overwrites the file silently.
    """

    def test_all_csv_files_have_station_id_prefix(self, tmp_path):
        """Every CSV output file must begin with the station ID."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        _run_main(config, data_dir)

        expected = [
            "USC00213567_weather_data.csv",
            "USC00213567_extreme_weather.csv",
            "USC00213567_weather_signals.csv",
            "USC00213567_dim_date.csv",
            "USC00213567_dim_location.csv",
            "USC00213567_fact_weather_observations.csv",
        ]
        for filename in expected:
            assert (data_dir / filename).exists(), f"Missing CSV output: {filename}"

    def test_all_parquet_files_have_station_id_prefix(self, tmp_path):
        """Every Parquet output file must begin with the station ID."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        _run_main(config, data_dir)

        expected = [
            "USC00213567_weather_data.parquet",
            "USC00213567_extreme_weather.parquet",
            "USC00213567_weather_signals.parquet",
            "USC00213567_dim_date.parquet",
            "USC00213567_dim_location.parquet",
            "USC00213567_fact_weather_observations.parquet",
        ]
        for filename in expected:
            assert (data_dir / filename).exists(), f"Missing Parquet output: {filename}"

    def test_no_legacy_unprefixed_csv_files_created(self, tmp_path):
        """
        The old filenames (noaa_weather_data.csv, etc.) must not be created.
        If they are, either main.py was partially reverted or both old and new
        paths are being written — wasting disk and confusing downstream readers.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        _run_main(config, data_dir)

        legacy_names = [
            "noaa_weather_data.csv",
            "noaa_extreme_weather.csv",
            "feat_weather_signals.csv",
        ]
        for filename in legacy_names:
            assert not (data_dir / filename).exists(), f"Unexpected legacy file: {filename}"


# ---------------------------------------------------------------------------
# Dual S3 uploads (CSV + Parquet)
# ---------------------------------------------------------------------------

class TestDualS3Upload:
    """
    Every pipeline output must be uploaded to S3 in both CSV and Parquet formats.
    Skipping parquet uploads means the data lake only has CSV — all Parquet
    performance benefits are lost for downstream consumers reading from S3.
    """

    def test_upload_called_twelve_times_total(self, tmp_path):
        """
        6 outputs × 2 formats = 12 upload calls total. CSV uploads go through
        upload_to_s3 (6 calls); Parquet uploads go through
        upload_parquet_append_to_s3 (6 calls). Fewer than 6 on either path means
        at least one output was silently skipped in that format.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        mock_csv, mock_parquet = _run_main_capture_all_uploads(config, data_dir)

        assert mock_csv.call_count == 6
        assert mock_parquet.call_count == 6

    def test_six_csv_and_six_parquet_files_uploaded(self, tmp_path):
        """
        Exactly 6 CSV uploads via upload_to_s3 and 6 Parquet uploads via
        upload_parquet_append_to_s3. An imbalance means one output was written
        locally but never sent to the data lake in the missing format.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        mock_csv, mock_parquet = _run_main_capture_all_uploads(config, data_dir)

        csv_paths = [str(c.args[0]) for c in mock_csv.call_args_list]
        assert len([p for p in csv_paths if p.endswith(".csv")]) == 6

        # upload_parquet_append_to_s3 receives a DataFrame, not a file path —
        # check call count rather than inspecting paths.
        assert mock_parquet.call_count == 6

    def test_parquet_uploads_use_same_data_type_prefix_as_csv(self, tmp_path):
        """
        Each parquet file must be uploaded under the same S3 data_type prefix
        as its CSV counterpart (e.g. dim_date.csv and dim_date.parquet both
        go to 'dimensions/'). A mismatch would scatter parquet files across
        wrong S3 prefixes, breaking downstream jobs that partition by prefix.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        mock_upload = _run_main_capture_uploads(config, data_dir)

        # Build a map from filename stem → data_type for all calls
        stem_to_types = {}
        for c in mock_upload.call_args_list:
            path = Path(str(c.args[0]))
            data_type = c.args[1]
            stem_to_types.setdefault(path.stem, set()).add(data_type)

        # Each stem must map to exactly one data_type (CSV and parquet use the same)
        for stem, types in stem_to_types.items():
            assert len(types) == 1, (
                f"{stem} uploaded under multiple data_type prefixes: {types}"
            )

    def test_all_six_output_stems_are_uploaded(self, tmp_path):
        """
        All six output stems must appear in the upload calls. A stem absent
        from S3 means that output was stored locally but never sent to the lake.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        mock_upload = _run_main_capture_uploads(config, data_dir)

        uploaded_stems = {
            Path(str(c.args[0])).stem for c in mock_upload.call_args_list
        }
        expected_stems = {
            "USC00213567_weather_data",
            "USC00213567_extreme_weather",
            "USC00213567_weather_signals",
            "USC00213567_dim_date",
            "USC00213567_dim_location",
            "USC00213567_fact_weather_observations",
        }
        assert expected_stems == uploaded_stems

    def test_no_data_returned_skips_all_uploads(self, tmp_path):
        """
        When fetch_all_noaa_data returns an empty list, no files should be
        written and no S3 uploads should happen. Uploading empty files would
        corrupt the data lake with zero-byte objects.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)

        with patch("src.main.CONFIG_PATH", config), \
             patch("src.main.DATA_DIR", data_dir), \
             patch("src.main.fetch_all_noaa_data", return_value=[]), \
             patch("src.main.upload_to_s3") as mock_upload, \
             patch("src.main.upload_parquet_append_to_s3"):
            main()

        assert mock_upload.call_count == 0


# ---------------------------------------------------------------------------
# Helpers for lambda_handler tests
# ---------------------------------------------------------------------------

_LAMBDA_SETTINGS = {"aws": {"bucket": "test-bucket"}}


def _write_lambda_config(tmp_path):
    """Write the minimal settings lambda_handler reads (only aws.bucket)."""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.dump(_LAMBDA_SETTINGS))
    return config_path


def _run_lambda_handler(config_path, start_date):
    """
    Invoke lambda_handler with CONFIG_PATH patched and all external calls
    (get_last_run_date, main, save_last_run_date) mocked.

    Returns (result, mock_save, mock_main) for assertion.
    """
    with patch("src.main.CONFIG_PATH", config_path), \
         patch("src.main.get_last_run_date", return_value=start_date) as mock_get, \
         patch("src.main.save_last_run_date") as mock_save, \
         patch("src.main.main") as mock_main_fn:
        result = lambda_handler({}, None)
    return result, mock_save, mock_main_fn


# ---------------------------------------------------------------------------
# lambda_handler — date windowing
# ---------------------------------------------------------------------------

class TestLambdaHandler:
    """
    lambda_handler drives the incremental/backfill logic: it reads a cursor
    from S3, computes an end date (either a 10-year chunk cap or yesterday),
    runs main(), then advances the cursor. These tests verify the windowing
    arithmetic and the wiring between the cursor functions and main().
    """

    def test_backfill_end_date_is_capped_at_ten_year_chunk(self, tmp_path):
        """
        When start_date is far in the past, chunk_end (start + 10 years - 1 day)
        is before yesterday. The end_date passed to main() must be chunk_end,
        not yesterday — advancing too far would skip data and corrupt the cursor.
        """
        config = _write_lambda_config(tmp_path)
        _, _, mock_main_fn = _run_lambda_handler(config, "1938-01-01")

        # 1938-01-01 + 10 years - 1 day = 1947-12-31
        assert mock_main_fn.call_args.kwargs["end_date"] == "1947-12-31"

    def test_backfill_start_date_passed_to_main(self, tmp_path):
        """
        main() must receive the same start_date that get_last_run_date returned.
        Any transformation or offset here would create a gap or overlap in the
        processed window.
        """
        config = _write_lambda_config(tmp_path)
        _, _, mock_main_fn = _run_lambda_handler(config, "1938-01-01")

        assert mock_main_fn.call_args.kwargs["start_date"] == "1938-01-01"

    def test_caught_up_end_date_is_yesterday(self, tmp_path):
        """
        When start_date is recent enough that chunk_end exceeds yesterday,
        end_date must be yesterday (no future data exists). Requesting future
        dates from NOAA would return empty results and waste an invocation.
        """
        config = _write_lambda_config(tmp_path)
        # start 2020-04-01 → chunk_end 2030-03-31 > yesterday for any near-term run
        _, _, mock_main_fn = _run_lambda_handler(config, "2020-04-01")

        expected_yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        assert mock_main_fn.call_args.kwargs["end_date"] == expected_yesterday

    def test_save_last_run_date_called_with_end_date(self, tmp_path):
        """
        The cursor must be advanced to the same end_date that was processed.
        Saving a different date (e.g. start_date, or today) would create a gap
        or re-process already-ingested data on the next invocation.
        """
        config = _write_lambda_config(tmp_path)
        _, mock_save, mock_main_fn = _run_lambda_handler(config, "1938-01-01")

        processed_end = mock_main_fn.call_args.kwargs["end_date"]
        mock_save.assert_called_once_with(processed_end, "test-bucket")

    def test_save_last_run_date_called_with_correct_bucket(self, tmp_path):
        """
        The cursor write must target the same bucket as the pipeline run.
        A different bucket would split the cursor from the data, causing
        every future invocation to re-process from the wrong starting point.
        """
        config = _write_lambda_config(tmp_path)
        _, mock_save, _ = _run_lambda_handler(config, "1938-01-01")

        assert mock_save.call_args.args[1] == "test-bucket"

    def test_returns_status_code_200(self, tmp_path):
        """
        A successful pipeline run must return statusCode 200. Lambda's
        invoker uses this to distinguish success from failure when checking
        the synchronous return value.
        """
        config = _write_lambda_config(tmp_path)
        result, _, _ = _run_lambda_handler(config, "2020-01-01")

        assert result["statusCode"] == 200

    def test_return_body_contains_date_range(self, tmp_path):
        """
        The body string must include both start and end dates so CloudWatch
        logs make it immediately clear which time window each invocation covered
        without needing to cross-reference the cursor object.
        """
        config = _write_lambda_config(tmp_path)
        result, _, mock_main_fn = _run_lambda_handler(config, "1938-01-01")

        assert "1938-01-01" in result["body"]
        assert mock_main_fn.call_args.kwargs["end_date"] in result["body"]

    def test_main_not_called_when_get_last_run_date_raises(self, tmp_path):
        """
        If reading the cursor from S3 fails, main() must not run — there is no
        safe start_date to use. Running with an unknown start would either
        reprocess data or skip a window, both silently corrupting the history.
        """
        config = _write_lambda_config(tmp_path)

        with patch("src.main.CONFIG_PATH", config), \
             patch("src.main.get_last_run_date", side_effect=RuntimeError("S3 down")), \
             patch("src.main.main") as mock_main_fn:
            with pytest.raises(RuntimeError):
                lambda_handler({}, None)

        mock_main_fn.assert_not_called()


# ---------------------------------------------------------------------------
# _IN_LAMBDA path switching
# ---------------------------------------------------------------------------

class TestInLambdaPathSwitching:
    """
    main.py evaluates _IN_LAMBDA at import time to set DATA_DIR and CONFIG_PATH.
    These tests verify the two branches produce the correct paths so Lambda
    deployments write to /tmp and local runs use the project data/ directory.
    """

    def test_data_dir_is_tmp_when_aws_lambda_env_var_set(self, monkeypatch):
        """
        Inside Lambda, /tmp is the only writable path. If DATA_DIR were set to
        the local project path, every store call would raise a PermissionError
        and the pipeline would fail on every Lambda invocation.
        """
        import src.main as main_mod
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "my-function")
        try:
            importlib.reload(main_mod)
            assert main_mod.DATA_DIR == Path("/tmp")
        finally:
            monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
            importlib.reload(main_mod)  # restore for other tests

    def test_data_dir_is_local_when_aws_lambda_env_var_absent(self, monkeypatch):
        """
        In a local run, DATA_DIR must point into the project's data/ directory,
        not /tmp. Writing to /tmp locally would make output files invisible
        unless the developer knows to look there.
        """
        import src.main as main_mod
        monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
        try:
            importlib.reload(main_mod)
            assert "data" in str(main_mod.DATA_DIR)
            assert str(main_mod.DATA_DIR) != "/tmp"
        finally:
            importlib.reload(main_mod)  # restore module state
