"""
Unit tests for main.py behaviors introduced by the station-ID naming and
dual CSV+Parquet output changes.

Strategy: patch CONFIG_PATH to a real (but minimal) temp yaml file, patch
DATA_DIR to a temp directory, and mock fetch_all_noaa_data + upload_to_s3
so no network calls are made. The store functions run against the real
filesystem, which also exercises the parquet write path end-to-end.
"""

import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.main import main


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
         patch("src.main.upload_to_s3"):
        main()


def _run_main_capture_uploads(config_path, data_dir):
    """Run main() and return the mock upload_to_s3 so call args can be inspected."""
    with patch("src.main.CONFIG_PATH", config_path), \
         patch("src.main.DATA_DIR", data_dir), \
         patch("src.main.fetch_all_noaa_data", return_value=_RAW_RECORDS), \
         patch("src.main.upload_to_s3") as mock_upload:
        main()
    return mock_upload


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
        6 outputs × 2 formats = 12 upload calls. Fewer than 12 means at least
        one format was skipped for at least one output; more than 12 means
        files are being double-uploaded.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        mock_upload = _run_main_capture_uploads(config, data_dir)

        assert mock_upload.call_count == 12

    def test_six_csv_and_six_parquet_files_uploaded(self, tmp_path):
        """
        Exactly 6 CSV and 6 Parquet uploads. This catches an imbalance where
        one format receives more uploads than the other (e.g. a file uploaded
        twice in one format but zero times in the other).
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = _write_config(tmp_path)
        mock_upload = _run_main_capture_uploads(config, data_dir)

        uploaded_paths = [str(c.args[0]) for c in mock_upload.call_args_list]
        csv_uploads = [p for p in uploaded_paths if p.endswith(".csv")]
        parquet_uploads = [p for p in uploaded_paths if p.endswith(".parquet")]

        assert len(csv_uploads) == 6
        assert len(parquet_uploads) == 6

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
             patch("src.main.upload_to_s3") as mock_upload:
            main()

        assert mock_upload.call_count == 0
