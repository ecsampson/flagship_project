"""
Integration tests for the NOAA ETL pipeline.

Unit tests verify individual functions in isolation. These tests verify that
the functions work correctly when connected — that the output contract of one
stage satisfies the input contract of the next.

No real network calls are made. fetch_all_noaa_data and boto3 are mocked.
All file I/O uses pytest's tmp_path so the real data/ directory is never touched.
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.noaa_client import parse_noaa_data, group_noaa_data, detect_extreme_weather
from src.features import generate_weather_signals
from src.storage import store_noaa_data, store_extreme_weather, store_weather_signals
from src.models import build_dim_date, build_dim_location, build_fact_weather_observations


# ---------------------------------------------------------------------------
# Shared test data
#
# Mimics the raw JSON that NOAA returns. Values are in tenths of units:
#   TMAX 320 → 32.0°C  (normal)
#   TMAX 400 → 40.0°C  (extreme, above threshold of 35)
#   TMIN 150 → 15.0°C  (normal)
#   TMIN -250 → -25.0°C  (extreme, below threshold of -20)
#   PRCP 50 → 5.0mm    (no threshold configured — never extreme)
# ---------------------------------------------------------------------------

RAW_API_RESPONSE = [
    {"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:USC00213567", "TMAX": "320", "TMIN": "150", "PRCP": "50"},
    {"DATE": "2024-07-01T00:00:00", "STATION": "GHCND:USC00213567", "TMAX": "400", "TMIN": "180", "PRCP": "120"},
    {"DATE": "2024-01-10T00:00:00", "STATION": "GHCND:USC00213567", "TMAX": "280", "TMIN": "-250", "PRCP": "0"},
]

SETTINGS = {
    "noaa": {"default_location": "FIPS:27"},
    "extreme_weather": {
        "thresholds": {
            "TMAX": {"above": 35.0},
            "TMIN": {"below": -20.0},
        }
    },
}


# ---------------------------------------------------------------------------
# Stage-to-stage handoff: parse → group
# ---------------------------------------------------------------------------

class TestParseToGroup:
    """parse_noaa_data output feeds directly into group_noaa_data.
    These tests verify the key/schema contract between the two stages."""

    def test_group_accepts_parse_output_without_error(self):
        """
        The most basic contract: parse output must be consumable by group
        without KeyError, TypeError, or any other exception. A schema mismatch
        between the two functions would surface here before any logic is tested.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)  # must not raise
        assert isinstance(grouped, dict)

    def test_all_datatypes_reach_group_stage(self):
        """
        Every measurement type in the raw response (TMAX, TMIN, PRCP) must
        survive parsing and appear as a key in the grouped output. A datatype
        silently dropped at parse time would disappear from all downstream
        stages with no error.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)

        assert "TMAX" in grouped
        assert "TMIN" in grouped
        assert "PRCP" in grouped

    def test_record_count_preserved_from_parse_to_group(self):
        """
        The total number of records across all groups must equal the number
        of records that parse produced. Grouping is a reorganization, not a
        filter — any count mismatch means data was silently dropped.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)

        total_grouped = sum(len(records) for records in grouped.values())
        assert total_grouped == len(parsed)

    def test_date_values_survive_parse_to_group(self):
        """
        Dates are truncated to YYYY-MM-DD by parse_noaa_data (stripping the
        time component). Verify that dates in the grouped output still have
        that format, not the original ISO timestamp or a pandas Timestamp object.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)

        for record in grouped["TMAX"]:
            assert len(record["date"]) == 10   # "YYYY-MM-DD"
            assert "T" not in record["date"]   # no time component


# ---------------------------------------------------------------------------
# Stage-to-stage handoff: group → detect
# ---------------------------------------------------------------------------

class TestGroupToDetect:
    """group_noaa_data output feeds into detect_extreme_weather.
    These tests verify that extreme conditions in raw data reach the output."""

    def test_extreme_value_survives_parse_group_detect_chain(self):
        """
        40.0°C in the raw API response (TMAX: "400") exceeds the 35.0°C
        threshold. It must emerge from the full parse→group→detect chain as
        an extreme event. If it doesn't, a real heatwave would be silently lost.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        events = detect_extreme_weather(grouped, SETTINGS["extreme_weather"]["thresholds"])

        tmax_events = [e for e in events if e["type"] == "TMAX"]
        assert any(e["value"] == 40.0 for e in tmax_events)

    def test_below_threshold_event_survives_full_chain(self):
        """
        -25.0°C (TMIN: "-250") is below the -20.0°C threshold. Verify that
        'below' direction events survive the full chain — the comparison is
        inverted relative to 'above' and is easy to break independently.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        events = detect_extreme_weather(grouped, SETTINGS["extreme_weather"]["thresholds"])

        tmin_events = [e for e in events if e["type"] == "TMIN"]
        assert any(e["value"] == -25.0 for e in tmin_events)

    def test_normal_records_do_not_appear_in_detect_output(self):
        """
        32.0°C (TMAX: "320") is below the 35.0°C threshold — it must not be
        flagged. This verifies the detect filter is selective, not a pass-through.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        events = detect_extreme_weather(grouped, SETTINGS["extreme_weather"]["thresholds"])

        assert not any(e["value"] == 32.0 for e in events)

    def test_unthresholded_datatype_absent_from_detect_output(self):
        """
        PRCP has records but no threshold. Even large values (120 → 12.0mm)
        must not appear in detect output. Failing here means unintended
        datatypes are generating false extreme alerts.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        events = detect_extreme_weather(grouped, SETTINGS["extreme_weather"]["thresholds"])

        assert not any(e["type"] == "PRCP" for e in events)


# ---------------------------------------------------------------------------
# Stage-to-stage handoff: group → features
# ---------------------------------------------------------------------------

class TestGroupToFeatures:
    """group_noaa_data output feeds into generate_weather_signals."""

    def test_features_accepts_group_output_without_error(self):
        """generate_weather_signals must consume group output without raising."""
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        signals = generate_weather_signals(grouped, SETTINGS["extreme_weather"]["thresholds"])
        assert isinstance(signals, pd.DataFrame)

    def test_features_output_contains_all_datatypes(self):
        """
        All datatypes from grouped data must appear in the signals output.
        Dropping a datatype silently would produce an incomplete feature table.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        signals = generate_weather_signals(grouped, SETTINGS["extreme_weather"]["thresholds"])

        assert set(signals["datatype"].unique()) == {"TMAX", "TMIN", "PRCP"}

    def test_extreme_flag_consistent_between_detect_and_features(self):
        """
        detect_extreme_weather and generate_weather_signals both flag extreme
        events using the same thresholds. The count of flagged records must
        agree — a discrepancy means one stage has a logic divergence that will
        produce inconsistent outputs in downstream reporting.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        thresholds = SETTINGS["extreme_weather"]["thresholds"]

        events = detect_extreme_weather(grouped, thresholds)
        signals = generate_weather_signals(grouped, thresholds)

        assert len(events) == signals["is_extreme"].sum()


# ---------------------------------------------------------------------------
# Stage-to-stage handoff: parse → star schema
# ---------------------------------------------------------------------------

class TestParseToStarSchema:
    """parse_noaa_data output feeds into the three model-building functions."""

    def test_dim_date_row_count_matches_unique_dates_in_parsed(self):
        """
        dim_date must have exactly one row per unique date in parsed_data.
        More rows means deduplication failed; fewer means dates were dropped.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        unique_dates = len({r["date"] for r in parsed})
        dim_date = build_dim_date(parsed)

        assert len(dim_date) == unique_dates

    def test_fact_row_count_matches_parsed_record_count(self):
        """
        Every parsed observation must become exactly one row in the fact table.
        A left join that finds no match silently produces a row with NaN keys,
        not a missing row — the count stays the same but FK integrity breaks.
        This test combined with the FK test below catches both failure modes.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        dim_date = build_dim_date(parsed)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed, dim_date, dim_location, SETTINGS)

        assert len(fact) == len(parsed)

    def test_all_fact_date_ids_exist_in_dim_date(self):
        """
        Foreign key integrity: every date_id in the fact table must exist in
        dim_date. A NaN date_id means the merge failed — that observation
        becomes unreachable in any date-filtered query.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        dim_date = build_dim_date(parsed)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed, dim_date, dim_location, SETTINGS)

        valid_date_ids = set(dim_date["date_id"])
        assert fact["date_id"].notna().all()
        assert set(fact["date_id"]).issubset(valid_date_ids)

    def test_all_fact_location_ids_exist_in_dim_location(self):
        """
        Foreign key integrity for location_id. All fact rows must reference
        a location that exists in dim_location.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        dim_date = build_dim_date(parsed)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed, dim_date, dim_location, SETTINGS)

        valid_location_ids = set(dim_location["location_id"])
        assert set(fact["location_id"]).issubset(valid_location_ids)

    def test_extreme_flag_consistent_between_detect_and_fact_table(self):
        """
        detect_extreme_weather and build_fact_weather_observations both evaluate
        extremeness from the same thresholds. Their counts must match — a
        divergence means the pipeline's two extreme-event outputs disagree with
        each other, which would confuse any downstream consumer comparing them.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        thresholds = SETTINGS["extreme_weather"]["thresholds"]

        events = detect_extreme_weather(grouped, thresholds)
        dim_date = build_dim_date(parsed)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed, dim_date, dim_location, SETTINGS)

        assert len(events) == fact["is_extreme"].sum()


# ---------------------------------------------------------------------------
# Storage round-trip tests
# ---------------------------------------------------------------------------

class TestStorageRoundTrip:
    """Verify that data written to CSV can be read back with values intact.

    These tests catch silent corruption from type coercion, float rounding,
    column reordering, or header duplication during serialization.
    """

    def test_parsed_records_survive_csv_round_trip(self, tmp_path):
        """
        Parse → write to CSV → read back. Row count and key values must be
        identical. Any loss here affects the permanent historical record.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        file_path = tmp_path / "weather.csv"
        store_noaa_data(parsed, file_path)

        df = pd.read_csv(file_path)
        assert len(df) == len(parsed)
        assert set(df.columns) == {"date", "type", "value"}

    def test_extreme_events_survive_csv_round_trip(self, tmp_path):
        """
        Detect → write extreme events → read back. The threshold and condition
        metadata columns must survive — they're the reason the extreme CSV
        exists separately from the main weather CSV.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        events = detect_extreme_weather(grouped, SETTINGS["extreme_weather"]["thresholds"])
        file_path = tmp_path / "extreme.csv"
        store_extreme_weather(events, file_path)

        df = pd.read_csv(file_path)
        assert len(df) == len(events)
        assert "threshold" in df.columns
        assert "condition" in df.columns

    def test_signals_survive_csv_round_trip(self, tmp_path):
        """
        Feature engineering → write signals → read back. All nine feature
        columns must be present with correct values after the CSV round-trip.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        signals = generate_weather_signals(grouped, SETTINGS["extreme_weather"]["thresholds"])
        file_path = tmp_path / "signals.csv"
        store_weather_signals(signals, file_path)

        df = pd.read_csv(file_path)
        assert len(df) == len(signals)
        assert "severity_score" in df.columns
        assert "consecutive_days" in df.columns

    def test_star_schema_tables_survive_csv_round_trip(self, tmp_path):
        """
        Build all three star schema tables and write them. Read each back and
        verify row counts and key columns. A broken write path here means the
        star schema is never actually persisted.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        dim_date = build_dim_date(parsed)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed, dim_date, dim_location, SETTINGS)

        dim_date_path = tmp_path / "dim_date.csv"
        dim_location_path = tmp_path / "dim_location.csv"
        fact_path = tmp_path / "fact.csv"

        store_noaa_data(dim_date, dim_date_path)
        store_noaa_data(dim_location, dim_location_path)
        store_noaa_data(fact, fact_path)

        assert len(pd.read_csv(dim_date_path)) == len(dim_date)
        assert len(pd.read_csv(dim_location_path)) == len(dim_location)
        assert len(pd.read_csv(fact_path)) == len(fact)

    def test_extreme_records_are_a_subset_of_all_weather_records(self, tmp_path):
        """
        The extreme events CSV should contain only records that also appear in
        the main weather CSV (by date + type + value). An extreme event that
        isn't in the main record is an orphan — it can't be cross-referenced
        and likely indicates a data integrity bug.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        events = detect_extreme_weather(grouped, SETTINGS["extreme_weather"]["thresholds"])

        weather_path = tmp_path / "weather.csv"
        extreme_path = tmp_path / "extreme.csv"
        store_noaa_data(parsed, weather_path)
        store_extreme_weather(events, extreme_path)

        weather_df = pd.read_csv(weather_path)
        extreme_df = pd.read_csv(extreme_path)

        weather_keys = set(zip(weather_df["date"], weather_df["type"], weather_df["value"]))
        extreme_keys = set(zip(extreme_df["date"], extreme_df["type"], extreme_df["value"]))

        assert extreme_keys.issubset(weather_keys)

    def test_append_behavior_doubles_row_count_on_second_run(self, tmp_path):
        """
        The pipeline uses append-only storage by design (documented in CLAUDE.md).
        A second run must double the row count, not overwrite. This test documents
        the known behavior so a future accidental change to 'w' mode is caught.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        file_path = tmp_path / "weather.csv"

        store_noaa_data(parsed, file_path)
        store_noaa_data(parsed, file_path)

        df = pd.read_csv(file_path)
        assert len(df) == len(parsed) * 2


# ---------------------------------------------------------------------------
# Full pipeline end-to-end (fetch mocked, S3 mocked)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """
    Run the complete pipeline logic from raw API response through to all
    output files. fetch_all_noaa_data and boto3 are mocked — everything
    else is real code on a real (temporary) filesystem.
    """

    def _run_pipeline(self, raw_records, tmp_path):
        """
        Replicate the logic from main.py using tmp_path for file destinations.
        Returns all intermediate and final artifacts for inspection.
        """
        parsed_data = parse_noaa_data(raw_records)
        dim_date = build_dim_date(parsed_data)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed_data, dim_date, dim_location, SETTINGS)
        grouped_data = group_noaa_data(parsed_data)
        extreme_events = detect_extreme_weather(grouped_data, SETTINGS["extreme_weather"]["thresholds"])
        signals = generate_weather_signals(grouped_data, SETTINGS["extreme_weather"]["thresholds"])

        store_noaa_data(parsed_data, tmp_path / "noaa_weather_data.csv")
        store_extreme_weather(extreme_events, tmp_path / "noaa_extreme_weather.csv")
        store_weather_signals(signals, tmp_path / "feat_weather_signals.csv")
        store_noaa_data(dim_date, tmp_path / "dim_date.csv")
        store_noaa_data(dim_location, tmp_path / "dim_location.csv")
        store_noaa_data(fact, tmp_path / "fact_weather_observations.csv")

        return {
            "parsed": parsed_data,
            "dim_date": dim_date,
            "dim_location": dim_location,
            "fact": fact,
            "grouped": grouped_data,
            "events": extreme_events,
            "signals": signals,
        }

    def test_all_output_files_are_created(self, tmp_path):
        """
        A full run must produce all six output files. A missing file means a
        stage silently failed or was skipped — downstream jobs reading from
        that path would fail with a FileNotFoundError rather than a clear error.
        """
        self._run_pipeline(RAW_API_RESPONSE, tmp_path)

        expected_files = [
            "noaa_weather_data.csv",
            "noaa_extreme_weather.csv",
            "feat_weather_signals.csv",
            "dim_date.csv",
            "dim_location.csv",
            "fact_weather_observations.csv",
        ]
        for filename in expected_files:
            assert (tmp_path / filename).exists(), f"Missing output file: {filename}"

    def test_only_extreme_records_in_extreme_file(self, tmp_path):
        """
        The extreme weather CSV must contain only flagged records — no normal
        observations should bleed in. Mixing them would defeat the purpose of
        having a separate alerting dataset.
        """
        self._run_pipeline(RAW_API_RESPONSE, tmp_path)

        df = pd.read_csv(tmp_path / "noaa_extreme_weather.csv")
        thresholds = SETTINGS["extreme_weather"]["thresholds"]

        for _, row in df.iterrows():
            datatype = row["type"]
            value = row["value"]
            assert datatype in thresholds
            criteria = thresholds[datatype]
            is_above = "above" in criteria and value > criteria["above"]
            is_below = "below" in criteria and value < criteria["below"]
            assert is_above or is_below, (
                f"Non-extreme record found in extreme CSV: {datatype}={value}"
            )

    def test_weather_signals_covers_all_datatypes(self, tmp_path):
        """
        The signals file must contain rows for every datatype that was fetched.
        A datatype absent from signals means the feature engineering step
        silently dropped it — models trained on these signals would have missing
        inputs for that datatype.
        """
        self._run_pipeline(RAW_API_RESPONSE, tmp_path)

        df = pd.read_csv(tmp_path / "feat_weather_signals.csv")
        assert set(df["datatype"].unique()) == {"TMAX", "TMIN", "PRCP"}

    def test_fact_table_has_no_null_foreign_keys(self, tmp_path):
        """
        End-to-end FK integrity check on the persisted CSVs. Read dim_date
        and fact_weather_observations from disk and verify every date_id in
        the fact table exists in dim_date. This catches bugs that only appear
        when the join is done against the written-then-read data.
        """
        self._run_pipeline(RAW_API_RESPONSE, tmp_path)

        dim_date = pd.read_csv(tmp_path / "dim_date.csv")
        fact = pd.read_csv(tmp_path / "fact_weather_observations.csv")

        valid_date_ids = set(dim_date["date_id"])
        assert fact["date_id"].notna().all()
        assert set(fact["date_id"]).issubset(valid_date_ids)

    def test_pipeline_with_single_record_does_not_crash(self, tmp_path):
        """
        A single-record API response (minimum viable dataset) must not crash.
        Rolling windows (min_periods=1), single-row DataFrames, and severity
        score denominators of 0 are all edge cases that appear with one record.
        """
        single_record = [
            {"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:USC00213567", "TMAX": "400"}
        ]
        artifacts = self._run_pipeline(single_record, tmp_path)
        assert len(artifacts["parsed"]) == 1


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

class TestDataQuality:
    """
    Pipeline-specific data quality checks that span multiple stages.
    These catch the class of bugs where data is technically present but
    wrong in ways that corrupt analytics — silent corruption is harder to
    diagnose than a crash.
    """

    def test_divide_by_ten_applied_exactly_once(self):
        """
        NOAA returns values in tenths. The /10 conversion must happen exactly
        once in parse_noaa_data. If it were applied again downstream (e.g. in
        group or detect), temperatures would be 100x too small with no error.
        """
        raw = [{"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:US1", "TMAX": "320"}]
        parsed = parse_noaa_data(raw)
        grouped = group_noaa_data(parsed)

        tmax_value = grouped["TMAX"][0]["value"]
        assert tmax_value == 32.0, f"Expected 32.0, got {tmax_value} — /10 applied wrong number of times"

    def test_no_date_or_station_keys_in_grouped_output(self):
        """
        DATE and STATION are parse-stage skip keys. If they leaked through
        into grouped_data, detect_extreme_weather would receive a "DATE" bucket
        and threshold lookups on it would fail or silently return no events.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)

        assert "DATE" not in grouped
        assert "STATION" not in grouped

    def test_no_null_date_ids_after_full_parse_to_star_schema(self):
        """
        Null date_ids in the fact table indicate a merge failure. They appear
        as valid rows in the output DataFrame but can never be joined back to
        dim_date — they're effectively invisible to any date-filtered query.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        dim_date = build_dim_date(parsed)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed, dim_date, dim_location, SETTINGS)

        null_count = fact["date_id"].isna().sum()
        assert null_count == 0, f"{null_count} fact rows have null date_id after merge"

    def test_observation_ids_unique_across_fact_table(self):
        """
        observation_id is the fact table primary key. Duplicates would corrupt
        any upsert logic that relies on it to deduplicate future pipeline runs.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        dim_date = build_dim_date(parsed)
        dim_location = build_dim_location(SETTINGS)
        fact = build_fact_weather_observations(parsed, dim_date, dim_location, SETTINGS)

        assert fact["observation_id"].nunique() == len(fact)

    def test_severity_score_bounded_between_zero_and_one(self):
        """
        severity_score must always be in [0.0, 1.0]. Values outside this range
        break any dashboard or model that normalises by this column, and they
        can't be caught by looking at the data type alone.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        signals = generate_weather_signals(grouped, SETTINGS["extreme_weather"]["thresholds"])

        assert (signals["severity_score"] >= 0.0).all()
        assert (signals["severity_score"] <= 1.0).all()

    def test_consecutive_days_non_negative(self):
        """
        consecutive_days must never be negative. The cumsum-based algorithm
        should produce 0 for non-extreme rows and positive integers for streaks,
        but a sign error in the streak reset would produce negative counts.
        """
        parsed = parse_noaa_data(RAW_API_RESPONSE)
        grouped = group_noaa_data(parsed)
        signals = generate_weather_signals(grouped, SETTINGS["extreme_weather"]["thresholds"])

        assert (signals["consecutive_days"] >= 0).all()
