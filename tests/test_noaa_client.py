import pytest
from src.noaa_client import parse_noaa_data, group_noaa_data, detect_extreme_weather


# ---------------------------------------------------------------------------
# parse_noaa_data
# ---------------------------------------------------------------------------

def test_parse_extracts_date_type_value():
    """
    The core contract of parse_noaa_data: given a raw NOAA row, it should
    produce one record per non-skip key with date truncated to YYYY-MM-DD,
    the key as 'type', and the raw integer divided by 10 as 'value'.

    Dividing by 10 is critical — NOAA stores temperatures in tenths of
    degrees, so skipping this produces values 10x too large with no error.
    """
    raw = [{"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:US1", "TMAX": "320"}]
    result = parse_noaa_data(raw)

    assert len(result) == 1
    assert result[0]["date"] == "2024-01-15"
    assert result[0]["type"] == "TMAX"
    assert result[0]["value"] == 32.0


def test_parse_skips_date_and_station_keys():
    """
    DATE and STATION are metadata, not measurements. If they leak through
    as records, downstream grouping and threshold logic will receive garbage
    rows. Verify they are absent from the output entirely.
    """
    raw = [{"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:US1", "PRCP": "50"}]
    result = parse_noaa_data(raw)

    types_in_output = [r["type"] for r in result]
    assert "DATE" not in types_in_output
    assert "STATION" not in types_in_output


def test_parse_handles_none_value_without_crashing():
    """
    NOAA omits measurements when a station didn't record that day — the
    field is present in the JSON but set to None. The parser should produce
    a record with value=None rather than raising a TypeError, so downstream
    steps can decide how to handle missing data explicitly.
    """
    raw = [{"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:US1", "SNOW": None}]
    result = parse_noaa_data(raw)

    assert len(result) == 1
    assert result[0]["value"] is None


def test_parse_handles_non_numeric_string_without_crashing():
    """
    Occasionally NOAA returns a flag string instead of a number (e.g. 'T'
    for trace precipitation). The parser should treat these as None rather
    than raising a ValueError and killing the pipeline mid-run.
    """
    raw = [{"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:US1", "PRCP": "T"}]
    result = parse_noaa_data(raw)

    assert len(result) == 1
    assert result[0]["value"] is None


def test_parse_multiple_rows_produces_correct_count():
    """
    With two rows each having two measurements, parse should produce exactly
    four records. A common bug is accidentally sharing state across rows.
    """
    raw = [
        {"DATE": "2024-01-15T00:00:00", "STATION": "GHCND:US1", "TMAX": "300", "TMIN": "100"},
        {"DATE": "2024-01-16T00:00:00", "STATION": "GHCND:US1", "TMAX": "310", "TMIN": "110"},
    ]
    result = parse_noaa_data(raw)

    assert len(result) == 4


# ---------------------------------------------------------------------------
# group_noaa_data
# ---------------------------------------------------------------------------

def test_group_separates_records_by_datatype():
    """
    group_noaa_data is the fan-out step — every downstream operation
    (threshold checks, aggregations) works per data type. Verify that TMAX
    and PRCP land in separate buckets and don't bleed into each other.
    """
    parsed = [
        {"date": "2024-01-15", "type": "TMAX", "value": 30.0},
        {"date": "2024-01-15", "type": "PRCP", "value": 5.0},
    ]
    grouped = group_noaa_data(parsed)

    assert "TMAX" in grouped
    assert "PRCP" in grouped
    assert len(grouped["TMAX"]) == 1
    assert len(grouped["PRCP"]) == 1


def test_group_collects_multiple_records_for_same_datatype():
    """
    Multiple days of the same measurement must all end up in one list.
    If grouping silently dropped records, threshold detection would miss
    events and we'd have no way to know.
    """
    parsed = [
        {"date": "2024-01-15", "type": "TMAX", "value": 30.0},
        {"date": "2024-01-16", "type": "TMAX", "value": 35.0},
        {"date": "2024-01-17", "type": "TMAX", "value": 28.0},
    ]
    grouped = group_noaa_data(parsed)

    assert len(grouped["TMAX"]) == 3


def test_group_preserves_record_contents():
    """
    Grouping should not modify record values — only re-organize them.
    A bug that truncates dates or resets values here would corrupt every
    record silently.
    """
    record = {"date": "2024-01-15", "type": "SNOW", "value": 12.5}
    grouped = group_noaa_data([record])

    assert grouped["SNOW"][0] == record


def test_group_empty_input_returns_empty_dict():
    """
    An empty API response is a valid scenario (no data for the date range).
    The function should return an empty dict, not raise an exception.
    """
    assert group_noaa_data([]) == {}


# ---------------------------------------------------------------------------
# detect_extreme_weather
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "TMAX": {"above": 35.0},
    "TMIN": {"below": -20.0},
    "PRCP": {"above": 50.0},
}


def test_detect_flags_value_above_threshold():
    """
    A temperature of 40°C exceeds the TMAX threshold of 35°C. The function
    must return one event with condition='above' and threshold=35.0.
    Getting this wrong means real heatwave events go undetected.
    """
    grouped = {"TMAX": [{"date": "2024-07-01", "type": "TMAX", "value": 40.0}]}
    events = detect_extreme_weather(grouped, THRESHOLDS)

    assert len(events) == 1
    assert events[0]["condition"] == "above"
    assert events[0]["threshold"] == 35.0
    assert events[0]["value"] == 40.0


def test_detect_flags_value_below_threshold():
    """
    A temperature of -25°C is below the TMIN threshold of -20°C. The
    function must handle 'below' direction correctly — the inequality is
    reversed compared to 'above', which is an easy off-by-direction bug.
    """
    grouped = {"TMIN": [{"date": "2024-01-10", "type": "TMIN", "value": -25.0}]}
    events = detect_extreme_weather(grouped, THRESHOLDS)

    assert len(events) == 1
    assert events[0]["condition"] == "below"
    assert events[0]["threshold"] == -20.0


def test_detect_does_not_flag_value_within_normal_range():
    """
    A TMAX of 20°C is well under the 35°C threshold. No event should be
    emitted. False positives are as harmful as false negatives — they
    erode trust in the pipeline's output.
    """
    grouped = {"TMAX": [{"date": "2024-04-01", "type": "TMAX", "value": 20.0}]}
    events = detect_extreme_weather(grouped, THRESHOLDS)

    assert len(events) == 0


def test_detect_does_not_flag_value_exactly_at_threshold():
    """
    A value equal to the threshold is not extreme — the check is strictly
    greater/less than (> not >=). This boundary condition is easy to get
    wrong and changes which events get stored.
    """
    grouped = {"TMAX": [{"date": "2024-07-01", "type": "TMAX", "value": 35.0}]}
    events = detect_extreme_weather(grouped, THRESHOLDS)

    assert len(events) == 0


def test_detect_ignores_datatype_not_in_thresholds():
    """
    SNOW has data but no threshold configured. The function must silently
    skip it rather than raising a KeyError. Adding a new data type to the
    fetch should never break extreme weather detection.
    """
    grouped = {"SNOW": [{"date": "2024-01-10", "type": "SNOW", "value": 100.0}]}
    events = detect_extreme_weather(grouped, THRESHOLDS)

    assert len(events) == 0


def test_detect_returns_only_extreme_records_from_mixed_data():
    """
    When a mix of normal and extreme values exists for the same type, only
    the extreme ones should appear in the output. Verifies that the filter
    works record-by-record, not day-by-day or type-by-type.
    """
    grouped = {
        "TMAX": [
            {"date": "2024-07-01", "type": "TMAX", "value": 40.0},  # extreme
            {"date": "2024-07-02", "type": "TMAX", "value": 25.0},  # normal
            {"date": "2024-07-03", "type": "TMAX", "value": 38.0},  # extreme
        ]
    }
    events = detect_extreme_weather(grouped, THRESHOLDS)

    assert len(events) == 2
    assert all(e["value"] > 35.0 for e in events)
