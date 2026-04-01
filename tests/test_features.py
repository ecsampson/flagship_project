import pandas as pd
import pytest
from src.features import generate_weather_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grouped(datatype, values, start_date="2024-01-01"):
    """Build a grouped_data dict with a single datatype and a list of values.

    Dates are generated as consecutive days from start_date. This keeps test
    data minimal and the date arithmetic easy to reason about.
    """
    dates = pd.date_range(start_date, periods=len(values), freq="D")
    rows = [
        {"date": str(d.date()), "type": datatype, "value": v}
        for d, v in zip(dates, values)
    ]
    return {datatype: rows}


THRESHOLDS = {
    "TMAX": {"above": 35.0},
    "TMIN": {"below": -20.0},
}


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_output_has_required_columns():
    """
    Downstream storage and star-schema models depend on all nine columns
    being present. A missing column raises a KeyError in production; an
    extra column is harmless but signals something went wrong.
    """
    grouped = make_grouped("TMAX", [30.0, 31.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    assert set(df.columns) == {
        "date", "datatype", "value",
        "rolling_avg_7d", "rolling_avg_30d",
        "deviation", "is_extreme", "severity_score", "consecutive_days",
    }


def test_output_row_count_matches_input():
    """
    Each non-null observation must appear exactly once. Extra rows mean a
    join went wrong; fewer rows mean data was silently dropped.
    """
    grouped = make_grouped("TMAX", [28.0, 30.0, 32.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert len(df) == 3


def test_null_values_are_dropped():
    """
    NOAA records with None values are intentionally excluded — a null
    temperature would corrupt rolling averages and severity scores. Verify
    the row count reflects the drop, not the original input length.
    """
    grouped = make_grouped("TMAX", [30.0, None, 32.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert len(df) == 2


def test_output_is_sorted_by_datatype_then_date():
    """
    Rolling windows depend on chronological order within each datatype.
    Out-of-order rows would make rolling_avg_7d look back into the future.
    Also verifies that records from different datatypes don't interleave.
    """
    grouped = {
        "TMAX": [
            {"date": "2024-01-03", "type": "TMAX", "value": 30.0},
            {"date": "2024-01-01", "type": "TMAX", "value": 28.0},
        ],
        "PRCP": [
            {"date": "2024-01-02", "type": "PRCP", "value": 5.0},
        ],
    }
    df = generate_weather_signals(grouped, THRESHOLDS)

    prcp_rows = df[df["datatype"] == "PRCP"]
    tmax_rows = df[df["datatype"] == "TMAX"]

    # Within TMAX the earlier date must come first
    assert tmax_rows.iloc[0]["date"] < tmax_rows.iloc[1]["date"]
    # All PRCP rows are contiguous (no TMAX rows mixed in)
    assert prcp_rows.index.tolist() == list(range(prcp_rows.index[0], prcp_rows.index[-1] + 1))


# ---------------------------------------------------------------------------
# Rolling averages
# ---------------------------------------------------------------------------

def test_rolling_avg_7d_single_row_equals_value():
    """
    With min_periods=1 a single observation's 7-day average is itself.
    If min_periods were not set, the result would be NaN and every
    subsequent feature derived from it would also be NaN.
    """
    grouped = make_grouped("TMAX", [30.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[0]["rolling_avg_7d"] == pytest.approx(30.0)


def test_rolling_avg_7d_correct_after_full_window():
    """
    After 7 days of [10, 20, 30, 40, 50, 60, 70] the 7-day average on
    day 7 must be exactly 40.0. A wrong window size (e.g. 8) would give
    a different value with no error.
    """
    grouped = make_grouped("TMAX", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[6]["rolling_avg_7d"] == pytest.approx(40.0)


def test_rolling_avg_30d_single_row_equals_value():
    """Same min_periods=1 contract for the 30-day window."""
    grouped = make_grouped("TMAX", [25.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[0]["rolling_avg_30d"] == pytest.approx(25.0)


def test_rolling_averages_are_computed_per_datatype():
    """
    Rolling windows must not cross datatype boundaries. If TMAX and TMIN
    records were averaged together, both the window size and the units would
    be wrong.
    """
    grouped = {
        "TMAX": [
            {"date": "2024-01-01", "type": "TMAX", "value": 100.0},
            {"date": "2024-01-02", "type": "TMAX", "value": 100.0},
        ],
        "TMIN": [
            {"date": "2024-01-01", "type": "TMIN", "value": 0.0},
            {"date": "2024-01-02", "type": "TMIN", "value": 0.0},
        ],
    }
    df = generate_weather_signals(grouped, THRESHOLDS)

    tmax_avgs = df[df["datatype"] == "TMAX"]["rolling_avg_7d"]
    tmin_avgs = df[df["datatype"] == "TMIN"]["rolling_avg_7d"]

    assert list(tmax_avgs) == pytest.approx([100.0, 100.0])
    assert list(tmin_avgs) == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# Deviation
# ---------------------------------------------------------------------------

def test_deviation_is_value_minus_rolling_avg_7d():
    """
    deviation = value - rolling_avg_7d. On day 1 both are the same value,
    so deviation must be 0.0. After the window fills in, a spike above the
    average should produce a positive deviation.
    """
    grouped = make_grouped("TMAX", [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 40.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    # First 7 days: value == rolling avg → deviation == 0
    assert df.iloc[0]["deviation"] == pytest.approx(0.0)
    # Day 8: value=40, rolling avg = (30*6 + 40) / 7 ≈ 31.43 → deviation ≈ 8.57
    assert df.iloc[7]["deviation"] == pytest.approx(40.0 - df.iloc[7]["rolling_avg_7d"])


# ---------------------------------------------------------------------------
# is_extreme
# ---------------------------------------------------------------------------

def test_is_extreme_true_for_value_above_threshold():
    """40°C exceeds TMAX threshold of 35°C — must be flagged."""
    grouped = make_grouped("TMAX", [40.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[0]["is_extreme"] == True


def test_is_extreme_true_for_value_below_threshold():
    """-25°C is below TMIN threshold of -20°C — must be flagged."""
    grouped = make_grouped("TMIN", [-25.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[0]["is_extreme"] == True


def test_is_extreme_false_for_value_within_normal_range():
    """20°C is under the 35°C TMAX threshold — must not be flagged."""
    grouped = make_grouped("TMAX", [20.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[0]["is_extreme"] == False


def test_is_extreme_false_for_value_exactly_at_threshold():
    """
    The check is strictly greater/less than (not >=). A value equal to the
    threshold sits on the boundary and must not be flagged. Getting this
    wrong adds false positives right at the threshold boundary.
    """
    grouped = make_grouped("TMAX", [35.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[0]["is_extreme"] == False


def test_is_extreme_false_for_datatype_not_in_thresholds():
    """
    PRCP has data but no threshold. Any value, no matter how large, must
    not be flagged. A missing key should silently skip, not crash or assume
    extremeness.
    """
    grouped = make_grouped("PRCP", [999.0])
    df = generate_weather_signals(grouped, {})
    assert df.iloc[0]["is_extreme"] == False


# ---------------------------------------------------------------------------
# severity_score
# ---------------------------------------------------------------------------

def test_severity_score_zero_for_non_extreme():
    """Non-extreme rows must have severity_score=0.0 — they have no extremeness to score."""
    grouped = make_grouped("TMAX", [20.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert df.iloc[0]["severity_score"] == pytest.approx(0.0)


def test_severity_score_one_for_worst_above_extreme():
    """
    When the only extreme value equals the observed maximum, the normalised
    score should be 1.0 — the worst recorded value always scores 1.
    Formula: (value - threshold) / (max - threshold) = (40-35)/(40-35) = 1.0
    """
    grouped = make_grouped("TMAX", [20.0, 40.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    extreme_row = df[df["value"] == 40.0].iloc[0]
    assert extreme_row["severity_score"] == pytest.approx(1.0)


def test_severity_score_fractional_for_partial_above_extreme():
    """
    With threshold=35, max=45, a value of 40 should score 0.5.
    Formula: (40-35)/(45-35) = 5/10 = 0.5
    This verifies the linear scaling between threshold (0) and max (1).
    """
    grouped = make_grouped("TMAX", [20.0, 40.0, 45.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    row_40 = df[df["value"] == 40.0].iloc[0]
    assert row_40["severity_score"] == pytest.approx(0.5)


def test_severity_score_for_below_threshold():
    """
    With threshold=-20, min=-30, a value of -25 should score 0.5.
    Formula: (threshold - value) / (threshold - min) = (-20 - -25) / (-20 - -30) = 5/10 = 0.5
    """
    thresholds = {"TMIN": {"below": -20.0}}
    grouped = make_grouped("TMIN", [0.0, -25.0, -30.0])
    df = generate_weather_signals(grouped, thresholds)

    row = df[df["value"] == -25.0].iloc[0]
    assert row["severity_score"] == pytest.approx(0.5)


def test_severity_score_capped_at_one():
    """
    Scores are clipped to 1.0. If a new record comes in that exceeds the
    historical maximum, the score must not exceed 1.0 — an unbounded score
    would break any dashboard that assumes [0, 1] range.
    """
    # Only one extreme value exists — denom = max - threshold = 40-35 = 5
    # Score = (40-35)/5 = 1.0. With clip this stays at 1.0.
    grouped = make_grouped("TMAX", [36.0, 40.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    assert (df["severity_score"] <= 1.0).all()


# ---------------------------------------------------------------------------
# consecutive_days
# ---------------------------------------------------------------------------

def test_consecutive_days_zero_for_non_extreme():
    """Non-extreme rows must have consecutive_days=0."""
    grouped = make_grouped("TMAX", [20.0, 22.0])
    df = generate_weather_signals(grouped, THRESHOLDS)
    assert (df["consecutive_days"] == 0).all()


def test_consecutive_days_one_for_single_extreme_day():
    """An isolated extreme event with normal days either side scores 1."""
    grouped = make_grouped("TMAX", [20.0, 40.0, 20.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    extreme_row = df[df["value"] == 40.0].iloc[0]
    assert extreme_row["consecutive_days"] == 1


def test_consecutive_days_increments_through_streak():
    """
    Three consecutive extreme days must score 1, 2, 3 in order.
    The cumulative count within a streak is what tells analysts how long
    a heatwave has been running — a flat count would lose that information.
    """
    grouped = make_grouped("TMAX", [40.0, 41.0, 42.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    assert list(df["consecutive_days"]) == [1, 2, 3]


def test_consecutive_days_resets_after_break():
    """
    A normal day between two extreme days must reset the streak counter.
    Pattern: extreme, normal, extreme → consecutive_days = [1, 0, 1].
    Without a reset, the second event would be counted as part of the first
    streak, misrepresenting a new heatwave as a continuation of the prior one.
    """
    grouped = make_grouped("TMAX", [40.0, 20.0, 40.0])
    df = generate_weather_signals(grouped, THRESHOLDS)

    assert list(df["consecutive_days"]) == [1, 0, 1]


def test_consecutive_days_independent_per_datatype():
    """
    A TMIN streak and a TMAX streak must be counted separately. If they
    shared a counter, a break in one datatype would incorrectly reset the
    streak in the other.
    """
    grouped = {
        "TMAX": [
            {"date": "2024-01-01", "type": "TMAX", "value": 40.0},
            {"date": "2024-01-02", "type": "TMAX", "value": 41.0},
        ],
        "TMIN": [
            {"date": "2024-01-01", "type": "TMIN", "value": -25.0},
            {"date": "2024-01-02", "type": "TMIN", "value": 0.0},   # streak break
        ],
    }
    df = generate_weather_signals(grouped, THRESHOLDS)

    tmax_days = list(df[df["datatype"] == "TMAX"]["consecutive_days"])
    tmin_days = list(df[df["datatype"] == "TMIN"]["consecutive_days"])

    assert tmax_days == [1, 2]   # unbroken streak
    assert tmin_days == [1, 0]   # streak broken on day 2
