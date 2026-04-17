import pandas as pd
import pytest
from src.models import build_dim_date, build_dim_location, build_fact_weather_observations


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PARSED_DATA = [
    {"date": "2024-01-15", "type": "TMAX", "value": 32.0},
    {"date": "2024-01-15", "type": "PRCP", "value": 5.0},
    {"date": "2024-01-20", "type": "TMAX", "value": 40.0},
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
# build_dim_date
# ---------------------------------------------------------------------------

def test_dim_date_one_row_per_unique_date():
    """
    Two records share the same date (2024-01-15). The dimension table must
    deduplicate — one row per date, not one row per observation. A duplicate
    date_id would corrupt every join in the fact table.
    """
    df = build_dim_date(PARSED_DATA)
    assert len(df) == 2


def test_dim_date_contains_required_columns():
    """
    Downstream joins depend on specific column names. A rename would break
    the merge in build_fact_weather_observations silently (it would produce
    all-NaN date_id values rather than raising an error).
    """
    df = build_dim_date(PARSED_DATA)
    assert set(df.columns) == {"date_id", "date", "year", "month", "day", "season", "is_weekend"}


def test_dim_date_date_id_is_unique():
    """
    date_id is the surrogate key. If two rows share a date_id the join
    produces a cartesian product, silently inflating the fact table.
    """
    df = build_dim_date(PARSED_DATA)
    assert df["date_id"].nunique() == len(df)


def test_dim_date_extracts_correct_calendar_fields():
    """
    Verify that year/month/day are derived from the date, not hardcoded or
    off-by-one. An off-by-one in month would mis-assign seasons for an
    entire month of data.
    """
    data = [{"date": "2024-07-04", "type": "TMAX", "value": 30.0}]
    df = build_dim_date(data)
    row = df.iloc[0]

    assert row["year"] == 2024
    assert row["month"] == 7
    assert row["day"] == 4


def test_dim_date_season_assignment():
    """
    Verify season boundaries for each quarter. These are used for seasonal
    aggregations — a wrong boundary shifts an entire month into the wrong
    reporting period.
    """
    data = [
        {"date": "2024-01-15", "type": "TMAX", "value": 1.0},   # Winter
        {"date": "2024-04-10", "type": "TMAX", "value": 1.0},   # Spring
        {"date": "2024-07-04", "type": "TMAX", "value": 1.0},   # Summer
        {"date": "2024-10-31", "type": "TMAX", "value": 1.0},   # Fall
    ]
    df = build_dim_date(data).set_index("month")

    assert df.loc[1, "season"] == "Winter"
    assert df.loc[4, "season"] == "Spring"
    assert df.loc[7, "season"] == "Summer"
    assert df.loc[10, "season"] == "Fall"


@pytest.mark.parametrize("date_str,expected_is_weekend", [
    ("2024-07-06", True),   # Saturday
    ("2024-07-07", True),   # Sunday
    ("2024-07-08", False),  # Monday
    ("2024-07-05", False),  # Friday
])
def test_dim_date_is_weekend_flag(date_str, expected_is_weekend):
    """
    is_weekend must be True only for Saturday and Sunday.
    A common bug is using dayofweek >= 6 (only Sunday) or <= 1 (Mon/Tue).
    """
    data = [{"date": date_str, "type": "TMAX", "value": 1.0}]
    df = build_dim_date(data)
    assert df.iloc[0]["is_weekend"] == expected_is_weekend


def test_dim_date_sorted_ascending():
    """
    Dates are fed in arbitrary order from the parser. The dimension table
    should be sorted so date_id=0 is always the earliest date, making it
    predictable for reporting and debugging.
    """
    data = [
        {"date": "2024-03-01", "type": "TMAX", "value": 1.0},
        {"date": "2024-01-01", "type": "TMAX", "value": 1.0},
        {"date": "2024-02-01", "type": "TMAX", "value": 1.0},
    ]
    df = build_dim_date(data)
    dates = pd.to_datetime(df["date"]).tolist()
    assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# build_dim_location
# ---------------------------------------------------------------------------

def test_dim_location_returns_single_row():
    """
    Location dimension for a single-station pipeline should always be
    exactly one row. Multiple rows would fan out every fact-table join.
    """
    df = build_dim_location(SETTINGS)
    assert len(df) == 1


def test_dim_location_contains_required_columns():
    """Downstream joins depend on location_id, fips_code, and the state metadata columns."""
    df = build_dim_location(SETTINGS)
    assert set(df.columns) == {"location_id", "fips_code", "state_name", "state_abbr", "region"}


def test_dim_location_parses_fips_prefix():
    """
    Settings store the location as "FIPS:27". The function must strip the
    "FIPS:" prefix and store only "27" as the fips_code — keeping the prefix
    would break any join or lookup against a bare FIPS code table.
    """
    df = build_dim_location(SETTINGS)
    assert df.iloc[0]["fips_code"] == "27"


def test_dim_location_resolves_known_fips_to_metadata():
    """
    FIPS code 27 maps to Minnesota in _LOCATION_LOOKUP. Verify that state
    name, abbreviation, and region are populated correctly and not left as
    the unknown-fallback values.
    """
    df = build_dim_location(SETTINGS)
    row = df.iloc[0]

    assert row["state_name"] == "Minnesota"
    assert row["state_abbr"] == "MN"
    assert row["region"] == "Midwest"


def test_dim_location_bare_station_id_without_prefix():
    """
    Station IDs like "USC00213567" have no "FIPS:" prefix. The function
    must handle both formats — splitting on ":" when present and passing
    the raw string through when not.
    """
    settings = {"noaa": {"default_location": "USC00213567"}}
    df = build_dim_location(settings)
    assert df.iloc[0]["fips_code"] == "USC00213567"


def test_dim_location_unknown_location_uses_fallback():
    """
    A location key not in _LOCATION_LOOKUP should not crash — it should
    produce a row with Unknown/? sentinel values. This allows the pipeline
    to run for new stations before their metadata is added to the lookup.
    """
    settings = {"noaa": {"default_location": "FIPS:99"}}
    df = build_dim_location(settings)
    row = df.iloc[0]

    assert row["state_name"] == "Unknown"
    assert row["state_abbr"] == "??"
    assert row["region"] == "Unknown"


# ---------------------------------------------------------------------------
# build_fact_weather_observations
# ---------------------------------------------------------------------------

@pytest.fixture
def dim_date():
    return build_dim_date(PARSED_DATA)


@pytest.fixture
def dim_location():
    return build_dim_location(SETTINGS)


def test_fact_one_row_per_observation(dim_date, dim_location):
    """
    Each (date, datatype) pair in parsed_data is one observation.
    PARSED_DATA has 3 records so the fact table must have 3 rows.
    Missing rows mean lost data; extra rows mean a join went wrong.
    """
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    assert len(df) == 3


def test_fact_contains_required_columns(dim_date, dim_location):
    """The star schema fact table must expose these exact columns for BI tools to query."""
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    assert set(df.columns) == {
        "observation_id", "date_id", "date", "location_id", "datatype", "value", "is_extreme"
    }


def test_fact_observation_id_is_unique(dim_date, dim_location):
    """observation_id is the primary key. Duplicates would corrupt upsert logic."""
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    assert df["observation_id"].nunique() == len(df)


def test_fact_joins_date_id_correctly(dim_date, dim_location):
    """
    Every row must have a non-null date_id after the merge. A null means
    the date in parsed_data didn't match any row in dim_date — the join
    silently dropped the foreign key and the row becomes unqueryable by date.
    """
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    assert df["date_id"].notna().all()


def test_fact_attaches_location_id(dim_date, dim_location):
    """
    Every row must carry the location_id from dim_location. A missing or
    wrong location_id means observations can't be filtered by geography.
    """
    expected_location_id = dim_location["location_id"].iloc[0]
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    assert (df["location_id"] == expected_location_id).all()


def test_fact_flags_extreme_value_as_true(dim_date, dim_location):
    """
    The 40.0 TMAX on 2024-01-20 exceeds the threshold of 35.0.
    is_extreme must be True for that row. A False here means a real
    extreme event was silently swallowed.
    """
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    extreme_rows = df[(df["datatype"] == "TMAX") & (df["value"] == 40.0)]

    assert len(extreme_rows) == 1
    assert extreme_rows.iloc[0]["is_extreme"] == True


def test_fact_does_not_flag_normal_value_as_extreme(dim_date, dim_location):
    """
    The 32.0 TMAX on 2024-01-15 is below the 35.0 threshold.
    is_extreme must be False. False positives erode trust and trigger
    unnecessary alerts.
    """
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    normal_rows = df[(df["datatype"] == "TMAX") & (df["value"] == 32.0)]

    assert len(normal_rows) == 1
    assert normal_rows.iloc[0]["is_extreme"] == False


def test_fact_datatype_without_threshold_is_not_extreme(dim_date, dim_location):
    """
    PRCP has no threshold configured in SETTINGS. Any PRCP value, no matter
    how high, must not be flagged as extreme. Flagging it would be a false
    positive based on missing config, not actual data.
    """
    df = build_fact_weather_observations(PARSED_DATA, dim_date, dim_location, SETTINGS)
    prcp_rows = df[df["datatype"] == "PRCP"]

    assert (prcp_rows["is_extreme"] == False).all()
