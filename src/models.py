import pandas as pd

# Maps location identifier to metadata. Keys are either FIPS codes (e.g. "27")
# or full station IDs (e.g. "USC00213567"). Extend to support more locations.
_LOCATION_LOOKUP = {
    "27": {"state_name": "Minnesota", "state_abbr": "MN", "region": "Midwest"},
    "USC00213567": {"state_name": "Minnesota", "state_abbr": "MN", "region": "Midwest"},
}

_SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}


def build_dim_date(parsed_data):
    """Build a date dimension table from parsed NOAA records.

    Returns a DataFrame with one row per unique date, enriched with
    calendar attributes (year, month, day, season, is_weekend).
    """
    dates = pd.to_datetime(
        pd.Series([r["date"] for r in parsed_data]).unique()
    ).sort_values()

    df = pd.DataFrame({"date": dates})
    df["date_id"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["season"] = df["month"].map(_SEASON_MAP)
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    return df[["date_id", "date", "year", "month", "day", "season", "is_weekend"]]


def build_dim_location(settings):
    """Build a location dimension table from NOAA config settings.

    Extracts the FIPS code from settings['noaa']['default_location']
    (e.g. "FIPS:27" → "27") and looks up state metadata.

    Returns a single-row DataFrame. Add entries to _LOCATION_LOOKUP
    to support additional stations without changing this function.
    """
    raw_location = settings["noaa"]["default_location"]
    # Support both "FIPS:27" style and bare station IDs like "USC00213567"
    location_key = raw_location.split(":")[1] if ":" in raw_location else raw_location
    meta = _LOCATION_LOOKUP.get(location_key, {
        "state_name": "Unknown",
        "state_abbr": "??",
        "region": "Unknown",
    })

    df = pd.DataFrame([{
        "location_id": 0,
        "fips_code": location_key,
        "state_name": meta["state_name"],
        "state_abbr": meta["state_abbr"],
        "region": meta["region"],
    }])

    return df[["location_id", "fips_code", "state_name", "state_abbr", "region"]]


def build_fact_weather_observations(parsed_data, dim_date, dim_location, settings):
    """Build the fact table of weather observations.

    Joins each observation to dim_date on date and attaches the single
    location_id from dim_location. Flags extreme events using the
    threshold rules from settings.

    Returns a DataFrame with one row per (date, datatype) observation.
    """
    thresholds = settings["extreme_weather"]["thresholds"]
    location_id = dim_location["location_id"].iloc[0]

    df = pd.DataFrame(parsed_data).rename(columns={"type": "datatype"})
    df["date"] = pd.to_datetime(df["date"])

    df = df.merge(dim_date[["date_id", "date"]], on="date", how="left")
    df["location_id"] = location_id

    df["is_extreme"] = df.apply(
        lambda row: _is_extreme(row["datatype"], row["value"], thresholds), axis=1
    )

    df["observation_id"] = range(len(df))

    return df[["observation_id", "date_id", "date", "location_id", "datatype", "value", "is_extreme"]]


def _is_extreme(datatype, value, thresholds):
    if value is None or datatype not in thresholds:
        return False
    criteria = thresholds[datatype]
    if "above" in criteria and value > criteria["above"]:
        return True
    if "below" in criteria and value < criteria["below"]:
        return True
    return False
