import os
import yaml
import pandas as pd
from pathlib import Path
from noaa_client import fetch_all_noaa_data, group_noaa_data, parse_noaa_data, detect_extreme_weather
from features import generate_weather_signals
from storage import (
    store_noaa_data, store_extreme_weather, store_weather_signals,
    upload_to_s3, upload_parquet_append_to_s3,
    get_last_run_date, save_last_run_date,
)
from models import build_dim_date, build_dim_location, build_fact_weather_observations


_IN_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

CONFIG_PATH = Path(__file__).parent / "settings.yaml" if _IN_LAMBDA else Path(__file__).parent.parent / "config" / "settings.yaml"
DATA_DIR = Path("/tmp") if _IN_LAMBDA else Path(__file__).parent.parent / "data"

def main(start_date=None, end_date=None):
    """Run the NOAA ETL pipeline: fetch, parse, detect extreme weather, and store results.

    Args:
        start_date: ISO date string override for the query startDate.
                    When None, the value from settings.yaml is used.
        end_date:   ISO date string override for the query endDate.
                    When None, the value from settings.yaml is used.
    """
    with open(CONFIG_PATH, "r") as f:
        settings = yaml.safe_load(f)

    aws_bucket = settings["aws"]["bucket"]

    noaa = settings["noaa"]
    api_key = os.environ.get("NOAA_API_KEY") or noaa["api_key"]
    params = {**noaa["query"]}

    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    station_id = noaa["query"]["stations"]
    if isinstance(station_id, list):
        station_id = station_id[0]

    # Paths include the station ID so outputs from different stations don't collide
    DATA_PATH = DATA_DIR / f"{station_id}_weather_data.csv"
    EXTREME_PATH = DATA_DIR / f"{station_id}_extreme_weather.csv"
    SIGNALS_PATH = DATA_DIR / f"{station_id}_weather_signals.csv"
    DIM_DATE_PATH = DATA_DIR / f"{station_id}_dim_date.csv"
    DIM_LOCATION_PATH = DATA_DIR / f"{station_id}_dim_location.csv"
    FACT_WEATHER_OBSERVATIONS_PATH = DATA_DIR / f"{station_id}_fact_weather_observations.csv"

    raw_records = fetch_all_noaa_data(
        api_key=api_key,
        base_url=noaa["base_url"],
        params=params
    )

    if not raw_records:
        print("No records returned from NOAA API.")
        return

    parsed_data = parse_noaa_data(raw_records)
    dim_date = build_dim_date(parsed_data)
    dim_location = build_dim_location(settings)
    fact_weather_observations = build_fact_weather_observations(parsed_data, dim_date, dim_location, settings)
    store_noaa_data(dim_date, DIM_DATE_PATH)
    store_noaa_data(dim_location, DIM_LOCATION_PATH)
    store_noaa_data(fact_weather_observations, FACT_WEATHER_OBSERVATIONS_PATH)
    grouped_data = group_noaa_data(parsed_data)
    extreme_events = detect_extreme_weather(grouped_data, settings["extreme_weather"]["thresholds"])
    weather_signals = generate_weather_signals(grouped_data, settings["extreme_weather"]["thresholds"])
    store_noaa_data(parsed_data, DATA_PATH)
    store_extreme_weather(extreme_events, EXTREME_PATH)
    store_weather_signals(weather_signals, SIGNALS_PATH)

    # CSV uploads (overwrite each run — CSVs are ephemeral scratch files in /tmp)
    for path, data_type in [
        (DIM_DATE_PATH, "dimensions"),
        (DIM_LOCATION_PATH, "dimensions"),
        (FACT_WEATHER_OBSERVATIONS_PATH, "facts"),
        (DATA_PATH, "raw"),
        (EXTREME_PATH, "extreme"),
        (SIGNALS_PATH, "features"),
    ]:
        upload_to_s3(path, data_type, aws_bucket)

    # Parquet uploads: append new rows to existing S3 objects and deduplicate.
    # dedup_cols is the business/natural key for each dataset — the columns that
    # together uniquely identify one logical record regardless of which run produced it.
    parquet_uploads = [
        (dim_date,                         "dimensions", DIM_DATE_PATH,                  ["date"]),
        (dim_location,                     "dimensions", DIM_LOCATION_PATH,               ["location_id"]),
        (fact_weather_observations,        "facts",      FACT_WEATHER_OBSERVATIONS_PATH,  ["date", "datatype"]),
        (pd.DataFrame(parsed_data),        "raw",        DATA_PATH,                       ["date", "type"]),
        (pd.DataFrame(extreme_events),     "extreme",    EXTREME_PATH,                    ["date", "type", "condition"]),
        (weather_signals,                  "features",   SIGNALS_PATH,                    ["date", "datatype"]),
    ]
    for df, data_type, path, dedup_cols in parquet_uploads:
        s3_key = f"{data_type}/parquet/{path.with_suffix('.parquet').name}"
        upload_parquet_append_to_s3(df, aws_bucket, s3_key, dedup_cols)


def lambda_handler(event, context):
    from datetime import date, timedelta

    with open(CONFIG_PATH, "r") as f:
        settings = yaml.safe_load(f)
    bucket = settings["aws"]["bucket"]

    start_date = get_last_run_date(bucket)
    end_date = (date.today() - timedelta(days=1)).isoformat()

    main(start_date=start_date, end_date=end_date)

    save_last_run_date(end_date, bucket)
    return {
        "statusCode": 200,
        "body": f"Pipeline completed: {start_date} to {end_date}",
    }


if __name__ == "__main__":
    main()
