import os
import yaml
from pathlib import Path
from noaa_client import fetch_all_noaa_data, group_noaa_data, parse_noaa_data, detect_extreme_weather
from features import generate_weather_signals
from storage import store_noaa_data, store_extreme_weather, store_weather_signals, upload_to_s3
from models import build_dim_date, build_dim_location, build_fact_weather_observations


_IN_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

CONFIG_PATH = Path(__file__).parent / "settings.yaml" if _IN_LAMBDA else Path(__file__).parent.parent / "config" / "settings.yaml"
DATA_DIR = Path("/tmp") if _IN_LAMBDA else Path(__file__).parent.parent / "data"

def main():
    """Run the NOAA ETL pipeline: fetch, parse, detect extreme weather, and store results."""
    with open(CONFIG_PATH, "r") as f:
        settings = yaml.safe_load(f)

    aws_bucket = settings["aws"]["bucket"]

    noaa = settings["noaa"]
    # Environment variable takes precedence so the config-file key is never required in production
    api_key = os.environ.get("NOAA_API_KEY") or noaa["api_key"]
    params = {**noaa["query"]}

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

    if raw_records:
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
        for path, data_type in [
            (DIM_DATE_PATH, "dimensions"),
            (DIM_LOCATION_PATH, "dimensions"),
            (FACT_WEATHER_OBSERVATIONS_PATH, "facts"),
            (DATA_PATH, "raw"),
            (EXTREME_PATH, "extreme"),
            (SIGNALS_PATH, "features"),
        ]:
            upload_to_s3(path, data_type, aws_bucket)
            upload_to_s3(path.with_suffix('.parquet'), data_type, aws_bucket)
    else:
        print("No records returned from NOAA API.")
        return


def lambda_handler(event, context):
    main()
    return {"statusCode": 200, "body": "Pipeline completed successfully"}


if __name__ == "__main__":
    main()
