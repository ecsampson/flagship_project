import os
import yaml
from pathlib import Path
from noaa_client import fetch_noaa_data, group_noaa_data, parse_noaa_data, detect_extreme_weather
from features import generate_weather_signals
from storage import store_noaa_data, store_extreme_weather, store_weather_signals, upload_to_s3

# Paths resolved relative to this file so the script works from any working directory
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
DATA_PATH = Path(__file__).parent.parent / "data" / "noaa_weather_data.csv"
EXTREME_PATH = Path(__file__).parent.parent / "data" / "noaa_extreme_weather.csv"
SIGNALS_PATH = Path(__file__).parent.parent / "data" / "noaa_weather_signals.csv"


def main():
    """Run the NOAA ETL pipeline: fetch, parse, detect extreme weather, and store results."""
    with open(CONFIG_PATH, "r") as f:
        settings = yaml.safe_load(f)

    aws_bucket = settings["aws"]["bucket"]

    noaa = settings["noaa"]
    # Environment variable takes precedence so the config-file key is never required in production
    api_key = os.environ.get("NOAA_API_KEY") or noaa["api_key"]
    params = {**noaa["query"]}

    response = fetch_noaa_data(
        api_key=api_key,
        base_url=noaa["base_url"],
        params=params
    )

    if response.status_code == 200:
        parsed_data = parse_noaa_data(response.json())
        grouped_data = group_noaa_data(parsed_data)
        extreme_events = detect_extreme_weather(grouped_data, settings["extreme_weather"]["thresholds"])
        weather_signals = generate_weather_signals(grouped_data, settings["extreme_weather"]["thresholds"])
        store_noaa_data(parsed_data, DATA_PATH)
        store_extreme_weather(extreme_events, EXTREME_PATH)
        store_weather_signals(weather_signals, SIGNALS_PATH)
        upload_to_s3(DATA_PATH, "raw", aws_bucket)
        upload_to_s3(EXTREME_PATH, "extreme", aws_bucket)
        upload_to_s3(SIGNALS_PATH, "features", aws_bucket)
    else:
        print(f"API request failed: {response.status_code} - {response.text}")
        return


if __name__ == "__main__":
    main()
