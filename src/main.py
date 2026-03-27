import os
import yaml
from pathlib import Path
from noaa_client import fetch_noaa_data, group_noaa_data, parse_noaa_data, detect_extreme_weather
from storage import store_noaa_data, store_extreme_weather

CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
DATA_PATH = Path(__file__).parent.parent / "data" / "noaa_weather_data.csv"
EXTREME_PATH = Path(__file__).parent.parent / "data" / "noaa_extreme_weather.csv"


def main():
    with open(CONFIG_PATH, "r") as f:
        settings = yaml.safe_load(f)

    noaa = settings["noaa"]
    api_key = os.environ.get("NOAA_API_KEY") or noaa["api_key"]
    params = {
        "datasetid": noaa["default_dataset"],
        "locationid": noaa["default_location"],
        **noaa["query"]
    }

    response = fetch_noaa_data(
        api_key=api_key,
        base_url=noaa["base_url"],
        params=params
    )

    if response.status_code == 200:
        parsed_data = parse_noaa_data(response.json())
        grouped_data = group_noaa_data(parsed_data)
        extreme_events = detect_extreme_weather(grouped_data, settings["extreme_weather"]["thresholds"])
        store_noaa_data(parsed_data, DATA_PATH)
        store_extreme_weather(extreme_events, EXTREME_PATH)
    else:
        print(f"API request failed: {response.status_code} - {response.text}")
        return


if __name__ == "__main__":
    main()
