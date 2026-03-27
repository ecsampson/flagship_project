import requests


def fetch_noaa_data(api_key, base_url, params):
    """Fetch data from the NOAA API and return the response object."""
    try:
        response = requests.get(
            base_url,
            headers={"token": api_key},
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request to {base_url} timed out.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Failed to connect to {base_url}.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error {response.status_code}: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")


def parse_noaa_data(data):
    """Parse raw NOAA API response into a list of normalized date/type/value dicts."""
    parsed = []
    for item in data.get("results", []):
        parsed.append({
            "date": item["date"][:10],
            "type": item["datatype"],
            "value": item["value"] / 10 if item.get("value") is not None else None
        })
    return parsed


def group_noaa_data(data):
    """Group a list of parsed NOAA records by their data type."""
    grouped = {}
    for item in data:
        grouped.setdefault(item["type"], []).append(item)
    return grouped


def detect_extreme_weather(grouped_data, thresholds):
    """Return records from grouped data that exceed the defined above/below thresholds."""
    extreme_events = []
    for datatype, records in grouped_data.items():
        if datatype not in thresholds:
            continue
        criteria = thresholds[datatype]
        for record in records:
            value = record["value"]
            if "above" in criteria and value > criteria["above"]:
                extreme_events.append({**record, "threshold": criteria["above"], "condition": "above"})
            elif "below" in criteria and value < criteria["below"]:
                extreme_events.append({**record, "threshold": criteria["below"], "condition": "below"})
    return extreme_events