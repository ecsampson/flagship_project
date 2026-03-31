import time
import requests


def fetch_all_noaa_data(api_key, base_url, params):
    """Fetch all records from the NOAA API using offset-based pagination.

    Repeatedly calls the endpoint, advancing the offset by 1000 each time,
    until a page returns fewer than 1000 records (signalling the final page).
    Sleeps 0.5 seconds between requests to stay within NOAA rate limits.

    Returns a flat list of all raw record dicts across all pages.
    """
    all_records = []
    offset = 0
    page_size = 9999

    while True:
        paged_params = {**params, "offset": offset, "limit": page_size}
        response = fetch_noaa_data(api_key, base_url, paged_params)
        page = response.json()

        all_records.extend(page)
        print(f"Fetched {len(all_records)} records so far (offset={offset})...")

        if len(page) < page_size:
            break

        offset += page_size
        time.sleep(0.5)

    return all_records


def fetch_noaa_data(api_key, base_url, params):
    """Fetch data from the NOAA API and return the response object."""
    try:
        response = requests.get(
            base_url,
            headers={"token": api_key},
            params=params,
            timeout=30
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
    skip_keys = {"DATE", "STATION"}
    for row in data:
        date = row["DATE"][:10]
        for key, raw in row.items():
            if key in skip_keys:
                continue
            try:
                value = int(raw) / 10
            except (ValueError, TypeError):
                value = None
            parsed.append({"date": date, "type": key, "value": value})
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