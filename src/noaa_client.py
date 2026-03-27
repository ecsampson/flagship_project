import requests


def fetch_noaa_data(api_key, base_url, params):
    response = requests.get(
        base_url,
        headers={"token": api_key},
        params=params
    )
    return response


def parse_noaa_data(data):
    parsed = []
    for item in data.get("results", []):
        parsed.append({
            "date": item["date"][:10],
            "type": item["datatype"],
            "value": item["value"] / 10
        })
    return parsed


def group_noaa_data(data):
    grouped = {}
    for item in data:
        grouped.setdefault(item["type"], []).append(item)
    return grouped


def detect_extreme_weather(grouped_data, thresholds):
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