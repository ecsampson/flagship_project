import pandas as pd


def generate_weather_signals(grouped_data, thresholds):
    """Generate a feature table from grouped NOAA data.

    Args:
        grouped_data: dict of {datatype: [{"date": str, "type": str, "value": float}]}
                      as returned by noaa_client.group_noaa_data()
        thresholds:   dict of {datatype: {"above": float} or {"below": float}}
                      as loaded from config["extreme_weather"]["thresholds"]

    Returns:
        DataFrame (feat_weather_signals) with columns:
            date, datatype, value, rolling_avg_7d, rolling_avg_30d,
            deviation, is_extreme, severity_score, consecutive_days
    """
    # --- Flatten grouped dict into a single DataFrame -----------------------
    records = []
    for datatype, rows in grouped_data.items():
        for row in rows:
            records.append({"date": row["date"], "datatype": row["type"], "value": row["value"]})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"])
    # Sort within each datatype by date so rolling windows are chronologically correct
    df = df.sort_values(["datatype", "date"]).reset_index(drop=True)

    # --- Rolling averages and deviation -------------------------------------
    # transform() keeps the original DataFrame shape and index intact, avoiding
    # the pandas 3.x issue where groupby().apply() promotes the group key out
    # of the columns. Windows are row-based, correct for gap-free daily data.
    grouped_value = df.groupby("datatype")["value"]
    df["rolling_avg_7d"] = grouped_value.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["rolling_avg_30d"] = grouped_value.transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["deviation"] = df["value"] - df["rolling_avg_7d"]

    # --- is_extreme ---------------------------------------------------------
    df["is_extreme"] = False
    for datatype, criteria in thresholds.items():
        dt_mask = df["datatype"] == datatype
        if "above" in criteria:
            df.loc[dt_mask & (df["value"] > criteria["above"]), "is_extreme"] = True
        if "below" in criteria:
            df.loc[dt_mask & (df["value"] < criteria["below"]), "is_extreme"] = True

    # --- severity_score -----------------------------------------------------
    # Normalisation reference: the observed min/max range for each datatype.
    # For "above" thresholds: score = (value - threshold) / (observed_max - threshold)
    # For "below" thresholds: score = (threshold - value) / (threshold - observed_min)
    # This maps the worst observed extreme to 1.0 and the threshold itself to 0.0.
    # Values outside that range are capped at 1.0.
    value_stats = df.groupby("datatype")["value"].agg(["min", "max"])

    df["severity_score"] = 0.0
    for datatype, criteria in thresholds.items():
        if datatype not in value_stats.index:
            continue
        stats = value_stats.loc[datatype]
        extreme_mask = (df["datatype"] == datatype) & df["is_extreme"]
        if "above" in criteria:
            threshold = criteria["above"]
            denom = stats["max"] - threshold
            if denom > 0:
                df.loc[extreme_mask, "severity_score"] = (
                    (df.loc[extreme_mask, "value"] - threshold) / denom
                ).clip(upper=1.0)
            else:
                df.loc[extreme_mask, "severity_score"] = 1.0
        elif "below" in criteria:
            threshold = criteria["below"]
            denom = threshold - stats["min"]
            if denom > 0:
                df.loc[extreme_mask, "severity_score"] = (
                    (threshold - df.loc[extreme_mask, "value"]) / denom
                ).clip(upper=1.0)
            else:
                df.loc[extreme_mask, "severity_score"] = 1.0

    # --- consecutive_days ---------------------------------------------------
    # (~is_extreme).cumsum() increments each time a streak breaks, giving every
    # run of consecutive extreme days a unique group ID. cumsum() within each
    # (datatype, run_id) group then counts days into the streak. Multiplying by
    # is_extreme zeroes out non-extreme rows.
    run_id = (~df["is_extreme"]).groupby(df["datatype"]).cumsum()
    df["consecutive_days"] = (
        df.groupby(["datatype", run_id])["is_extreme"]
        .cumsum()
        .mul(df["is_extreme"])
        .astype(int)
    )

    feat_weather_signals = df[[
        "date", "datatype", "value",
        "rolling_avg_7d", "rolling_avg_30d",
        "deviation", "is_extreme", "severity_score", "consecutive_days",
    ]].reset_index(drop=True)

    return feat_weather_signals
