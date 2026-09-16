import pandas as pd
from anomaly_data import get_tmax_data

def build_feature_table(df):
    """Reindex NOAA data to a complete daily calendar and add model features.

Fills in missing calendar dates so downstream lag/rolling calculations
are based on true day-to-day adjacency, not just row position. Missing
dates get NaN for tmax.

Args:
    df: DataFrame with columns date and tmax. May have missing dates
        (gaps in the underlying station record).

Returns:
    DataFrame with one row per calendar day between df's min and max
    date, columns:
        date         - calendar date
        tmax         - observed value; NaN on gap-days
        day_of_year  - 1-366, from date
        year         - from date
        tmax_lag1    - tmax from the previous calendar day; NaN if
                       that day was itself a gap, or is the first
                       row in the table
"""
    df = df.set_index('date')
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_range)
    df.index.name = 'date'

    df['day_of_year'] = df.index.dayofyear
    df['year'] = df.index.year
    df['tmax_lag1'] = df['tmax'].shift(1)

    df = df.reset_index()
    return df