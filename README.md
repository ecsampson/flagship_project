# NOAA Weather ETL Pipeline

A data engineering portfolio project that fetches decade-scale weather data from the NOAA API, detects extreme weather events against NWS-defined thresholds, engineers time-series features, and persists outputs to both local storage and AWS S3 in CSV and Parquet formats. Built as a hands-on demonstration of ETL patterns, star schema modeling, and AI-assisted engineering.

---

## Architecture

```
NOAA NCEI API
     │
     ▼
fetch_all_noaa_data()       — paginated HTTP with rate limiting
     │
     ▼
parse_noaa_data()           — normalize JSON, divide tenths-of-units values by 10
     │
     ├──► build_dim_date()           ──► dim_date.csv / .parquet
     ├──► build_dim_location()       ──► dim_location.csv / .parquet
     ├──► build_fact_weather_obs()   ──► fact_weather_observations.csv / .parquet
     │
     ├──► group_noaa_data()
     │         │
     │         ├──► detect_extreme_weather()   ──► extreme_weather.csv / .parquet
     │         └──► generate_weather_signals() ──► weather_signals.csv / .parquet
     │
     └──────────────────────────────────────── weather_data.csv / .parquet
                                                        │
                                                        ▼
                                                   AWS S3 Bucket
                                          (raw/ extreme/ features/ dimensions/ facts/)
```

**Key design decisions:**
- NOAA returns values in tenths of units (e.g. 320 = 32.0°C) — conversion happens once, at parse time, to avoid the risk of double-applying it downstream.
- All store functions write CSV in append mode and Parquet as a full overwrite. CSV accumulates a historical log; Parquet is always a clean, complete snapshot for analytical queries.
- Output files are prefixed with the station ID (`USC00213567_weather_data.csv`) so the pipeline can be extended to multiple stations without filename collisions.

---

## Output Schema

All files land in `data/` locally and are mirrored to S3 under the corresponding prefix.

### `fact_weather_observations` — `facts/`
The central fact table. One row per (date, datatype) observation.

| Column | Type | Description |
|---|---|---|
| observation_id | int | Primary key |
| date_id | int | Foreign key → dim_date |
| location_id | int | Foreign key → dim_location |
| datatype | str | Measurement type (TMAX, PRCP, etc.) |
| value | float | Measurement value in standard units |
| is_extreme | bool | True if value exceeds NWS threshold |

### `dim_date` — `dimensions/`
Date dimension for joining and calendar-based analysis.

| Column | Type | Description |
|---|---|---|
| date_id | int | Primary key |
| date | datetime | Calendar date |
| year / month / day | int | Calendar components |
| season | str | Winter / Spring / Summer / Fall |
| is_weekend | bool | True for Saturday and Sunday |

### `dim_location` — `dimensions/`
Single-row location dimension. Extend `_LOCATION_LOOKUP` in `models.py` to support additional stations.

| Column | Type | Description |
|---|---|---|
| location_id | int | Primary key |
| fips_code | str | FIPS state or station code |
| state_name / state_abbr | str | Human-readable state |
| region | str | US region (e.g. Midwest) |

### `weather_signals` — `features/`
Feature-engineered table for trend analysis and ML. One row per (date, datatype).

| Column | Type | Description |
|---|---|---|
| date | datetime | Observation date |
| datatype | str | Measurement type |
| value | float | Raw measurement value |
| rolling_avg_7d | float | 7-day rolling mean |
| rolling_avg_30d | float | 30-day rolling mean |
| deviation | float | Value minus 7-day average |
| is_extreme | bool | Threshold breach flag |
| severity_score | float | 0–1 score; 1.0 = worst observed extreme |
| consecutive_days | int | Days into current extreme streak |

### `weather_data` — `raw/`
Flat normalized records for every observation. Inputs to all downstream tables.

| Column | Type | Description |
|---|---|---|
| date | str | YYYY-MM-DD |
| type | str | Datatype (TMAX, TMIN, PRCP, etc.) |
| value | float | Measurement in standard units |

### `extreme_weather` — `extreme/`
Subset of `weather_data` — only records that breach a threshold. Enriched with threshold metadata for alerting and reporting.

| Column | Type | Description |
|---|---|---|
| date / type / value | — | Same as weather_data |
| threshold | float | The threshold value that was breached |
| condition | str | "above" or "below" |

---

## Extreme Weather Thresholds

Thresholds are aligned to NWS Twin Cities watch/warning/advisory criteria for station USC00213567 (Minnesota).
Source: [NWS Twin Cities — WWA Criteria](https://www.weather.gov/mpx/wwa_criteria)

| Datatype | Description | Threshold |
|---|---|---|
| TMAX | Maximum temperature | > 37.8°C (100°F) |
| TMIN | Minimum temperature | < −37.2°C (−35°F) |
| PRCP | Precipitation | > 25.0 mm (1.0 in) |
| SNOW | Snowfall | > 152.4 mm (6.0 in) |
| SNWD | Snow depth | > 152.4 mm (6.0 in) |
| AWND | Average wind speed | > 17.9 m/s (40 mph) |
| WSF2 | Fastest 2-min wind speed | > 25.9 m/s (58 mph) |

---

## Tech Stack

| Tool | Role |
|---|---|
| Python 3.14 | Core language |
| pandas | Data transformation and feature engineering |
| pyarrow | Parquet serialization |
| boto3 | AWS S3 uploads |
| PyYAML | Configuration loading |
| requests | NOAA API HTTP client |
| pytest | Unit and integration testing (142 tests) |
| AWS S3 | Cloud storage for all CSV and Parquet outputs |

---

## How to Run

**Prerequisites:** Python 3.9+, an active NOAA API token, and AWS credentials with S3 write access.

```bash
# 1. Clone and set up the environment
git clone <repo-url>
cd flagship_project
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# 2. Configure
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml — set your station ID, date range, and S3 bucket

# 3. Set required environment variables
export NOAA_API_KEY="your_noaa_api_key"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# 4. Run the pipeline
cd src && python main.py

# 5. Run the test suite
cd .. && python -m pytest tests/
```

---

## AI Usage Notes

This project was built with Claude (Anthropic) as a pair programming tool throughout development. The goal was to use AI the way a professional would — to accelerate execution, not to replace engineering judgment.

**What AI assisted with:**
- Boilerplate and structure for functions I had already designed (fetch pagination, CSV append logic)
- Writing docstrings and test cases once the function behavior was defined
- Debugging type errors and pandas API changes (the pandas 3.x `groupby().apply()` promotion behavior in `features.py`)
- Suggesting the columnar storage trade-off between CSV append and Parquet overwrite

**What I decided independently:**
- The overall pipeline architecture and stage boundaries
- The choice to use a star schema (fact + dims) rather than a flat output
- Threshold values — researched and sourced from NWS Twin Cities directly
- Which datatypes to pull and what constitutes a useful feature for downstream analysis
- When the AI's suggestion didn't fit (e.g. it suggested deduplication logic that would have changed the pipeline's append-by-design behavior)

The test suite (142 tests across unit, integration, and data quality layers) was written collaboratively — I defined what needed to be tested and why; AI helped translate that into pytest code.

The intent is to demonstrate that AI tools raise the ceiling on what one engineer can ship, while engineering judgment still determines what gets built and why.
