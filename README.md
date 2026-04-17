# NOAA Weather ETL Pipeline

A data engineering portfolio project that fetches 87+ years of weather data from the NOAA API, detects extreme weather events against NWS-defined thresholds, engineers time-series features, persists outputs to AWS S3 in Parquet format, and surfaces results in a Power BI dashboard. The pipeline runs automatically every day via AWS Lambda and EventBridge with no manual intervention. Built as a hands-on demonstration of ETL patterns, star schema modeling, incremental loading, and AI-assisted engineering.

---

## Architecture

```
AWS EventBridge (daily schedule)
     │
     ▼
AWS Lambda — lambda_handler()
     │  reads last_run_date from S3 metadata
     │  computes 10-year backfill chunk or yesterday (incremental)
     ▼
NOAA NCEI API
     │
     ▼
fetch_all_noaa_data()       — paginated HTTP with rate limiting
     │
     ▼
parse_noaa_data()           — normalize JSON, divide tenths-of-units values by 10
     │
     ├──► build_dim_date()           ──► dim_date.parquet
     ├──► build_dim_location()       ──► dim_location.parquet
     ├──► build_fact_weather_obs()   ──► fact_weather_observations.parquet
     │
     ├──► group_noaa_data()
     │         │
     │         ├──► detect_extreme_weather()   ──► extreme_weather.parquet
     │         └──► generate_weather_signals() ──► weather_signals.parquet
     │
     └──────────────────────────────────────── weather_data.parquet
                                                        │
                                                        ▼
                                                   AWS S3 Bucket
                                          (raw/ extreme/ features/ dimensions/ facts/)
                                                        │
                                                        ▼
                                               AWS Athena + Glue
                                          (external tables over S3 Parquet)
```

**Key design decisions:**
- NOAA returns values in tenths of units (e.g. 320 = 32.0°C) — conversion happens once, at parse time, to avoid the risk of double-applying it downstream.
- Parquet files in S3 use an **append + deduplicate** pattern: each run reads the existing S3 object, concatenates new rows, drops duplicates on the natural key (`date + datatype`), and writes back. This means reruns and Lambda retries are safe — no duplicate rows accumulate.
- `date_id` in `dim_date` is derived as an integer in `YYYYMMDD` format (e.g. `20240315`) rather than a sequential row index. This makes the surrogate key stable across incremental runs — the same date always maps to the same `date_id`, so foreign keys in the fact table remain valid after any backfill or retry.
- CSV files are written locally to `/tmp` each run for debugging and are also uploaded to S3, but they are not the source of truth for analytics — Parquet is.
- Output files are prefixed with the station ID (`USW00014922_weather_data.parquet`) so the pipeline can be extended to multiple stations without filename collisions.

---

## Data Coverage

| Property | Value |
|---|---|
| Station | USW00014922 — Minneapolis St. Paul International Airport |
| Date range | 1938-01-01 to present (updated daily) |
| Records | 1,034,944+ observations |
| Datatypes | TMAX, TMIN, PRCP, SNOW, SNWD, AWND, WSF2 |
| Location | Minnesota, Midwest US |

Station USW00014922 is one of the longest-running NOAA GHCND stations in the Upper Midwest, with continuous daily observations from 1938. The full 87-year backfill was loaded incrementally in 10-year chunks to stay within Lambda's execution limits.

---

## Automation

The pipeline is fully automated and requires no manual intervention after deployment.

**How it works:**

1. **EventBridge** triggers the Lambda function on a daily schedule.
2. `lambda_handler` reads `metadata/last_run_date.json` from S3 to determine where the last successful run ended.
3. It computes the next window: `start = last_run_date`, `end = min(start + 10 years, yesterday)`.
4. During the initial backfill, each invocation processes one decade of history. Once caught up, the window naturally collapses to a single day and the pipeline enters normal incremental mode.
5. After a successful run, `last_run_date` is updated in S3 — advancing the cursor only on success. A failed run retries the same window on the next invocation.

**Backfill schedule (first ~9 invocations):**

| Run | Window |
|-----|--------|
| 1 | 1938-01-01 → 1947-12-31 |
| 2 | 1947-12-31 → 1957-12-30 |
| … | … |
| 9 | ~2015 → 2025 |
| 10+ | Yesterday only (incremental) |

---

## AWS Infrastructure

| Service | Role |
|---------|------|
| **S3** | Primary data store for all Parquet and CSV outputs, plus pipeline metadata (`last_run_date.json`) |
| **Lambda** | Runs the full ETL pipeline on a schedule; Python 3.12 runtime |
| **EventBridge** | Triggers Lambda on a daily cron schedule |
| **Athena** | Ad-hoc SQL queries over S3 Parquet data |
| **Glue Data Catalog** | Stores table definitions used by Athena |
| **IAM** | Per-function execution role with least-privilege S3 and logging permissions |

The Lambda deployment package is kept under 3 MB by omitting pandas, pyarrow, and numpy from the zip — these are provided by the **AWS SDK for pandas** managed layer:

```
arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python312:22
```

---

## Athena

All Parquet outputs are queryable via AWS Athena through manually defined external tables in the Glue Data Catalog. Each table points at an S3 prefix and uses the Parquet SerDe.

**Tables:**

| Table | S3 Prefix | Description |
|-------|-----------|-------------|
| `fact_weather_observations` | `facts/parquet/` | One row per (date, datatype) observation |
| `dim_date` | `dimensions/parquet/` | Date dimension with calendar attributes |
| `dim_location` | `dimensions/parquet/` | Station/location metadata |
| `feat_weather_signals` | `features/parquet/` | Feature-engineered rolling stats and extremes |

Example query — hottest days on record:

```sql
SELECT f.date, f.value AS tmax_celsius
FROM fact_weather_observations f
WHERE f.datatype = 'TMAX'
  AND f.is_extreme = true
ORDER BY f.value DESC
LIMIT 20;
```

Example query — extreme winters by decade:

```sql
SELECT
    FLOOR(d.year / 10) * 10 AS decade,
    COUNT(*) AS extreme_cold_days
FROM fact_weather_observations f
JOIN dim_date d ON f.date_id = d.date_id
WHERE f.datatype = 'TMIN'
  AND f.is_extreme = true
GROUP BY 1
ORDER BY 1;
```

---

## Power BI Dashboard

The dashboard (`visuals/data_analysis.pbix`) connects directly to the Parquet outputs and provides four visualizations for exploring 87+ years of Minneapolis weather history.

| Visual | Type | Description |
|--------|------|-------------|
| Extreme Event Count by Decade | Bar chart | Total extreme weather events grouped by decade — shows long-term trends in frequency |
| Extreme Event Count by Season | Pie chart | Breakdown of extreme events by season (Winter / Spring / Summer / Fall) |
| Extreme Event Count by Datatype | Pie chart | Breakdown of extreme events by measurement type (TMAX, TMIN, PRCP, SNOW, etc.) |
| Average TMAX and TMIN by Year | Line chart | Annual average maximum and minimum temperatures — visualizes the long-term temperature record |

---

## Output Schema

All files land in `data/` locally and are mirrored to S3 under the corresponding prefix.

### `fact_weather_observations` — `facts/`
The central fact table. One row per (date, datatype) observation.

| Column | Type | Description |
|--------|------|--------------|
| observation_id | int | Surrogate key |
| date_id | int | Foreign key → dim_date |
| date | datetime | Observation date (also stored here for dedup and direct querying) |
| location_id | int | Foreign key → dim_location |
| datatype | str | Measurement type (TMAX, PRCP, etc.) |
| value | float | Measurement value in standard units |
| is_extreme | bool | True if value exceeds NWS threshold |

### `dim_date` — `dimensions/`
Date dimension for joining and calendar-based analysis.

| Column | Type | Description |
|--------|------|-------------|
| date_id | int | Primary key — derived as YYYYMMDD integer for stability across incremental runs |
| date | datetime | Calendar date |
| year / month / day | int | Calendar components |
| season | str | Winter / Spring / Summer / Fall |
| is_weekend | bool | True for Saturday and Sunday |

### `dim_location` — `dimensions/`
Single-row location dimension. Extend `_LOCATION_LOOKUP` in `models.py` to support additional stations.

| Column | Type | Description |
|--------|------|-------------|
| location_id | int | Primary key |
| fips_code | str | FIPS state or station code |
| state_name / state_abbr | str | Human-readable state |
| region | str | US region (e.g. Midwest) |

### `weather_signals` — `features/`
Feature-engineered table for trend analysis and ML. One row per (date, datatype).

| Column | Type | Description |
|--------|------|-------------|
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
|--------|------|-------------|
| date | str | YYYY-MM-DD |
| type | str | Datatype (TMAX, TMIN, PRCP, etc.) |
| value | float | Measurement in standard units |

### `extreme_weather` — `extreme/`
Subset of `weather_data` — only records that breach a threshold. Enriched with threshold metadata for alerting and reporting.

| Column | Type | Description |
|--------|------|-------------|
| date / type / value | — | Same as weather_data |
| threshold | float | The threshold value that was breached |
| condition | str | "above" or "below" |

---

## Extreme Weather Thresholds

Thresholds are aligned to NWS Twin Cities watch/warning/advisory criteria for station USW00014922 (Minneapolis St. Paul International Airport).

Source: [NWS Twin Cities — WWA Criteria](https://www.weather.gov/mpx/wwa_criteria)

| Datatype | Description | Threshold |
|----------|-------------|-----------|
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
|------|------|
| Python 3.12 | Core language (Lambda runtime) |
| pandas | Data transformation and feature engineering |
| pyarrow | Parquet serialization |
| boto3 | AWS SDK — S3, Lambda, and metadata operations |
| PyYAML | Configuration loading |
| requests | NOAA API HTTP client |
| pytest | Unit and integration testing (142 tests) |
| Power BI | Dashboard and visualization layer |
| AWS SDK for pandas layer | Provides pandas + pyarrow + numpy in Lambda |

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

# 4. Run the pipeline locally
cd src && python main.py

# 5. Run the test suite
cd .. && python -m pytest tests/
```

---

## Project Timeline

| Phase | Status | Description |
|-------|--------|-------------|
| Core ETL pipeline | Complete | Fetch, parse, group, detect, store |
| Star schema modeling | Complete | fact_weather_observations + 2 dims |
| Feature engineering | Complete | Rolling averages, severity scores, streaks |
| Dual CSV + Parquet output | Complete | Both formats written per run |
| AWS S3 integration | Complete | All outputs mirrored to S3 |
| AWS Lambda deployment | Complete | Packaged and deployed with managed layer |
| Backfill + incremental loading | Complete | 10-year chunks → daily incremental |
| Athena queryability | Complete | External tables over S3 Parquet |
| EventBridge scheduling | Complete | Daily automated runs |
| Stable surrogate keys | Complete | YYYYMMDD date_id — safe across incremental runs |
| Dashboard / visualization | Complete | Power BI dashboard with 4 visuals (bar, 2× pie, line) |
| Multi-station support | Planned | — |

---

## AI Usage Notes

This project was built with Claude (Anthropic) as a pair programming tool throughout development. The goal was to use AI the way a professional would — to accelerate execution, not to replace engineering judgment.

**What AI assisted with:**
- Boilerplate and structure for functions I had already designed (fetch pagination, CSV append logic)
- Writing docstrings and test cases once the function behavior was defined
- Debugging type errors and pandas API changes (the pandas 3.x `groupby().apply()` promotion behavior in `features.py`)
- Lambda packaging — diagnosing the pyarrow `.so` stripping issue and switching to the managed AWS SDK for pandas layer
- Implementing the `upload_parquet_append_to_s3` append + deduplicate pattern
- Drafting the backfill chunking logic in `lambda_handler`
- Implementing the `date_id` fix in `models.py` — replacing `range(len(df))` with the YYYYMMDD integer derivation once I identified the collision root cause
- Adding `sync_source_files()` to `build_lambda.py` and scaffolding `update_lambda.py` for the S3 upload + Lambda deploy workflow

**What I decided independently:**
- The overall pipeline architecture and stage boundaries
- The choice to use a star schema (fact + dims) rather than a flat output
- Threshold values — researched and sourced from NWS Twin Cities directly
- Which datatypes to pull and what constitutes a useful feature for downstream analysis
- The backfill + incremental design pattern and the 10-year chunk size
- Station selection (USW00014922) and confirming the 1938 data start date empirically via API
- Diagnosing the `date_id` collision bug — recognizing that sequential row indexing breaks under incremental loads and that a date-derived key was the correct fix
- The decision to use Power BI for the dashboard and all four visualization choices: which chart types to use, which dimensions to slice by (decade, season, datatype), and which metrics to trend over time (TMAX/TMIN by year)

The test suite (142 tests across unit, integration, and data quality layers) was written collaboratively — I defined what needed to be tested and why; AI helped translate that into pytest code.

The intent is to demonstrate that AI tools raise the ceiling on what one engineer can ship, while engineering judgment still determines what gets built and why.
