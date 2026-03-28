# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A data engineering portfolio project that fetches weather data from the NOAA API, detects extreme weather events, and stores results as CSV files. Built to demonstrate ETL patterns and AI-assisted development.

## Commands

```bash
# Setup
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Run the ETL pipeline
cd src && python main.py

# Override API key via environment variable
NOAA_API_KEY="your_key" python main.py

# Run tests
python -m pytest tests/

# Run a single test
python -m pytest tests/test_main.py::test_function_name
```

No linting configuration exists yet (no flake8/pylint/black config).

## Architecture

The pipeline flows through four stages, each handled by a dedicated function:

1. **Fetch** (`noaa_client.fetch_noaa_data`) — HTTP call to NOAA API with timeout/error handling; raises `RuntimeError` on failure
2. **Parse** (`noaa_client.parse_noaa_data`) — Normalizes raw JSON into `{date, type, value}` dicts; divides values by 10 (NOAA returns tenths of units)
3. **Group** (`noaa_client.group_noaa_data`) — Groups parsed records by data type (TMAX, TMIN, PRCP, SNOW, SNWD)
4. **Detect** (`noaa_client.detect_extreme_weather`) — Filters records exceeding thresholds from config; enriches events with threshold/condition metadata

`main.py` wires these steps together and delegates persistence to `storage.py`, which appends records to two CSV files:
- `data/noaa_weather_data.csv` — all weather records
- `data/noaa_extreme_weather.csv` — only extreme events

## Configuration

`config/settings.yaml` controls all API parameters (station, date range, data types) and extreme weather thresholds. An example template is at `config/settings.yaml.example`.

The NOAA API key is loaded from the `NOAA_API_KEY` environment variable first, falling back to the config file. **Always use the environment variable in practice** — the key in the config file should not be committed.

## Key Conventions

- Functions in `noaa_client.py` are stateless and designed to be independently testable
- CSV files use append-only mode — re-running the pipeline will duplicate records; there is no deduplication yet
- Tests live in `tests/` and should mirror the `src/` structure; `test_main.py` currently exists as an empty placeholder
