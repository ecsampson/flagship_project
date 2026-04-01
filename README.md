# Flagship Project

## Project Vision and Goals
This project aims to gather weather data to analyze patterns and anomalies in recent weather events, helping to better understand trends in extreme weather conditions. The project will utilize NOAA APIs for data collection, Python for data transformation and cleaning, cloud storage for data management, and Power BI for analytics.

## Primary Outcome
- Improve understanding of data engineering and architecture.
- Showcase AI-assisted coding while building a portfolio-worthy project.
- Develop practical skills in data processing, transformation, and visualization.

## Key Skills
- Data Engineering
- Python development
- Understanding of data fundamentals (ETL, batch processing, etc.)
- Utilizing AI as a tool for coding (not just copy-pasting)
- Self-learning and skill development through hands-on project work

## Project Scope

### Project Structure
- src/ # Source Code
- test/ # Unit Tests
- docs/ # Documentation
- config/ # Configuration
- assets/ # Images, media, etc
- README.md
- LICENSE
- .gitignore

### Essential Features
- NOAA API data extraction
- Data transformation and cleaning using Python
- Storing data in a cloud-based server
- Displaying analytics and trends in Power BI

### Non-Essential Features
- Frontend customer display of the data
- Stream processing
- Multi-lingual capabilities

## Resources
- [ChatGPT](https://chat.openai.com)
- [DataTalksClub YouTube Channel](https://www.youtube.com/@DataTalksClub)
- [NOAA Web Services API v2](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)



### Timeline:

- Created new functions group_noaa_data, store_noaa_data, parse_noaa_data and fetch_noaa_data inside of noaa_client.py
- These functions fetch the data from the NOAA API, then parse and group the data by datatype. Then stores the data inside of a csv found in /data/noaa_weather_data.csv
- Increased the amount of data being pulled to a decade worth of data from only a singular station instead of the entire state.
- Added dimension and fact tables.
- Added pagination in order to process the large amount of data in the data ingestion.



### Next Day Plans:
- Do station-specifc file names to sort the data
    - Future thinking for scaling to multiple stations
- Adding Parquet conversion
- Creating unit tests inside of the /tests folder in order to throughly test my process