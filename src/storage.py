import pandas as pd

def store_noaa_data(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)
    print(f"NOAA data stored in {file_path}")

def store_extreme_weather(extreme_events, file_path):
    df = pd.DataFrame(extreme_events)
    df.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)
    print(f"Extreme weather events stored in {file_path}")