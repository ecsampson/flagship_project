def store_noaa_data(data, file_path):
    with open(file_path, "w") as f:
        f.write("Date,Type,Value\n")
        for item in data:
            f.write(f"{item['date']},{item['type']},{item['value']}\n")
    print(f"NOAA data stored in {file_path}")


def store_extreme_weather(extreme_events, file_path):
    with open(file_path, "w") as f:
        f.write("Date,Type,Value,Threshold,Condition\n")
        for item in extreme_events:
            f.write(f"{item['date']},{item['type']},{item['value']},{item['threshold']},{item['condition']}\n")
    print(f"Extreme weather events stored in {file_path}")
