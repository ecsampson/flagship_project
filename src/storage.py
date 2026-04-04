import boto3
import pandas as pd


def store_noaa_data(data, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)
    print(f"NOAA data stored in {file_path}")
    parquet_path = file_path.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy', version='2.4')
    print(f"NOAA data stored in {parquet_path}")

def store_extreme_weather(extreme_events, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(extreme_events)
    df.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)
    print(f"Extreme weather events stored in {file_path}")
    parquet_path = file_path.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy', version='2.4')
    print(f"Extreme weather events stored in {parquet_path}")

def store_weather_signals(weather_signals, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    weather_signals.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)
    print(f"Weather signals stored in {file_path}")
    parquet_path = file_path.with_suffix('.parquet')
    weather_signals.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy', version='2.4')
    print(f"Weather signals stored in {parquet_path}")

def upload_to_s3(file_path, data_type, bucket_name):
    """Upload a local file to S3 under raw/ or extreme/ depending on data_type.

    AWS credentials are resolved automatically by boto3 from the environment
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).

    Args:
        file_path: Path to the local file to upload.
        data_type: "raw", "extreme", "features", "dimensions", or "facts".
        bucket_name: Target S3 bucket name.
    """
    if data_type not in ("raw", "extreme", "features", "dimensions", "facts"):
        raise ValueError(f"data_type must be 'raw', 'extreme', 'features', 'dimensions', or 'facts', got '{data_type}'")

    s3_client = boto3.client("s3")

    fmt = "parquet" if file_path.suffix == ".parquet" else "csv"
    s3_key = f"{data_type}/{fmt}/{file_path.name}"
    s3_client.upload_file(str(file_path), bucket_name, s3_key)
    print(f"Uploaded {file_path.name} to s3://{bucket_name}/{s3_key}")
