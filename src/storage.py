import io
import json

import boto3
import pandas as pd
from botocore.exceptions import ClientError


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


def upload_parquet_append_to_s3(df, bucket, s3_key, dedup_cols):
    """Read existing Parquet from S3, append new rows, deduplicate, write back.

    If the S3 object does not exist yet, the new data is written as-is.
    On overlap (same dedup key present in both old and new data), the new
    row wins (keep="last" after concat).

    Args:
        df:         DataFrame of new records to append.
        bucket:     S3 bucket name.
        s3_key:     Full S3 object key, e.g. "raw/parquet/USW00014922_weather_data.parquet".
        dedup_cols: Column names that together form the natural/business key.
    """
    if df.empty:
        print(f"No new rows to append for s3://{bucket}/{s3_key} — skipping.")
        return

    s3 = boto3.client("s3")

    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        existing = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        combined = pd.concat([existing, df], ignore_index=True)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            combined = df.copy()
        else:
            raise

    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    dupes_dropped = before - len(combined)

    buf = io.BytesIO()
    combined.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=buf.getvalue())

    print(
        f"s3://{bucket}/{s3_key} — "
        f"appended {len(df)} rows, dropped {dupes_dropped} dupes, "
        f"{len(combined)} total rows"
    )


def get_last_run_date(bucket):
    """Return the last successfully processed end date from S3 metadata.

    Reads s3://{bucket}/metadata/last_run_date.json.
    Falls back to "1938-01-01" (earliest available data for USW00014922)
    if the file does not exist, triggering a full backfill on the first run.

    Args:
        bucket: S3 bucket name.

    Returns:
        ISO date string, e.g. "2024-03-15".
    """
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key="metadata/last_run_date.json")
        data = json.loads(obj["Body"].read())
        last_date = data["last_run_date"]
        print(f"Resuming from last run date: {last_date}")
        return last_date
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print("No last_run_date.json found — starting full backfill from 1938-01-01.")
            return "1938-01-01"
        raise


def save_last_run_date(date_str, bucket):
    """Persist the successfully processed end date to S3 metadata.

    Writes {"last_run_date": date_str} to
    s3://{bucket}/metadata/last_run_date.json.
    Called only after a successful pipeline run so a failed run does not
    advance the cursor.

    Args:
        date_str: ISO date string, e.g. "2024-03-15".
        bucket:   S3 bucket name.
    """
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key="metadata/last_run_date.json",
        Body=json.dumps({"last_run_date": date_str}),
        ContentType="application/json",
    )
    print(f"Saved last_run_date {date_str} to s3://{bucket}/metadata/last_run_date.json")
