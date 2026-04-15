import boto3

s3 = boto3.client('s3')
s3.upload_file('lambda_package.zip', 'flagship-project-weather-data', 'lambda/lambda_package.zip')
print("Uploaded to S3")

lambda_client = boto3.client('lambda', region_name='us-east-1')
response = lambda_client.update_function_code(
    FunctionName='flagship_weather_pipeline',
    S3Bucket='flagship-project-weather-data',
    S3Key='lambda/lambda_package.zip'
)
print(f"Lambda updated: {response['LastUpdateStatus']}")
