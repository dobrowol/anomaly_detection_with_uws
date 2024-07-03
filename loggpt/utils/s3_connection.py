import configparser

import boto3


def get_s3_bucket(bucket_name):

    config = configparser.ConfigParser()
    config.read("s3config.ini")
    access_key = config["default"]["access_key"]
    secret_key = config["default"]["secret_key"]
    host = config["default"]["host_base"]
    host_bucket = config["default"]["host_bucket"]

    session = boto3.Session()

    s3 = session.resource(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url="https://" + host_bucket,
    )

    return s3.Bucket(bucket_name)
