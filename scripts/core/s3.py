#!/usr/bin/env python3

# Manage files in AWS S3 bucket
# Useful for certain APIs that require a file to be uploaded to S3 before processing

import os
import sys
import uuid
from contextlib import contextmanager
from urllib import parse
from tempfile import NamedTemporaryFile

import boto3
from boto3.s3.transfer import S3UploadFailedError
from botocore.exceptions import ClientError

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.load_secrets import load_secrets

load_secrets()

BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")

s3 = boto3.resource("s3")
bucket = s3.Bucket(BUCKET_NAME)  # type: ignore


def list_files(prefix):
    return [obj.key for obj in bucket.objects.filter(Prefix=prefix)]


def upload_file(file_path, key):
    try:
        bucket.upload_file(file_path, key)
    except S3UploadFailedError as e:
        print(f"Failed to upload {file_path} to {key}. {e}")
        return False
    return True


def download_file(key, file_path):
    try:
        bucket.download_file(key, file_path)
    except ClientError as e:
        print(f"Failed to download {key} to {file_path}. {e}")
        return False
    return True


def delete_file(key):
    try:
        bucket.Object(key).delete()
    except ClientError as e:
        print(f"Failed to delete {key}. {e}")
        return False
    return True


def get_presigned_url(key, expiration=3600):
    return s3.meta.client.generate_presigned_url(  # type: ignore
        ClientMethod="get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=expiration,
    )


@contextmanager
def create_temp_object(file_path, key=None):
    key = key or str(uuid.uuid4())
    bucket.upload_file(
        file_path,
        key,
        ExtraArgs={
            "Tagging": parse.urlencode({"temp": "true"})
        },  # temp tag ensures that the file is deleted after 1 day in case of failure
    )
    yield key
    bucket.Object(key).delete()


@contextmanager
def create_temp_object_from_bytes(input_bytes, key=None):
    with NamedTemporaryFile() as temp_file:
        temp_file.write(input_bytes)
        temp_file.flush()
        temp_file.seek(0)
        with create_temp_object(temp_file.name, key) as key:  # type: ignore
            yield key


def main(args):
    if args.command == "list":
        print(list_files(args.prefix))
    elif args.command == "upload":
        key = args.key or os.path.basename(args.file_path)
        if upload_file(args.file_path, key):
            print(f"Uploaded {args.file_path} to {key}")
    elif args.command == "download":
        file_path = args.file_path or os.path.basename(args.key)
        if download_file(args.key, file_path):
            print(f"Downloaded {args.key} to {file_path}")
    elif args.command == "delete":
        if delete_file(args.key):
            print(f"Deleted {args.key}")
    elif args.command == "presigned-url":
        print(get_presigned_url(args.key, args.expiration))
    else:
        print("Invalid command")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage files on S3")
    parser.add_argument(
        "command", choices=["list", "upload", "download", "delete", "presigned-url"]
    )
    parser.add_argument("--prefix", default="")
    parser.add_argument("--key", default="")
    parser.add_argument("--file-path", default="")
    parser.add_argument("--expiration", type=int, default=3600)  # seconds

    args = parser.parse_args()
    main(args)
