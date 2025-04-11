#!/usr/bin/env python3

# Manage files in Google Cloud Storage Bucket
# Useful for certain APIs that require a file to be uploaded to GCS before processing

import os
import sys
import uuid
from contextlib import contextmanager

from google.cloud import storage
from google.cloud.exceptions import NotFound

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.load_secrets import load_secrets

load_secrets()

BUCKET_NAME = os.environ.get("GOOGLE_BUCKET_NAME")
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")

storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)
bucket.storage_class = "STANDARD"


def _check_bucket_exists():
    try:
        bucket.reload()
        return True
    except NotFound as e:
        return False


if not _check_bucket_exists():
    bucket = storage_client.create_bucket(
        bucket,
        location="us",
    )


def list_files(prefix):
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]


def upload_file(file_path, key):
    try:
        blob = bucket.blob(key)
        blob.upload_from_filename(file_path)
        return True
    except:
        print(f"Failed to upload {file_path} to {key}.")
        return False


def upload_from_bytes(input_bytes, key=None):
    key = key or str(uuid.uuid4())
    blob = bucket.blob(key)
    blob.upload_from_string(input_bytes)
    return key


def download_file(key, file_path):
    try:
        blob = bucket.blob(key)
        blob.download_to_filename(file_path)
        return True
    except:
        print(f"Failed to download {key} to {file_path}.")
        return False


def delete_file(key):
    try:
        blob = bucket.blob(key)
        blob.delete()
        return True
    except:
        print(f"Failed to delete {key}.")
        return False


@contextmanager
def create_temp_object_from_bytes(input_bytes, key=None):
    key = key or str(uuid.uuid4())
    blob = bucket.blob(key)
    blob.upload_from_string(input_bytes)
    try:
        yield key
    finally:
        blob.delete()


@contextmanager
def create_temp_object(file_path, key=None):
    key = key or str(uuid.uuid4())
    blob = bucket.blob(key)
    blob.upload_from_filename(file_path)
    try:
        yield key
    finally:
        blob.delete()


def get_gcs_uri(key):
    return f"gs://{BUCKET_NAME}/{key}"


def get_presigned_url(key, expiration=3600):
    blob = bucket.blob(key)
    url = blob.generate_signed_url(expiration=expiration)
    return url


def main(args):
    if args[0] == "upload":
        file_path = args[1]
        key = args[2]
        upload_file(file_path, key)
    elif args[0] == "download":
        key = args[1]
        file_path = args[2]
        download_file(key, file_path)
    elif args[0] == "delete":
        key = args[1]
        delete_file(key)
    elif args[0] == "list":
        prefix = args[1] if len(args) > 1 else ""
        print(list_files(prefix))
    else:
        print(
            "Usage: python ./scripts/core/gcs.py <upload|download|delete|list> [args]"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
