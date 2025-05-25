import os
import zipfile
import tempfile
from contextlib import contextmanager


class WriteableZip:
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name

        # If it's an existing file path
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.temp_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # When done, re-pack
        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(self.temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.temp_dir)
                    z.write(full_path, rel_path)
        self.temp_dir_obj.cleanup()

    def namelist(self):
        file_list = []
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.temp_dir)
                file_list.append(rel_path.replace(os.sep, "/"))
        return file_list

    def open(self, name, mode="r"):
        # Only support text and binary read/write for files in the temp dir
        # Mode can be "r", "w", "rb", "wb", etc.
        abs_path = os.path.join(self.temp_dir, name)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        # Support binary/text modes
        return open(abs_path, mode)


@contextmanager
def writeable_zip(zip_path):
    """
    Yields a WriteableZip object that mimics ZipFile but allows reading/writing.
    """
    wz = WriteableZip(zip_path)
    try:
        yield wz
    finally:
        wz.__exit__(None, None, None)
