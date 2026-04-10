import hmac
import os
import sqlite3
import sys
import tempfile
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

google_credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
google_credentials_contents = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_FILE")
if (
    google_credentials_path
    and google_credentials_contents
    and not os.path.exists(google_credentials_path)
):
    with open(google_credentials_path, "w") as f:
        f.write(google_credentials_contents)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.asr.deepspeech import deepspeech_transcribe_from_file
from scripts.asr.google_speech import google_transcribe_from_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = (
    Path("/data/folder_metadata.sqlite3")
    if Path("/data").exists()
    else BASE_DIR / "folder_metadata.sqlite3"
)
MAX_FOLDER_PATH_LENGTH = 512
MAX_ASSET_ID_LENGTH = 255


def _get_configured_api_key() -> str:
    api_key = os.environ.get("API_KEY", "").strip()
    if len(api_key) < 32:
        raise RuntimeError(
            "API_KEY must be set to a random secret with at least 32 characters."
        )
    return api_key


def get_db_path() -> Path:
    raw_path = os.environ.get("MUX_FOLDER_METADATA_DB_PATH", "").strip()
    return Path(raw_path) if raw_path else DEFAULT_DB_PATH


def get_db() -> sqlite3.Connection:
    connection = sqlite3.connect(get_db_path())
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    connection = get_db()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS mux_asset_folder_metadata (
                asset_id TEXT PRIMARY KEY,
                folder_path TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_mux_asset_folder_metadata_folder_path
            ON mux_asset_folder_metadata (folder_path)
            """)
        connection.commit()
    finally:
        connection.close()


def normalize_folder_path(value: str) -> str:
    parts = [part.strip() for part in value.split("/") if part.strip()]
    normalized = "/".join(parts)
    if len(normalized) > MAX_FOLDER_PATH_LENGTH:
        raise ValueError(
            f"folderPath must be {MAX_FOLDER_PATH_LENGTH} characters or fewer."
        )
    return normalized


def validate_asset_id(asset_id: str) -> str:
    normalized = asset_id.strip()
    if not normalized:
        raise ValueError("assetId is required.")
    if len(normalized) > MAX_ASSET_ID_LENGTH:
        raise ValueError(f"assetId must be {MAX_ASSET_ID_LENGTH} characters or fewer.")
    if not normalized.isalnum():
        raise ValueError("assetId must be alphanumeric.")
    return normalized


def require_json_body() -> dict:
    if not request.is_json:
        raise ValueError("Content-Type must be application/json.")
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def get_bearer_token() -> str:
    header = request.headers.get("Authorization", "").strip()
    if not header.startswith("Bearer "):
        return ""
    return header[7:].strip()


def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            expected_api_key = _get_configured_api_key()
        except RuntimeError:
            return jsonify({"error": "Server auth is not configured."}), 500

        provided_api_key = get_bearer_token()
        if not provided_api_key:
            return jsonify({"error": "Authorization required."}), 401
        if not hmac.compare_digest(provided_api_key, expected_api_key):
            return jsonify({"error": "Unauthorized."}), 401
        return f(*args, **kwargs)

    return decorated_function


@app.after_request
def apply_security_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/api/v1/mux-folder-metadata", methods=["GET"])
@api_key_required
def list_mux_folder_metadata():
    connection = get_db()
    try:
        rows = connection.execute("""
            SELECT asset_id, folder_path, updated_at
            FROM mux_asset_folder_metadata
            ORDER BY asset_id ASC
            """).fetchall()
    finally:
        connection.close()

    return (
        jsonify(
            {
                "items": [
                    {
                        "assetId": row["asset_id"],
                        "folderPath": row["folder_path"],
                        "updatedAt": row["updated_at"],
                    }
                    for row in rows
                ]
            }
        ),
        200,
    )


@app.route("/api/v1/mux-folder-metadata/<asset_id>", methods=["PUT"])
@api_key_required
def put_mux_folder_metadata(asset_id: str):
    try:
        normalized_asset_id = validate_asset_id(asset_id)
        body = require_json_body()
        folder_path = normalize_folder_path(str(body.get("folderPath", "")))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    connection = get_db()
    try:
        connection.execute(
            """
            INSERT INTO mux_asset_folder_metadata (asset_id, folder_path, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(asset_id) DO UPDATE SET
                folder_path = excluded.folder_path,
                updated_at = CURRENT_TIMESTAMP
            """,
            (normalized_asset_id, folder_path),
        )
        row = connection.execute(
            """
            SELECT asset_id, folder_path, updated_at
            FROM mux_asset_folder_metadata
            WHERE asset_id = ?
            """,
            (normalized_asset_id,),
        ).fetchone()
        connection.commit()
    finally:
        connection.close()

    return (
        jsonify(
            {
                "item": {
                    "assetId": row["asset_id"],
                    "folderPath": row["folder_path"],
                    "updatedAt": row["updated_at"],
                }
            }
        ),
        200,
    )


@app.route("/api/v1/mux-folder-metadata/<asset_id>", methods=["DELETE"])
@api_key_required
def delete_mux_folder_metadata(asset_id: str):
    try:
        normalized_asset_id = validate_asset_id(asset_id)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    connection = get_db()
    try:
        connection.execute(
            "DELETE FROM mux_asset_folder_metadata WHERE asset_id = ?",
            (normalized_asset_id,),
        )
        connection.commit()
    finally:
        connection.close()

    return jsonify({"ok": True}), 200


@app.route("/api/v1/asr/<model>", methods=["POST"])
@api_key_required
def run_asr(model: str):
    models = {
        "deepspeech": deepspeech_transcribe_from_file,
        "google": lambda f: google_transcribe_from_file(f)
        .results[0]
        .alternatives[0]
        .transcript,
    }

    if model not in models:
        return (
            f"Invalid model '{model}', must choose one from {list(models.keys())}",
            400,
        )

    if "audio_file" not in request.files or not request.files["audio_file"].filename:
        return "No audio file part in the request", 400

    file = request.files["audio_file"]
    filename = secure_filename(file.filename)  # type: ignore[arg-type]
    with tempfile.NamedTemporaryFile(
        delete=False,
        prefix=f"{uuid.uuid4()}_",
        suffix=f"_{filename}",
        dir=BASE_DIR,
    ) as temp_file:
        filepath = temp_file.name
    file.save(filepath)

    try:
        return models[model](filepath), 200
    except Exception as error:
        return f"Transcription failed: {error}", 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


init_db()


if __name__ == "__main__":
    init_db()

    # run dev server with: python browser_tests/run_models/main.py
    # test with: curl -X POST -F "audio_file=@data/ExamplesWithComments/TIMIT_sample_0.wav" http://127.0.0.1:5000/api/v1/asr/deepspeech -H "Authorization: Bearer <secure-api-key>"
    # test with: curl -X GET http://127.0.0.1:5000/api/v1/mux-folder-metadata -H "Authorization: Bearer <secure-api-key>"
    app.run(debug=True)
