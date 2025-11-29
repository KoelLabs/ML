import os
import sys
import uuid
from functools import wraps

if not os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
    with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "w") as f:
        f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_FILE"])

from flask import Flask, request
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.asr.deepspeech import deepspeech_transcribe_from_file
from scripts.asr.google_speech import google_transcribe_from_file

app = Flask(__name__)


def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = str(request.headers.get("Authorization")).split(" ")[-1]
        if not api_key:
            return "API Key is missing", 401
        if api_key != os.environ.get("API_KEY", "secret"):
            return "Invalid API Key", 403
        return f(*args, **kwargs)

    return decorated_function


@app.route("/api/v1/asr/<model>", methods=["POST"])
@api_key_required
def run_asr(model: str):
    MODELS = {
        "deepspeech": deepspeech_transcribe_from_file,
        "google": lambda f: google_transcribe_from_file(f)
        .results[0]
        .alternatives[0]
        .transcript,
    }

    if model not in MODELS:
        return f"Invalid model '{model}', must choose one from {MODELS.keys()}", 400

    if "audio_file" not in request.files or not request.files["audio_file"].filename:
        return "No audio file part in the request", 400

    file = request.files["audio_file"]
    filename = secure_filename(file.filename)  # type: ignore
    filepath = os.path.join(
        os.path.dirname(__file__), str(uuid.uuid4()) + "_" + filename
    )
    file.save(filepath)

    try:
        return MODELS[model](filepath), 200
    except Exception as e:
        return f"Transcription failed: {e}", 500
    finally:
        os.remove(filepath)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    # run dev server with: python browser_tests/run_models/main.py
    # test with: curl -X POST -F "audio_file=@data/ExamplesWithComments/TIMIT_sample_0.wav" http://127.0.0.1:5000/api/v1/asr/deepspeech -H "Authorization: Bearer secret"
    app.run(debug=True)
