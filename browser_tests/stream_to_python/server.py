from flask import Flask, send_from_directory, request
from flask_cors import CORS, cross_origin
from flask_sock import Sock
import torch
from transformers import AutoProcessor, AutoModelForCTC
import numpy as np
import scipy.io.wavfile as wavfile
import webrtcvad

from feedback import score_words_cer, score_words_wfed, feedback
import json

# Constants
SAMPLE_RATE = 16000
NUM_SECONDS_PER_CHUNK = 2
USE_VAD = False
NUM_CHUNKS_ACCUMULATED = 5
DEBUG = False

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)  # allow CORS for all domains on all routes.
app.config["CORS_HEADERS"] = "Content-Type"
sock = Sock(app)

# Load Wav2Vec2 model
model_id = "KoelLabs/xlsr-timit-b0"
# model_id = "speech31/wav2vec2-large-english-TIMIT-phoneme_v3"
# model_id = "ginic/hyperparam_tuning_1_wav2vec2-large-xlsr-buckeye-ipa"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id)


def transcribe_audio(audio: np.ndarray) -> str:
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


def confidence_score(logits, predicted_ids) -> "tuple[np.ndarray, float]":
    scores = torch.nn.functional.softmax(logits, dim=-1)
    pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
    mask = torch.logical_and(
        predicted_ids.not_equal(processor.tokenizer.word_delimiter_token_id),
        predicted_ids.not_equal(processor.tokenizer.pad_token_id),
    )

    character_scores = pred_scores.masked_select(mask)
    total_average = torch.sum(character_scores) / len(character_scores)
    return character_scores.numpy(), total_average.float().item()


# Speech Cleaning
vad = webrtcvad.Vad(1)


def zero_out_non_speech(audio: np.ndarray) -> np.ndarray:
    audio = audio.copy()
    audio_int16 = np.int16(audio * 32768).tobytes()
    frame_size = 2 * SAMPLE_RATE * 30 // 1000
    for f in range(0, len(audio_int16), frame_size):
        frame = audio_int16[f : f + frame_size]
        if len(frame) < frame_size:
            frame += b"\0" * (frame_size - len(frame))
        if not vad.is_speech(frame, SAMPLE_RATE):
            audio[f : f + frame_size] = 0
    return audio


# server /
@app.route("/")
@cross_origin()
def index():
    return send_from_directory("static", "index.html")


# serve static files
@app.route("/<path:path>")
@cross_origin()
def send_static(path):
    return send_from_directory("static", path)


# REST endpoint
@app.route("/score_words_cer", methods=["GET"])
@cross_origin()
def get_score_words_cer():
    target = request.args.get("target", "").strip()
    target_by_word = json.loads(request.args.get("tbw") or "null")
    speech = request.args.get("speech", "").strip()
    if not speech:
        return json.dumps([[[word, seq, "", 0] for word, seq in target_by_word], 0])
    word_scores = score_words_cer(target, target_by_word, speech)
    return json.dumps(word_scores)


@app.route("/score_words_wfed", methods=["GET"])
@cross_origin()
def get_score_words_wfed():
    target = request.args.get("target", "").strip()
    target_by_word = json.loads(request.args.get("tbw") or "null")
    speech = request.args.get("speech", "").strip()
    if not speech:
        return json.dumps([[[word, seq, "", 0] for word, seq in target_by_word], 0])
    word_scores = score_words_wfed(target, target_by_word, speech)
    return json.dumps(word_scores)


@app.route("/feedback", methods=["GET"])
@cross_origin()
def get_feedback():
    target = request.args.get("target", "").strip()
    target_by_word = json.loads(request.args.get("tbw") or "null")
    speech = request.args.get("speech", "").strip()
    return json.dumps(feedback(target, target_by_word, speech))


# WebSocket endpoint for transcription
@sock.route("/stream")
@cross_origin()
def stream(ws):
    buffer = b""  # Buffer to hold audio chunks

    full_transcription = ""
    combined = np.array([], dtype=np.float32)
    num_chunks_accumulated = 0
    transcription = ""
    while True:
        try:
            # Receive audio data from the client
            data = ws.receive()
            if data:
                buffer += data
                # Process chunks when buffer reaches certain size
                if (
                    len(buffer) // 4 > SAMPLE_RATE * NUM_SECONDS_PER_CHUNK
                ):  # Adjust size for chunk processing
                    num_chunks_accumulated += 1

                    audio = np.frombuffer(buffer, dtype=np.float32)
                    if USE_VAD:
                        audio = zero_out_non_speech(audio)

                    combined = np.concatenate([combined, audio])

                    if num_chunks_accumulated < NUM_CHUNKS_ACCUMULATED:
                        transcription += transcribe_audio(audio)
                        ws.send(full_transcription + transcription)
                    else:
                        full_transcription += transcribe_audio(combined)
                        ws.send(full_transcription)
                        combined = np.array([], dtype=np.float32)
                        num_chunks_accumulated = 0
                        transcription = ""

                    if DEBUG:
                        wavfile.write("audio.wav", SAMPLE_RATE, audio)
                        wavfile.write("combined.wav", SAMPLE_RATE, combined)

                    buffer = b""  # Clear the buffer
        except Exception as e:
            print(f"Error: {e}")
            print(f"Line: {e.__traceback__.tb_lineno}")
            break


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)
