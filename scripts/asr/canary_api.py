import os
import sys
import requests
import numpy as np
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.s3 import get_presigned_url, create_temp_object
from core.audio import audio_record_to_array, audio_array_to_wav_file


def canary_transcribe_file(audio_path):
    with create_temp_object(audio_path) as key:
        audio_url = get_presigned_url(key, expiration=7200)
        url = "https://nvidia-canary-qwen-2-5b.hf.space/gradio_api/call/transcribe"
        res = requests.post(
            url,
            json={
                "data": [
                    {
                        "path": audio_url,
                        "meta": {"_type": "gradio.FileData"},
                    }
                ]
            },
        )
        res.raise_for_status()
        event_id = res.json().get("event_id") or res.text.strip('"')
        output_url = f"{url}/{event_id}"
        res = requests.get(output_url)
        res.raise_for_status()
        text = res.text.replace("\\n", "\n").strip()
        lines = text.split("event: complete\ndata: ")[1].split("\n\n")[:-1]
        return "".join(l.split("] ")[1] for l in lines if l).strip()


def canary_transcribe_from_array(input_array):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(input_array, f.name)
        return canary_transcribe_file(f.name)


def canary_transcribe_from_mic():
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    return canary_transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/canary_api.py <audio file>")
        print("Usage: python ./scripts/asr/canary_api.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(canary_transcribe_from_mic())
    else:
        print(canary_transcribe_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
