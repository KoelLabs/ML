import os
import sys
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import (
    audio_array_to_wav_file,
    audio_record_to_array,
    TARGET_SAMPLE_RATE,
)
import librosa
import soundfile as sf
from transformers import AutoProcessor, SeamlessM4Tv2Model

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

CHUNK_SECONDS = 10


def chunk_audio(audio, sample_rate):
    chunk_size = CHUNK_SECONDS * sample_rate
    return [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]


def seamlessm4t_transcribe_chunk(audio_chunk):
    audio_inputs = processor(
        audio=audio_chunk,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        src_lang="eng",
    )

    output_tokens = model.generate(
        **audio_inputs,
        tgt_lang="eng",
        generate_speech=False,
        max_new_tokens=512,
    )

    text = processor.batch_decode(
        output_tokens[0],
        skip_special_tokens=True,
    )[0]

    return text


def seamlessm4t_transcribe_from_file(input_path: str):
    audio, orig_freq = sf.read(input_path, dtype="float32")

    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample
    audio = librosa.resample(
        audio,
        orig_sr=orig_freq,
        target_sr=TARGET_SAMPLE_RATE,
    )

    chunks = chunk_audio(audio, TARGET_SAMPLE_RATE)

    transcripts = []
    for chunk in chunks:
        transcripts.append(seamlessm4t_transcribe_chunk(chunk))

    return " ".join(transcripts)


def seamlessm4t_transcribe_from_array(wav_array):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(wav_array, f.name)
        return seamlessm4t_transcribe_from_file(f.name)


def seamlessm4t_transcribe_from_mic():
    wav_array = audio_record_to_array()
    return seamlessm4t_transcribe_from_array(wav_array)


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/asr/seamlessm4t.py <audio file>")
        print("Usage: python ./scripts/asr/seamlessm4t.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        print(seamlessm4t_transcribe_from_mic())
    else:
        print(seamlessm4t_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
