import os
import sys
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_array_to_wav_file, audio_record_to_array
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")


def seamlessm4t_transcribe_from_file(input_path: str):
    audio, orig_freq = torchaudio.load(input_path)
    audio = torchaudio.functional.resample(
        audio, orig_freq=orig_freq, new_freq=16_000
    )  # must be a 16 kHz waveform array
    audio_inputs = processor(audios=audio, return_tensors="pt")
    output_tokens = model.generate(
        **audio_inputs, tgt_lang="eng", generate_speech=False
    )
    text_from_audio = processor.decode(
        output_tokens[0].tolist()[0], skip_special_tokens=True
    )
    return text_from_audio


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
