import torch
from transformers import AutoProcessor, AutoModelForCTC

import sys
import os
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.audio import audio_record_to_file, audio_file_to_array
from scripts.ipa_transcription.xlsr import MODEL_IDS

processors: "dict[str, AutoProcessor]" = {}
models: "dict[str, AutoModelForCTC]" = {}


def xlsr_transcribe_timestamped(input_path, model_id=MODEL_IDS[0]):
    processor: AutoProcessor = processors.get(
        model_id
    ) or AutoProcessor.from_pretrained(model_id)
    processors[model_id] = processor
    model: AutoModelForCTC = (
        models.get(model_id) or AutoModelForCTC.from_pretrained(model_id).cpu()
    )
    models[model_id] = model

    SAMPLE_RATE = 16000
    speech = audio_file_to_array(input_path, desired_sample_rate=SAMPLE_RATE)
    input_values = (
        processor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        .input_values.cpu()
        .type(torch.float32)
    )

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # get the start and end timestamp for each phoneme
    predicted_ids = predicted_ids[0].tolist()
    duration_sec = input_values.shape[1] / SAMPLE_RATE

    ids_w_time = [
        (i / len(predicted_ids) * duration_sec, _id)
        for i, _id in enumerate(predicted_ids)
    ]

    current_phoneme_id = processor.tokenizer.pad_token_id
    current_start_time = 0
    phonemes_with_time = []
    for time, _id in ids_w_time:
        if current_phoneme_id != _id:
            if current_phoneme_id != processor.tokenizer.pad_token_id:
                phonemes_with_time.append(
                    (processor.decode(current_phoneme_id), current_start_time, time)
                )
            current_start_time = time
            current_phoneme_id = _id

    return transcription, phonemes_with_time


def xlsr_transcribe_timestamped_from_mic(model_id=MODEL_IDS[0]):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return xlsr_transcribe_timestamped(f.name, model_id)


def main(args):
    model_id = MODEL_IDS[0] if len(args) < 2 else args[1]
    if args[0] == "mic":
        print(xlsr_transcribe_timestamped_from_mic(model_id))
    else:
        try:
            input_path = args[0]
            print(xlsr_transcribe_timestamped(input_path, model_id))
        except Exception as e:
            print(e)
            print(
                "Usage: python ./scripts/ipa_transcription/xlsr_timestamped.py mic [model_id]"
            )
            print(
                "Usage: python ./scripts/ipa_transcription/xlsr_timestamped.py <input_wav_path> [model_id]"
            )
            print(
                "Test with: python ./scripts/audio.py play ./data/hello_tts.wav 0.16:0.32"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
