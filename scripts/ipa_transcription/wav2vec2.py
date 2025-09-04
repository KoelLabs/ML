#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_array, audio_file_to_array

import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC

# set espeak library path for macOS
import sys

if sys.platform == "darwin":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    _ESPEAK_LIBRARY = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib"
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_IDS = [
    "KoelLabs/xlsr-english-01",  # Our newest model
    "KoelLabs/xlsr-timit-b0",  # Our first model
    "facebook/wav2vec2-lv-60-espeak-cv-ft",  # Samir's recommended best for English
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",  # very similar to 60
    "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",  # Recommended by Samir, but not for English
    "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa-plus-2000",  # better quality, slightly slower but also not for English
    "ginic/wav2vec-large-xlsr-en-ipa",  # OK quality, also annotates stressed syllables with ' as the only (ginic) model to do so
    "ginic/data_seed_4_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/gender_split_30_female_5_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/gender_split_70_female_5_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/vary_individuals_old_only_3_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/vary_individuals_young_only_3_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/hyperparam_tuning_1_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "speech31/wav2vec2-large-TIMIT-IPA",  # works quite well and has word boundaries as the only model
    "speech31/wav2vec2-large-TIMIT-IPA2",  # works quite well but no word boundaries
    "speech31/wav2vec2-large-english-TIMIT-phoneme_v3",  # works quite well but no word boundaries
    "speech31/XLS-R-300m-english-ipa",  # slightly weirder spelling
    "speech31/wavlm-large-english-ipa",  # adds extra sounds that are not there
    "speech31/hubert-base-english-ipa",  # adds extra sounds
    "snu-nia-12/wav2vec2-large_nia12_phone-ipa_english",  # works quite well
    "Jubliano/wav2vec2-large-xls-r-300m-ipa",  # quite big and slow to load, very weird transcriptions
    "Jubliano/wav2vec2-large-xls-r-300m-ipa-nl",  # smaller, still weird
    "Jubliano/wav2vec2-large-xls-r-300m-ipa-INTERNATIONAL1.5",  # OK, a bit unconventional spelling
    "Jubliano/wav2vec2-large-xls-r-300m-ipa-INTERNATIONAL1.9.2WithoutSpaces",  # not bad, not good
    "vitouphy/wav2vec2-xls-r-300m-timit-phoneme",  # specifically for arpabet phoneme prediction, works well, similar to snu-nia-12
    "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme",  # works quite well, esp. for L2 and real convo English speech
]


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_model(model_id, device=DEVICE):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device)
    return model, processor


def transcribe_batch(batch, model, processor):
    input_values = (
        processor(
            [x[1] for x in batch],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return [processor.decode(ids) for ids in predicted_ids]


def transcribe_batch_filtered(batch, model, processor, vocab):
    input_values = (
        processor(
            [x[1] for x in batch],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # probabilities[:, :, processor.tokenizer.pad_token_id] /= 1.2
    probabilities[:, :, processor.tokenizer.pad_token_id] = probabilities[
        :, :, processor.tokenizer.pad_token_id
    ] * (probabilities[:, :, processor.tokenizer.pad_token_id] > 0.5)

    # filter out unwanted tokens
    target_vocab = set("".join(vocab.values()))
    for t in set(processor.tokenizer.vocab.keys()).difference(target_vocab):
        if t in processor.tokenizer.special_tokens_map.values():
            continue
        probabilities[:, :, processor.tokenizer.vocab[t]] = 0

    predicted_ids = torch.argmax(probabilities, dim=-1)
    return [processor.decode(ids) for ids in predicted_ids]


def transcribe_batch_timestamped(batch, model, processor):
    input_values = (
        processor(
            [x[1] for x in batch],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids_batch = torch.argmax(logits, dim=-1)
    transcription_batch = [processor.decode(ids) for ids in predicted_ids_batch]

    # get the start and end timestamp for each phoneme
    phonemes_with_time_batch = []
    for predicted_ids in predicted_ids_batch:
        predicted_ids = predicted_ids.tolist()
        duration_sec = input_values.shape[1] / processor.feature_extractor.sampling_rate

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

        phonemes_with_time_batch.append(phonemes_with_time)

    return transcription_batch, phonemes_with_time_batch


def transcribe_from_file(input_path, model, processor, include_timestamps):
    wav_array = audio_file_to_array(input_path).astype(np.float32) / 32768  # type: ignore

    if include_timestamps:
        transcription_batch, phonemes_with_time_batch = transcribe_batch_timestamped(
            [(None, wav_array)], model, processor
        )
        return transcription_batch[0], phonemes_with_time_batch[0]
    else:
        return transcribe_batch([(None, wav_array)], model, processor)[0]


def transcribe_from_mic(model, processor, include_timestamps):
    wav_array = audio_record_to_array().astype(np.float32) / 32768
    if include_timestamps:
        transcription_batch, phonemes_with_time_batch = transcribe_batch_timestamped(
            [(None, wav_array)], model, processor
        )
        return transcription_batch[0], phonemes_with_time_batch[0]
    else:
        return transcribe_batch([(None, wav_array)], model, processor)[0]


def main(args):
    if len(args) < 1:
        print(
            "Usage: python ./scripts/ipa_transcription/wav2vec2.py <audio file> [model_id] [--timestamped]"
        )
        print(
            "Usage: python ./scripts/ipa_transcription/wav2vec2.py mic [model_id] [--timestamped]"
        )
        return

    input_path = args[0]
    model_id = args[1] if len(args) > 1 and args[1] != "--timestamped" else MODEL_IDS[0]
    include_timestamps = "--timestamped" in args
    model, processor = load_model(model_id)
    if input_path == "mic":
        print(transcribe_from_mic(model, processor, include_timestamps))
    else:
        print(transcribe_from_file(input_path, model, processor, include_timestamps))


if __name__ == "__main__":
    main(sys.argv[1:])
