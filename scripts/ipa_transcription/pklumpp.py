#!/usr/bin/env python3

import sys
import os
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_record_to_file, audio_file_to_array

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from repos.phd_model.phonetics.ipa import symbol_to_descriptor, to_symbol
from repos.phd_model.model.wav2vec2 import Wav2Vec2

import torch
import numpy as np
from fast_ctc_decode import viterbi_search  # type: ignore

MODEL_IDS = ["pklumpp/Wav2Vec2_CommonPhone"]

models = {}


def decode_lattice(
    lattice: np.ndarray,
    enc_feats: "np.ndarray | None" = None,
    cnn_feats: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]":
    """
    Blank index must be 0
    Input lattice is expected in the form of (T, S), without batch dimension
    Outputs state sequence (phones), along with encoder features, cnn features and softmax probability of emitted symbol
    """
    _, path = viterbi_search(lattice, alphabet=list(np.arange(lattice.shape[-1])))
    probs = lattice[path, :]
    states = np.argmax(probs, axis=1)
    probs = probs[np.arange(len(states)), states]
    enc = None
    if enc_feats is not None:
        enc = enc_feats[path]
    cnn = None
    if cnn_feats is not None:
        cnn = cnn_feats[path]
    return states, enc, cnn, probs


def pklumpp_transcribe(input_path, model_id=MODEL_IDS[0]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_id not in models:
        # Load model from Huggingface hub
        wav2vec2 = Wav2Vec2.from_pretrained(
            "pklumpp/Wav2Vec2_CommonPhone",
        )
        wav2vec2.to(device)
        wav2vec2.eval()

        # Save model
        models[model_id] = wav2vec2
    else:
        wav2vec2 = models[model_id]

    # Load audio file
    audio = audio_file_to_array(input_path)
    audio = audio.reshape((1, audio.shape[0]))

    # IMPORTANT: Always standardize input audio
    mean = audio.mean()
    std = audio.std()
    audio = (audio - mean) / (std + 1e-9)

    # Create torch tensor, move to device and feed the model
    audio = torch.tensor(
        audio,
        dtype=torch.float,
        device=device,
    )
    with torch.no_grad():
        y_pred, enc_features, cnn_features = wav2vec2(audio)

    # Decode CTC output for first sample in batch
    phone_sequence, enc_feats, cnn_feats, probs = decode_lattice(
        lattice=y_pred[0].cpu().numpy(),
        enc_feats=enc_features[0].cpu().numpy(),
        cnn_feats=cnn_features[0].cpu().numpy(),
    )
    # phone_sequence contains indices right now. Convert to actual IPA symbols
    symbol_sequence = [to_symbol(i) for i in phone_sequence]

    for symbol in symbol_sequence:
        print(symbol_to_descriptor(symbol))

    return "".join(symbol_sequence)


def pklumpp_transcribe_from_mic(model_id=MODEL_IDS[0]):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return pklumpp_transcribe(f.name, model_id)


def main(args):
    model_id = MODEL_IDS[0] if len(args) < 2 else args[1]
    if args[0] == "mic":
        print(pklumpp_transcribe_from_mic(model_id))
    else:
        try:
            input_path = args[0]
            print(pklumpp_transcribe(input_path, model_id))
        except Exception as e:
            print(e)
            print("Usage: python ./scripts/ipa_transcription/pklumpp.py mic [model_id]")
            print(
                "Usage: python ./scripts/ipa_transcription/pklumpp.py <input_wav_path> [model_id]"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
