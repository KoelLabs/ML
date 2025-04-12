#!/usr/bin/env python3

import os, sys
import warnings
from speechbrain.dataio.encoder import logger

warnings.filterwarnings("ignore", category=FutureWarning)
logger.setLevel("ERROR")

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.core.audio import audio_record_to_array

import torch
import torchaudio
import numpy as np
from langcodes import standardize_tag
from speechbrain.inference.classifiers import EncoderClassifier


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data")
ID_TO_LANGUAGE = {0: 'ab: Abkhazian',1: 'af: Afrikaans',2: 'am: Amharic',3: 'ar: Arabic',4: 'as: Assamese',5: 'az: Azerbaijani',6: 'ba: Bashkir',7: 'be: Belarusian',8: 'bg: Bulgarian',9: 'bn: Bengali',10: 'bo: Tibetan',11: 'br: Breton',12: 'bs: Bosnian',13: 'ca: Catalan',14: 'ceb: Cebuano',15: 'cs: Czech',16: 'cy: Welsh',17: 'da: Danish',18: 'de: German',19: 'el: Greek',20: 'en: English',21: 'eo: Esperanto',22: 'es: Spanish',23: 'et: Estonian',24: 'eu: Basque',25: 'fa: Persian',26: 'fi: Finnish',27: 'fo: Faroese',28: 'fr: French',29: 'gl: Galician',30: 'gn: Guarani',31: 'gu: Gujarati',32: 'gv: Manx',33: 'ha: Hausa',34: 'haw: Hawaiian',35: 'hi: Hindi',36: 'hr: Croatian',37: 'ht: Haitian',38: 'hu: Hungarian',39: 'hy: Armenian',40: 'ia: Interlingua',41: 'id: Indonesian',42: 'is: Icelandic',43: 'it: Italian',44: 'iw: Hebrew',45: 'ja: Japanese',46: 'jw: Javanese',47: 'ka: Georgian',48: 'kk: Kazakh',49: 'km: Central Khmer',50: 'kn: Kannada',51: 'ko: Korean',52: 'la: Latin',53: 'lb: Luxembourgish',54: 'ln: Lingala',55: 'lo: Lao',56: 'lt: Lithuanian',57: 'lv: Latvian',58: 'mg: Malagasy',59: 'mi: Maori',60: 'mk: Macedonian',61: 'ml: Malayalam',62: 'mn: Mongolian',63: 'mr: Marathi',64: 'ms: Malay',65: 'mt: Maltese',66: 'my: Burmese',67: 'ne: Nepali',68: 'nl: Dutch',69: 'nn: Norwegian Nynorsk',70: 'no: Norwegian',71: 'oc: Occitan',72: 'pa: Panjabi',73: 'pl: Polish',74: 'ps: Pushto',75: 'pt: Portuguese',76: 'ro: Romanian',77: 'ru: Russian',78: 'sa: Sanskrit',79: 'sco: Scots',80: 'sd: Sindhi',81: 'si: Sinhala',82: 'sk: Slovak',83: 'sl: Slovenian',84: 'sn: Shona',85: 'so: Somali',86: 'sq: Albanian',87: 'sr: Serbian',88: 'su: Sundanese',89: 'sv: Swedish',90: 'sw: Swahili',91: 'ta: Tamil',92: 'te: Telugu',93: 'tg: Tajik',94: 'th: Thai',95: 'tk: Turkmen',96: 'tl: Tagalog',97: 'tr: Turkish',98: 'tt: Tatar',99: 'uk: Ukrainian',100: 'ur: Urdu',101: 'uz: Uzbek',102: 'vi: Vietnamese',103: 'war: Waray',104: 'yi: Yiddish',105: 'yo: Yoruba',106: 'zh: Chinese'}  # fmt: skip

assert (
    len(torchaudio.list_audio_backends()) > 0
), "Torchaudio backend not found. Try `pip install soundfile==0.13.1`"

language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir=os.path.join(DATA_DIR, "speechbrain/lang-id-voxlingua107-ecapa"),
)
assert language_id, "Failed to load the model. Check the path and try again."


def _prediction_to_top_languages(prediction, topk=3):
    topk_indices = torch.topk(prediction[0], topk).indices
    return [
        (
            standardize_tag(ID_TO_LANGUAGE.get(int(i.item()), "en").split(": ")[0]),
            prediction[0][0][i].exp().item(),
        )
        for i in topk_indices[0]
    ]


def identify_language_from_array(
    wav_array: np.ndarray,
) -> "tuple[str, float, list[tuple[str, float]]]":
    if wav_array.dtype != np.float32:
        wav_array = wav_array.astype(np.float32) / 32768

    # add batch dimension
    signal = torch.from_numpy(wav_array).unsqueeze(0)
    prediction = language_id.classify_batch(signal)  # type: ignore
    return (
        standardize_tag(prediction[3][0].split(": ")[0]),
        prediction[1].exp().item(),
        _prediction_to_top_languages(prediction, topk=3),
    )


def identify_language_from_file(
    audio_path: str,
) -> "tuple[str, float, list[tuple[str, float]]]":
    signal = language_id.load_audio(audio_path)  # type: ignore
    prediction = language_id.classify_batch(signal)  # type: ignore
    return (
        standardize_tag(prediction[3][0].split(": ")[0]),
        prediction[1].exp().item(),
        _prediction_to_top_languages(prediction, topk=3),
    )


def identify_language_from_mic():
    signal = audio_record_to_array().astype(np.float32) / 32768
    return identify_language_from_array(signal)


def main(args):
    if len(args) != 1:
        print(
            "Usage: python ./scripts/language_identification/speechbrain_encoder.py <audio file>"
        )
        print(
            "Usage: python ./scripts/language_identification/speechbrain_encoder.py mic"
        )
        return

    input_path = args[0]
    if input_path == "mic":
        print(identify_language_from_mic())
    else:
        print(identify_language_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
