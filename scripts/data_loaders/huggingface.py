# Script to upload datasets to huggingface
# Usage: python ./scripts/data_loaders/huggingface.py DoReCo EpaDB L2Arctic L2ArcticSpontaneousSplit Buckeye PSST SpeechOcean SpeechOceanNoTH TIMIT ISLE SpeechAccentArchive

import os
import sys

import numpy as np
from datasets import DatasetDict, Dataset, Audio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE
from data_loaders.common import split_utterance_into_multiple

from data_loaders.DoReCo import DoReCoDataset
from data_loaders.EpaDB import EpaDBDataset
from data_loaders.L2ARCTIC import L2ArcticDataset, all_arctic_speaker_splits
from data_loaders.OSUBuckeye import all_buckeye_speaker_splits
from data_loaders.PSST import PSSTDataset
from data_loaders.SpeechOcean import SpeechOceanDataset
from data_loaders.TIMIT import TIMITDataset
from data_loaders.ISLE import ISLEDataset
from data_loaders.SpeechAccent import SpeechAccentDataset


# ==================================================
# =================== Generators ===================
def gen_doreco():
    dataset = DoReCoDataset(
        split="all",
        include_speaker_info=True,
        include_text=True,
    )
    for sample in dataset:
        assert sample[1].dtype == np.int16  # type: ignore
        metadata = sample[2]  # type: ignore
        yield {
            "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
            "ipa": sample[0],  # type: ignore
            "text": sample[3],  # type: ignore
            "id": metadata["id"],  # type: ignore
            "speaker_code": metadata["spk_code"],  # type: ignore
            "speaker_age": metadata["spk_age"],  # type: ignore
            "speaker_gender": metadata["spk_sex"],  # type: ignore
            "recording_year": metadata["rec_date"],  # type: ignore
            "recoding_topic": metadata["genre"],  # type: ignore
            "sound_quality": metadata["sound_quality"],  # type: ignore
            "background_noise": metadata["background_noise"],  # type: ignore
        }


def gen_epadb(split):
    def generator():
        dataset = EpaDBDataset(
            split=split,
            include_speaker_info=True,
            include_text=True,
        )
        for sample in dataset:
            assert sample[1].dtype == np.int16  # type: ignore
            metadata = sample[2]  # type: ignore
            yield {
                "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": sample[0],  # type: ignore
                "text": sample[3],  # type: ignore
                "speaker_code": metadata["speaker_id"],  # type: ignore
            }

    return generator


def gen_l2arctic(split):
    def generator():
        dataset = (
            L2ArcticDataset(
                split="suitcase_corpus",
                include_speaker_info=True,
                include_text=True,
            )
            if split == "spontaneous"
            else all_arctic_speaker_splits(
                include_speaker_info=True,
                include_text=True,
            )
        )
        for sample in dataset:
            assert sample[1].dtype == np.int16  # type: ignore
            metadata = sample[2]  # type: ignore
            yield {
                "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": sample[0],  # type: ignore
                "text": sample[3],  # type: ignore
                "speaker_code": metadata["id"],  # type: ignore
                "speaker_gender": metadata["gender"].lower(),  # type: ignore
                "speaker_native_language": metadata["native-language"],  # type: ignore
            }

    return generator


def gen_l2arctic_suitcase_split():
    # suitcase
    dataset = L2ArcticDataset(
        split="suitcase_corpus",
        include_timestamps=True,
        include_speaker_info=True,
    )
    for sample in dataset:
        metadata = sample[3]  # type: ignore
        for subsample in split_utterance_into_multiple(sample[2], sample[1], 0.01, 10):  # type: ignore
            assert subsample[1].dtype == np.int16  # type: ignore
            yield {
                "audio": {"array": subsample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": subsample[0],  # type: ignore
                "speaker_code": metadata["id"],  # type: ignore
                "speaker_gender": metadata["gender"].lower(),  # type: ignore
                "speaker_native_language": metadata["native-language"],  # type: ignore
            }


def gen_buckeye():
    dataset = all_buckeye_speaker_splits(include_speaker_info=True)
    for sample in dataset:
        assert sample[1].dtype == np.int16  # type: ignore
        metadata = sample[2]  # type: ignore
        yield {
            "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
            "ipa": sample[0],  # type: ignore
            "speaker_code": metadata["id"],
            "speaker_gender": metadata["gender"][0],
            "speaker_age": metadata["age"],
            "interviewer_gender": metadata["interviewer_gender"],
        }


def gen_buckeye_split():
    dataset = all_buckeye_speaker_splits(
        include_timestamps=True, include_speaker_info=True
    )
    for sample in dataset:
        metadata = sample[3]  # type: ignore
        for subsample in split_utterance_into_multiple(sample[2], sample[1], 0.01, 30):  # type: ignore
            assert subsample[1].dtype == np.int16  # type: ignore
            yield {
                "audio": {"array": subsample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": subsample[0],  # type: ignore
                "speaker_code": metadata["id"],
                "speaker_gender": metadata["gender"][0],
                "speaker_age": metadata["age"],
                "interviewer_gender": metadata["interviewer_gender"],
            }


def gen_psst(split):
    def generator():
        dataset = PSSTDataset(
            split=split, include_speaker_info=True, force_offline=True
        )
        for sample in dataset:
            assert sample[1].dtype == np.int16  # type: ignore
            metadata = sample[2]  # type: ignore
            yield {
                "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": sample[0],  # type: ignore
                "utterance_id": metadata["utterance_id"],  # type: ignore
                "utterance_text_prompt": metadata["text_prompt"],  # type: ignore
                "recording_test": metadata["test"],  # type: ignore
                "recording_session": metadata["session"],  # type: ignore
                "pronunciation_is_correct": metadata["correct"],  # type: ignore
                "speaker_aq_index": metadata["aq_index"],  # type: ignore
            }

    return generator


def gen_speech_ocean(split):
    def generator():
        dataset = SpeechOceanDataset(
            split=split, include_speaker_info=True, include_text=True
        )
        for sample in dataset:
            assert sample[1].dtype == np.int16  # type: ignore
            metadata = sample[2]  # type: ignore
            yield {
                "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": sample[0],  # type: ignore
                "text": sample[3],  # type: ignore
                "speaker_code": metadata["speaker_id"],  # type: ignore
                "speaker_gender": metadata["gender"],  # type: ignore
                "speaker_age": metadata["age"],  # type: ignore
                "pronunciation_accuracy_0_to_10": metadata["accuracy"],  # type: ignore
                "pronunciation_completeness_fraction": metadata["completeness"],  # type: ignore
                "pauseless_flow_0_to_10": metadata["fluency"],  # type: ignore
                "cadence_and_intonation_0_to_10": metadata["prosodic"],  # type: ignore
            }

    return generator


def gen_speech_ocean_no_th(split):
    # remove samples with "ð" or "θ"
    def generator():
        for sample in gen_speech_ocean(split=split)():
            if "ð" in sample["ipa"] or "θ" in sample["ipa"]:
                continue
            yield sample

    return generator


def gen_timit(split):
    def generator():
        dataset = TIMITDataset(
            split=split, include_speaker_info=True, include_text=True
        )
        for sample in dataset:
            assert sample[1].dtype == np.int16  # type: ignore
            metadata = sample[2]  # type: ignore
            yield {
                "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": sample[0],  # type: ignore
                "text": sample[3],  # type: ignore
                "speaker_code": metadata["id"],  # type: ignore
                "speaker_gender": metadata["SEX"],  # type: ignore
                "speaker_dialect": metadata["DIALECT"],  # type: ignore
                "speaker_birth_date": metadata["BIRTH_DATE"],  # type: ignore
                "speaker_height": metadata["HEIGHT"],  # type: ignore
                "speaker_ethnicity": metadata["RACE"],  # type: ignore
                "speaker_education": metadata["EDUCATION"],  # type: ignore
                "recording_date": metadata["RECORDING_DATE"],  # type: ignore
            }

    return generator


def gen_isle():
    dataset = ISLEDataset(
        split="all",
        include_speaker_info=True,
        include_text=True,
    )
    for sample in dataset:
        assert sample[1].dtype == np.int16  # type: ignore
        metadata = sample[2]  # type: ignore
        yield {
            "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
            "ipa": sample[0],  # type: ignore
            "text": sample[3],  # type: ignore
            "speaker_code": metadata["speaker_id"],  # type: ignore
            "speaker_native_language": metadata["native_language"],  # type: ignore
            "recording_session": metadata["session"],  # type: ignore
            "recording_sub_dir": metadata["sub_dir"],  # type: ignore
            "comments": metadata["comments"],  # type: ignore
        }


def gen_speech_accent_archive():
    dataset = SpeechAccentDataset(include_speaker_info=True, include_text=True)
    for sample in dataset:
        assert sample[1].dtype == np.int16  # type: ignore
        metadata = sample[2]  # type: ignore
        yield {
            "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
            "ipa": sample[0],  # type: ignore
            "text": sample[3],  # type: ignore
            "speaker_code": metadata["speakerid"],  # type: ignore
            "speaker_native_language": metadata["native_language"],  # type: ignore
            "speaker_native_language_alternative": metadata["native_language_alternative"],  # type: ignore
            "speaker_gender": metadata["gender"],  # type: ignore
            "speaker_age": metadata["age"],  # type: ignore
            "speaker_spoken_english_since_age": metadata["spoken_english_since_age"],  # type: ignore
            "speaker_country": metadata["country"],  # type: ignore
            "speaker_birthplace": metadata["birthplace"],  # type: ignore
            "speaker_english_residence": metadata["english_residence"],  # type: ignore
            "speaker_english_residence_length_years": metadata["english_residence_length_years"],  # type: ignore
            "speaker_learning_style": metadata["learning_style"],  # type: ignore
            "speaker_ethnologue_language_code": metadata["ethnologue_language_code"],  # type: ignore
            "comments": metadata["notes"],  # type: ignore
        }


# =================== Generators ===================
# ==================================================
# =================== Push to Hub ==================

if __name__ == "__main__":
    if "DoReCo" in sys.argv:
        print("Pushing DoReCo to the hub...")
        doreco_ds: Dataset = Dataset.from_generator(gen_doreco).cast_column("audio", Audio())  # type: ignore
        doreco_ds.push_to_hub("KoelLabs/DoReCo", private=True)
    if "EpaDB" in sys.argv:
        print("Pushing EpaDB to the hub...")
        epadb_dict = DatasetDict(
            {
                "train": Dataset.from_generator(gen_epadb("train")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
                "test": Dataset.from_generator(gen_epadb("test")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
            }
        )
        epadb_dict.push_to_hub("KoelLabs/EpaDB", private=True)
    if "L2Arctic" in sys.argv:
        print("Pushing L2Arctic to the hub...")
        l2arctic_dict = DatasetDict(
            {
                "spontaneous": Dataset.from_generator(gen_l2arctic("spontaneous")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
                "scripted": Dataset.from_generator(gen_l2arctic("scripted")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
            }
        )
        l2arctic_dict.push_to_hub("KoelLabs/L2Arctic", private=True)
    if "L2ArcticSpontaneousSplit" in sys.argv:
        suitcase_ds = Dataset.from_generator(gen_l2arctic_suitcase_split).cast_column("audio", Audio())  # type: ignore
        suitcase_ds.push_to_hub("KoelLabs/L2ArcticSpontaneousSplit", private=True)  # type: ignore
    if "Buckeye" in sys.argv:
        print("Pushing Buckeye to the hub...")
        buckeye_ds: Dataset = Dataset.from_generator(gen_buckeye_split).cast_column("audio", Audio())  # type: ignore
        buckeye_ds.push_to_hub("KoelLabs/Buckeye", private=True)
    if "PSST" in sys.argv:
        print("Pushing PSST to the hub...")
        psst_dict = DatasetDict(
            {
                "train": Dataset.from_generator(gen_psst("train")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
                "valid": Dataset.from_generator(gen_psst("valid")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
                "test": Dataset.from_generator(gen_psst("test")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
            }
        )
        psst_dict.push_to_hub("KoelLabs/PSST", private=True)
    if "SpeechOcean" in sys.argv:
        print("Pushing SpeechOcean to the hub...")
        speech_ocean_dict = DatasetDict(
            {
                "train": Dataset.from_generator(gen_speech_ocean("train")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
                "test": Dataset.from_generator(gen_speech_ocean("test")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
            }
        )
        speech_ocean_dict.push_to_hub("KoelLabs/SpeechOcean", private=True)
    if "SpeechOceanNoTH" in sys.argv:
        print("Pushing SpeechOcean without TH samples to the hub...")
        speech_ocean_no_th_dict = DatasetDict(
            {
                "train": Dataset.from_generator(gen_speech_ocean_no_th("train")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
                "test": Dataset.from_generator(gen_speech_ocean_no_th("test")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
            }
        )
        speech_ocean_no_th_dict.push_to_hub("KoelLabs/SpeechOceanNoTH", private=True)
    if "TIMIT" in sys.argv:
        print("Pushing TIMIT to the hub...")
        timit_dict = DatasetDict(
            {
                "train": Dataset.from_generator(gen_timit("train")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
                "test": Dataset.from_generator(gen_timit("test")).cast_column(  # type: ignore
                    "audio", Audio()
                ),
            }
        )
        timit_dict.push_to_hub("KoelLabs/TIMIT", private=True)
    if "ISLE" in sys.argv:
        print("Pushing ISLE to the hub...")
        isle_ds: Dataset = Dataset.from_generator(gen_isle).cast_column("audio", Audio())  # type: ignore
        isle_ds.push_to_hub("KoelLabs/ISLE", private=True)
    if "SpeechAccentArchive" in sys.argv:
        print("Pushing Speech Accent Archive to the hub...")
        saa_ds: Dataset = Dataset.from_generator(gen_speech_accent_archive).cast_column("audio", Audio())  # type: ignore
        saa_ds.push_to_hub("KoelLabs/SpeechAccentArchive", private=True)
