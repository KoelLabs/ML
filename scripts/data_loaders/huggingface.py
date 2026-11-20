# Script to upload datasets to huggingface
# Usage: python ./scripts/data_loaders/huggingface.py DoReCo EpaDB L2Arctic L2ArcticSpontaneousSplit Buckeye PSST SpeechOcean SpeechOceanNoTH TIMIT ISLE SpeechAccentArchive

import os
import sys
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetDict, Dataset, Audio, Features, Value
from huggingface_hub import CommitOperationAdd, HfApi

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE, audio_array_to_bytes

from data_loaders.DoReCo import (
    COMMERCIAL_LANGUAGES,
    TOP_DIR as DORECO_DIR,
    DoReCoDataset,
)


def ipa_tokens(timestamped_phonemes):
    return [phone for phone, _, _ in timestamped_phonemes if phone]


# ==================================================
# =================== Generators ===================
def gen_doreco(dataset):
    for ipa, audio, phones, metadata, text in dataset:
        yield {
            "audio": {"bytes": audio_array_to_bytes(audio), "path": None},
            "ipa": ipa,
            "text": text,
            "language": metadata["language"],
            "recording": metadata["name"],
            "speaker_code": metadata["utt_spk_code"],
            "utterance_ref": metadata["utterance_ref"],
            "phones": [
                {"ipa": phone, "start_sample": start, "end_sample": end}
                for phone, start, end in phones
            ],
        }


def export_doreco(root=DORECO_DIR):
    """Upload the Parquet splits listed in the existing dataset card."""
    repo_id = "KoelLabs/DoReCo"
    api = HfApi()
    repo = api.dataset_info(repo_id)
    if not repo.gated:
        raise ValueError(f"Expected the gated dataset {repo_id}")

    schema = Features(
        {
            "audio": Audio(sampling_rate=TARGET_SAMPLE_RATE),
            **dict.fromkeys(
                (
                    "ipa",
                    "text",
                    "language",
                    "recording",
                    "speaker_code",
                    "utterance_ref",
                ),
                Value("string"),
            ),
            "phones": [
                {
                    "ipa": Value("string"),
                    "start_sample": Value("int32"),
                    "end_sample": Value("int32"),
                }
            ],
        }
    ).arrow_schema
    operations = []
    with TemporaryDirectory() as staging:
        for code in COMMERCIAL_LANGUAGES:
            dataset = DoReCoDataset(
                language=code,
                root=root,
                include_timestamps=True,
                include_speaker_info=True,
                include_text=True,
            )
            missing = dataset.missing_recordings()
            if missing:
                raise FileNotFoundError(
                    f"Missing DoReCo recordings for {code}: {missing}"
                )
            if not len(dataset):
                raise ValueError(f"No DoReCo utterances for {code}")
            filename = f"{code}.parquet"
            path = Path(staging) / filename
            print(f"Preparing DoReCo {code}", flush=True)
            with pq.ParquetWriter(path, schema, compression="zstd") as writer:
                rows = gen_doreco(dataset)
                while batch := list(islice(rows, 64)):
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            operations.append(CommitOperationAdd(f"multilingual/{filename}", path))

        return api.create_commit(
            repo_id,
            operations=operations,
            repo_type="dataset",
            commit_message="Upload DoReCo multilingual splits",
            parent_commit=repo.sha,
        )


def gen_epadb(split):
    def generator():
        from core.text import english2ipa
        from data_loaders.EpaDB import EpaDBDataset

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
                "g2p": english2ipa(sample[3]),  # type: ignore
                "speaker_code": metadata["speaker_id"],  # type: ignore
            }

    return generator


def gen_l2arctic(split):
    def generator():
        from core.text import english2ipa
        from data_loaders.L2ARCTIC import L2ArcticDataset, all_arctic_speaker_splits

        dataset = (
            L2ArcticDataset(
                split="suitcase_corpus",
                include_timestamps=True,
                include_speaker_info=True,
                include_text=True,
            )
            if split == "spontaneous"
            else all_arctic_speaker_splits(
                include_timestamps=True,
                include_speaker_info=True,
                include_text=True,
            )
        )
        for sample in dataset:
            ipa, audio, timestamped_phonemes, metadata, text = sample  # type: ignore
            assert audio.dtype == np.int16  # type: ignore
            yield {
                "audio": {"array": audio.astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": ipa,  # type: ignore
                "ipa_tokens": ipa_tokens(timestamped_phonemes),  # type: ignore
                "text": text,  # type: ignore
                "g2p": english2ipa(text),  # type: ignore
                "speaker_code": metadata["id"],  # type: ignore
                "speaker_gender": metadata["gender"].lower(),  # type: ignore
                "speaker_native_language": metadata["native-language"],  # type: ignore
            }

    return generator


def gen_l2arctic_suitcase_split():
    from core.text import english2ipa
    from data_loaders.common import split_utterance_into_multiple
    from data_loaders.L2ARCTIC import L2ArcticDataset
    from forced_alignment.needleman_wunsch import needleman_wunsch

    # suitcase
    dataset = L2ArcticDataset(
        split="suitcase_corpus",
        include_timestamps=True,
        include_speaker_info=True,
        include_text=True,
    )
    for sample in dataset:
        _, audio, timestamped_phonemes, metadata, text = sample  # type: ignore
        g2p = english2ipa(text)  # type: ignore
        ipa, g2p = needleman_wunsch(list(filter(lambda p: p, map(lambda p: p[0], timestamped_phonemes))), g2p)  # type: ignore
        for subsample in split_utterance_into_multiple(timestamped_phonemes, audio, 0.01, 10):  # type: ignore
            sub_ipa, sub_audio, sub_timestamped_phonemes = subsample  # type: ignore
            assert sub_audio.dtype == np.int16  # type: ignore

            subg2p = []
            for p, _, _ in sub_timestamped_phonemes:
                if not p:
                    continue
                phone = ipa.pop(0)
                while phone == "-":
                    subg2p.append(g2p.pop(0).replace("-", ""))
                    phone = ipa.pop(0)
                assert phone == p
                subg2p.append(g2p.pop(0).replace("-", ""))

            yield {
                "audio": {"array": sub_audio.astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": sub_ipa,  # type: ignore
                "ipa_tokens": ipa_tokens(sub_timestamped_phonemes),  # type: ignore
                "g2p": "".join(subg2p),
                "speaker_code": metadata["id"],  # type: ignore
                "speaker_gender": metadata["gender"].lower(),  # type: ignore
                "speaker_native_language": metadata["native-language"],  # type: ignore
            }


def gen_buckeye():
    from core.text import english2ipa
    from data_loaders.OSUBuckeye import all_buckeye_speaker_splits

    dataset = all_buckeye_speaker_splits(include_speaker_info=True, include_text=True)
    for sample in dataset:
        assert sample[1].dtype == np.int16  # type: ignore
        metadata = sample[2]  # type: ignore
        yield {
            "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
            "ipa": sample[0],  # type: ignore
            "text": sample[3],  # type: ignore
            "g2p": english2ipa(sample[3]),  # type: ignore
            "speaker_code": metadata["id"],
            "speaker_gender": metadata["gender"][0],
            "speaker_age": metadata["age"],
            "interviewer_gender": metadata["interviewer_gender"],
        }


def gen_buckeye_split():
    from core.text import english2ipa
    from data_loaders.common import split_utterance_into_multiple
    from data_loaders.OSUBuckeye import all_buckeye_speaker_splits
    from forced_alignment.needleman_wunsch import needleman_wunsch

    dataset = all_buckeye_speaker_splits(
        include_timestamps=True,
        include_speaker_info=True,
        include_text=True,
    )
    for sample in dataset:
        metadata = sample[3]  # type: ignore
        g2p = english2ipa(sample[4])  # type: ignore
        ipa, g2p = needleman_wunsch(list(filter(lambda p: p, map(lambda p: p[0], sample[2]))), g2p)  # type: ignore
        for subsample in split_utterance_into_multiple(sample[2], sample[1], 0.01, 30):  # type: ignore
            assert subsample[1].dtype == np.int16  # type: ignore

            subg2p = []
            for p, _, _ in subsample[2]:
                if not p:
                    continue
                phone = ipa.pop(0)
                while phone == "-":
                    subg2p.append(g2p.pop(0).replace("-", ""))
                    phone = ipa.pop(0)
                assert phone == p
                subg2p.append(g2p.pop(0).replace("-", ""))

            yield {
                "audio": {"array": subsample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": subsample[0],  # type: ignore
                "g2p": "".join(subg2p),
                "speaker_code": metadata["id"],
                "speaker_gender": metadata["gender"][0],
                "speaker_age": metadata["age"],
                "interviewer_gender": metadata["interviewer_gender"],
            }


def gen_psst(split):
    def generator():
        from core.text import english2ipa
        from data_loaders.PSST import PSSTDataset

        dataset = PSSTDataset(
            split=split,
            include_speaker_info=True,
            include_text=True,
            force_offline=True,
        )
        for sample in dataset:
            assert sample[1].dtype == np.int16  # type: ignore
            metadata = sample[2]  # type: ignore
            yield {
                "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
                "ipa": sample[0],  # type: ignore
                "text": sample[3],  # type: ignore
                "g2p": english2ipa(sample[3]),  # type: ignore
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
        from core.text import english2ipa
        from data_loaders.SpeechOcean import SpeechOceanDataset

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
                "g2p": english2ipa(sample[3]),  # type: ignore
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
        from core.text import english2ipa
        from data_loaders.TIMIT import TIMITDataset

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
                "g2p": english2ipa(sample[3]),  # type: ignore
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
    from core.text import english2ipa
    from data_loaders.ISLE import ISLEDataset

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
            "g2p": english2ipa(sample[3]),  # type: ignore
            "speaker_code": metadata["speaker_id"],  # type: ignore
            "speaker_native_language": metadata["native_language"],  # type: ignore
            "recording_session": metadata["session"],  # type: ignore
            "recording_sub_dir": metadata["sub_dir"],  # type: ignore
            "comments": metadata["comments"],  # type: ignore
        }


def gen_speech_accent_archive():
    from core.text import english2ipa
    from data_loaders.SpeechAccent import SpeechAccentDataset

    dataset = SpeechAccentDataset(include_speaker_info=True, include_text=True)
    for sample in dataset:
        assert sample[1].dtype == np.int16  # type: ignore
        metadata = sample[2]  # type: ignore
        yield {
            "audio": {"array": sample[1].astype(np.float32) / np.iinfo(np.int16).max, "sampling_rate": TARGET_SAMPLE_RATE},  # type: ignore
            "ipa": sample[0],  # type: ignore
            "text": sample[3],  # type: ignore
            "g2p": english2ipa(sample[3]),  # type: ignore
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
        print(export_doreco().commit_url)
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
