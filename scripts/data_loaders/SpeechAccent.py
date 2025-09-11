# Speech Accent Archive: https://accent.gmu.edu/

import os
import sys

import zipfile
import textgrids
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import (
    audio_bytes_to_wav_array,
    audio_file_to_array,
    TARGET_SAMPLE_RATE,
)
from core.codes import xsampa2ipa, arpabet2ipa

KAGGLE_ZIP = os.path.join(
    os.path.dirname(__file__), "..", "..", ".data", "speech-accent-snake-snack.zip"
)
FULL_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "..", ".data", "SpeechAccentArchive"
)
FULL_DATASET_METADATA = os.path.join(
    FULL_DATASET, "excel_spreadsheets", "speakers-1 november 2023.xlsx"
)
# FULL_DATASET_WAVS = os.path.join(FULL_DATASET, "processed_wav_files")
FULL_DATASET_TRANSCRIPTS = os.path.join(FULL_DATASET, "textgrids")
VALID_SPLITS = ["full", "kaggle"]

WORD_LIST = {'please', 'call', 'stella', 'stellaw', 'ask', 'her', 'to', 'bring', 'these', 'things', 'with', 'from', 'the', 'store', 'six', 'spoons', 'of', 'fresh', 'snowpeas', 'five', 'thick', 'slabs', 'blue', 'cheese', 'and', 'maybe', 'a', 'snack', 'for', 'brother', 'bob', 'we', 'also', 'need', 'small', 'plastic', 'snake', 'big', 'toy', 'frog', 'for', 'kids', 'she', 'can', 'scoop', 'things', 'into', 'three', 'red', 'bags', 'we', 'will', 'go', 'meet', 'her', 'wednesday', 'at', 'train', 'station', 'stationx', '-del', 'sp'} # fmt: skip


def split_words_with_dictionary(text, word_list=WORD_LIST):
    n = len(text)
    dp = [False] * (n + 1)
    dp[0] = True
    result = [[]] * (n + 1)

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j]:
                word = text[j:i]
                if word.lower() in word_list:
                    dp[i] = True
                    if j == 0:
                        result[i] = [word]
                    else:
                        result[i] = result[j] + [word]
                    break

    assert dp[n], text
    return result[n] if dp[n] else text


def _textgrid_contains_tier(textgrid_path, tier):
    tg = textgrids.TextGrid()
    with open(textgrid_path, "rb") as f:
        tg.parse(f.read())
    return tier in tg.keys()


class SpeechAccentDataset(BaseDataset):
    def __init__(
        self,
        split="full",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        include_samples_without_phonemes=False,
        max_phonemes=None,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)
        self.max_phonemes = max_phonemes

        assert split in VALID_SPLITS, (
            split + " is not a valid split in " + str(VALID_SPLITS)
        )
        if split == "kaggle":
            assert (
                include_timestamps == False
            ), "Kaggle dataset is outdated and does not contain timestamp annotations"
            assert (
                include_text == False
            ), "Kaggle dataset is outdated and does not support text transcriptions"

            print(
                "WARNING: Kaggle dataset is outdated and is missing IPA and text transcriptions"
            )
            self.zip = zipfile.ZipFile(KAGGLE_ZIP)
            with self.zip.open("speakers_all.csv") as f:
                self.df = pd.read_csv(f)
                self.df = self.df[~self.df["file_missing?"]]
        elif split == "full":
            assert (
                not include_samples_without_phonemes
            ), "Parsing of samples without phoneme annotations not handled yet"

            self.df = pd.read_excel(FULL_DATASET_METADATA, dtype={"notes": str})
            # NOTE: some audio files in processed_wav_files have no corresponding textgrid
            # for some use cases they might be useful, for now, they are not included
            self.textgrids = [
                os.path.join(FULL_DATASET_TRANSCRIPTS, subdir, file)
                for subdir in os.listdir(FULL_DATASET_TRANSCRIPTS)
                if os.path.isdir(os.path.join(FULL_DATASET_TRANSCRIPTS, subdir))
                for file in os.listdir(os.path.join(FULL_DATASET_TRANSCRIPTS, subdir))
                if file.endswith(".TextGrid")
                and (
                    include_samples_without_phonemes
                    or _textgrid_contains_tier(
                        os.path.join(FULL_DATASET_TRANSCRIPTS, subdir, file), "phones"
                    )
                    or _textgrid_contains_tier(
                        os.path.join(FULL_DATASET_TRANSCRIPTS, subdir, file), "MAU"
                    )
                )
            ]

    def __del__(self):
        if self.split == "kaggle":
            self.zip.close()

    def __len__(self):
        if self.split == "kaggle":
            return len(self.df)
        else:
            return len(self.textgrids)

    def _get_ix(self, ix):
        if self.split == "full":
            transcript_path = self.textgrids[ix]
            tg = textgrids.TextGrid()
            with open(transcript_path, "rb") as f:
                tg.parse(f.read())

            audio_path = transcript_path.removesuffix(".TextGrid") + ".wav"
            audio = audio_file_to_array(audio_path)

            try:
                phone_tier, word_tier, phone2ipa = "MAU", "ORT", xsampa2ipa
                phonemes = tg.interval_tier_to_array(phone_tier)
            except:
                phone_tier, word_tier, phone2ipa = "phones", "words", arpabet2ipa
                phonemes = tg.interval_tier_to_array(phone_tier)
            if phonemes[0]["label"].startswith("(...)"):
                phone2ipa = lambda x: x

            # TODO: use rtf files to fix textgrid annotations for IPA
            timestamped_phonemes = [
                (
                    phone2ipa(c["label"]).removeprefix("(...)"),
                    int(c["begin"] * TARGET_SAMPLE_RATE),
                    int(c["end"] * TARGET_SAMPLE_RATE),
                )
                for c in phonemes
                if c["label"] and c["label"] not in ["sil", "<p:>", "(...)"]
            ]
            if self.max_phonemes is not None:
                timestamped_phonemes = timestamped_phonemes[: self.max_phonemes]
                audio = audio[: timestamped_phonemes[-1][2]]
            ipa = "".join(t[0] for t in timestamped_phonemes)

            # TODO: clean up word annotations instead of this approximated processing
            words = tg.interval_tier_to_array(word_tier)
            if self.max_phonemes is not None:
                words = [
                    w
                    for w in words
                    if int(w.get("xmin", w["begin"]) * TARGET_SAMPLE_RATE)
                    <= timestamped_phonemes[-1][2]
                ]
            to_remove = [
                "chatter",
                "chatter2",
                "selfintroduction",
                "self-talk",
                "experimenter_comments_taketwo",
            ]
            offset = 0
            for word in words:
                if word["label"].strip() in to_remove:
                    start, end = int(
                        word.get("xmin", word["begin"]) * TARGET_SAMPLE_RATE
                    ), int(word.get("xmax", word["end"]) * TARGET_SAMPLE_RATE)
                    before = audio[: start - offset]
                    after = audio[end - offset :]
                    audio = np.append(before, after)
                    offset += end - start
            text = " ".join(
                c["label"].strip()
                for c in words
                if c["label"] and c["label"] not in [*to_remove, "sil", "sp"]
            )
            text = text.replace("`", "")
            text = (
                text.replace("-noise", "")
                .replace("-hes", "")
                .replace("-insrt", "")
                .replace("-laugh", "")
                .replace("-sub", "")
                .replace("-missing", "")
                .replace("-birdsinging", "")
                .replace("-cough", "")
                .replace("-check-as", "")
                .replace("-check", "")
                .replace("-change", "")
                .replace("-experimenterspeechinbackground", "")
                .replace("-missound", "")
                .replace("-rep-del", "")
                .replace("-jes", "")
                .replace("1", "")
                .replace("_s", "")
                .replace("-regroup", "")
                .replace("-kof", "")
            )
            text = text.replace("2", "to")
            parts = (
                (
                    (t.replace("-rep", ""), t.replace("-rep", ""))
                    if t.endswith("-rep")
                    else (t,)
                )
                for t in text.split(" ")
            )
            parts = (t.removesuffix("-").removesuffix("-rep") for p in parts for t in p)
            parts = [
                w for t in parts for w in split_words_with_dictionary(t) if w != "sp"
            ]
            while "-del" in parts:
                ix = parts.index("-del")
                parts = parts[: ix - 1] + parts[ix + 1 :]
            parts = ("stella" if p == "stellaw" else p for p in parts)
            parts = ("station" if p == "stationx" else p for p in parts)
            text = " ".join(parts)

            sample_id = (
                transcript_path.split(os.path.sep)[-1].removesuffix(".TextGrid")
                + ".mp3"
            )
            # Remap some mismatched sample ids
            remap = {
                "poonchi1.mp3": "pahari1.mp3",
                "japanese1a.mp3": "japanese1.mp3",
                "kirgiz1.mp3": "kyrgyz2.mp3",
                "sinhalese1.mp3": "sinhala1.mp3",
                "sinhalese2.mp3": "sinhala2.mp3",
                "sinhalese3.mp3": "sinhala3.mp3",
                "sinhalese4.mp3": "sinhala4.mp3",
                "sa_a1.mp3": "sa'a1.mp3",
            }
            assert (
                sample_id not in remap.values() or sample_id == "japanese1.mp3"
            ), sample_id
            sample_id = remap.get(sample_id, sample_id)

            row = self.df[self.df["speech_sample"] == sample_id].iloc[0]
            row.fillna("", inplace=True)
            metadata_dict = {
                "speakerid": row["speakerid"],
                "native_language": row["native_language"],
                "native_language_alternative": row["alternative_native_language"],
                "birthplace": row["city"]
                + (", " + row["state_or_province"] if row["state_or_province"] else ""),
                "country": row["country"],
                "age": int(row["age"]),
                "gender": row["gender"],
                "spoken_english_since_age": int(row["onset_age"]),
                "english_residence_length_years": float(row["length_of_residence"]),
                "learning_style": row["learning_style"],  # academic vs naturalistic
                "ethnologue_language_code": row["ethnologue_language_code"],
                "notes": row["notes"],
                "audio_path": audio_path,
            }
            if row["english_residence"] in ["NULL", 3.5, "mac system 8.5"]:
                metadata_dict["english_residence"] = ""
            else:
                metadata_dict["english_residence"] = row["english_residence"]
        elif self.split == "kaggle":
            row = self.df.iloc[ix]
            filename = row["filename"].replace("'i", "")
            with self.zip.open(f"recordings/recordings/{filename}.mp3") as f:
                audio = audio_bytes_to_wav_array(f.read(), "mp3")

            ipa = None
            text = None
            timestamped_phonemes = None
            metadata_dict = {
                "age": int(row["age"]),
                "spoken_english_since_age": int(row["age_onset"]),
                "birthplace": row["birthplace"],
                "native_language": row["native_language"],
                "gender": row["sex"],
                "country": row["country"],
                "speakerid": row["speakerid"],
            }

        result = [ipa, audio]
        if self.include_timestamps:
            result.append(timestamped_phonemes)
        if self.include_speaker_info:
            result.append(metadata_dict)
        if self.include_text:
            result.append(text)
        return tuple(result)


if __name__ == "__main__":
    dataset = SpeechAccentDataset(include_speaker_info=True, include_text=True)
    for _ in dataset:
        pass
    print(dataset[0])
