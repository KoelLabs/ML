# L1 American English Dialects from https://catalog.ldc.upenn.edu/LDC93S1

import os
import sys

import zipfile


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import audio_bytes_to_array
from core.codes import parse_timit
from core.text import english2ipa, remove_punctuation

TIMIT_ZIP = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "TIMIT.zip")
SOURCE_SAMPLE_RATE = 16000


class TIMITDataset(BaseDataset):
    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        include_g2p=False,
        g2p_filter_type="letters_rmv_tie",
    ):
        super().__init__(
            split,
            include_timestamps,
            include_speaker_info,
            include_text,
            include_g2p,
            g2p_filter_type,
        )
        self.zip = zipfile.ZipFile(TIMIT_ZIP, "r")
        files = self.zip.namelist()
        self.files = list(
            set(
                map(
                    lambda x: x.split(".")[0],
                    filter(lambda x: x.startswith("data/" + split.upper()), files),
                )
            )
        )

    def __del__(self):
        self.zip.close()

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        filename = self.files[ix]
        speaker_id = filename.split("/")[-2][-4:]
        speaker = SPEAKERS[speaker_id]

        with self.zip.open(filename + ".WAV.wav") as wav_file:
            audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)

        with self.zip.open(filename + ".PHN") as phn_file:
            timestamped_phonemes = parse_timit(
                phn_file.read().decode("utf-8").split("\n")
            )
        ipa = "".join([x[0] for x in timestamped_phonemes])

        if self.include_text:
            with self.zip.open(filename + ".TXT") as txt_file:
                text = " ".join(txt_file.read().decode("utf-8").strip().split(" ")[2:])
                text = remove_punctuation(text)
        if self.include_g2p:
            g2p_ipa = english2ipa(
                text, filter_type=self.g2p_filter_type
            )  # removes punctuation and converts to ipa

        start_signal = timestamped_phonemes.pop(0)
        audio = audio[start_signal[2] :]
        timestamped_phonemes = [
            (x[0], x[1] - start_signal[2], x[2] - start_signal[2])
            for x in timestamped_phonemes
        ]

        result = []
        result.append(ipa)
        result.append(audio)
        if self.include_timestamps:
            result.append(timestamped_phonemes)
        if self.include_speaker_info:
            result.append(speaker)
        if self.include_text:
            result.append(text)
        if self.include_g2p:
            result.append(g2p_ipa)

        return tuple(result)


DIALECTS = {
    1: "New England",
    2: "Northern",
    3: "North Midland",
    4: "South Midland",
    5: "Southern",
    6: "New York City",
    7: "Western",
    8: "Army Brat",
}

SPEAKERS = {}
with zipfile.ZipFile(TIMIT_ZIP, "r") as zip:
    with zip.open("SPKRINFO.TXT") as file:
        # read as csv with ; indicating comment and double space as separator
        for line in file:
            line = line.decode("utf-8").strip()
            if not line.startswith(";"):
                parts = line.split("  ")
                SPEAKERS[parts[0]] = {
                    "SEX": parts[1],
                    "DIALECT": DIALECTS[int(parts[2])],
                    "SPLIT": parts[3],
                    "RECORDING_DATE": parts[4],
                    "BIRTH_DATE": parts[5],
                    "HEIGHT": parts[6],
                    "RACE": parts[7],
                    "EDUCATION": parts[8],
                }
                if len(parts) == 10:
                    SPEAKERS[parts[0]]["COMMENTS"] = parts[9]

if __name__ == "__main__":
    dataset = TIMITDataset()
    print(dataset[0])
