# VoxAngeles: phone-level segmentations of the UCLA Phonetics Lab Archive
# https://github.com/pacscilab/voxangeles
# Audio mirror: https://huggingface.co/datasets/speech31/voxangeles
#
# Download instructions:
#   Audio and audited phoneme strings are loaded through Hugging Face datasets:
#       python -c "from datasets import load_dataset; load_dataset('speech31/voxangeles', split='test')"
#   Phone timestamps require the VoxAngeles durations table:
#       mkdir -p ./.data/VoxAngeles
#       curl -L -o ./.data/VoxAngeles/voxangeles_durations.tsv \
#         https://raw.githubusercontent.com/pacscilab/voxangeles/main/data/phonetic_measurements/voxangeles_durations.tsv
#
# TODO: manual verification

import os
import sys

import numpy as np
import pandas as pd
from datasets import DownloadConfig, load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import TARGET_SAMPLE_RATE, audio_resample

HF_DATASET = "speech31/voxangeles"
HF_SPLIT = "test"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "VoxAngeles")
DURATIONS_TSV = os.path.join(DATA_DIR, "voxangeles_durations.tsv")


def read_durations_tsv(path):
    rows = []
    with open(path, encoding="utf-16") as f:
        columns = f.readline().strip().split("\t")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == len(columns) + 1 and fields[6] == "":
                fields.pop(6)
            assert len(fields) == len(columns), line
            rows.append(fields)
    return pd.DataFrame(rows, columns=columns)


class VoxAngelesDataset(BaseDataset):
    """
    Valid splits are "all" or any ISO 639-3 language code included in the
    speech31/voxangeles Hugging Face dataset, for example "abk" or "ace".
    """

    def __init__(
        self,
        split="all",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        force_offline=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        self.dataset = load_dataset(
            HF_DATASET,
            split=HF_SPLIT,
            cache_dir=DATA_DIR,
            download_config=DownloadConfig(local_files_only=force_offline),
        )
        if split != "all":
            indices = [
                i for i, lang in enumerate(self.dataset["lang"]) if lang == split
            ]
            self.dataset = self.dataset.select(indices)
        assert len(self.dataset) > 0, f"Split {split} not found in {HF_DATASET}"

        self.timestamped_phonemes_by_file = {}
        if include_timestamps:
            assert os.path.exists(DURATIONS_TSV), (
                "Phone timestamps require "
                + DURATIONS_TSV
                + ". Download it from "
                + "https://raw.githubusercontent.com/pacscilab/voxangeles/main/"
                + "data/phonetic_measurements/voxangeles_durations.tsv"
            )
            durations = read_durations_tsv(DURATIONS_TSV)
            if split != "all":
                durations = durations[durations["lang"] == split]

            for file_id, rows in durations.groupby("file", sort=False):
                # pstart/pend and wstart/wend are in seconds. The HF mirror provides
                # word-level audio, so timestamps are made relative to the word start.
                word_start = float(rows["wstart"].min())
                self.timestamped_phonemes_by_file[file_id] = [
                    (
                        row["phone"],
                        int((float(row["pstart"]) - word_start) * TARGET_SAMPLE_RATE),
                        int((float(row["pend"]) - word_start) * TARGET_SAMPLE_RATE),
                    )
                    for _, row in rows.iterrows()
                    if isinstance(row["phone"], str) and row["phone"].strip()
                ]

        self.vocab = {
            phone
            for phones in self.dataset["phn"]
            for phone in str(phones).split()
            if phone.strip()
        }

    def __len__(self):
        return len(self.dataset)

    def _get_ix(self, ix):
        row = self.dataset[ix]
        audio = row["audio"]
        audio_array = np.asarray(audio["array"])
        if audio_array.ndim == 2:
            audio_array = np.mean(audio_array, axis=1)
        if audio_array.dtype.kind == "f":
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
        else:
            audio_array = audio_array.astype(np.int16)
        audio_array = audio_resample(
            audio_array, audio["sampling_rate"], TARGET_SAMPLE_RATE
        )

        ipa = "".join(str(row["phn"]).split())
        outputs = [ipa, audio_array]
        if self.include_timestamps:
            outputs.append(self.timestamped_phonemes_by_file.get(row["file"]))
        if self.include_speaker_info:
            outputs.append(
                {
                    "file": row["file"],
                    "language": row["lang"],
                    "source": HF_DATASET,
                }
            )
        if self.include_text:
            outputs.append(row["word"])
        return tuple(outputs)


if __name__ == "__main__":
    dataset = VoxAngelesDataset(
        split="all",
        include_timestamps=os.path.exists(DURATIONS_TSV),
        include_speaker_info=True,
        include_text=True,
    )
    print(len(dataset))
    interactive_flag_samples(dataset)
