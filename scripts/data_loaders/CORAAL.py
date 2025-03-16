# African American English speech from https://oraal.github.io/coraal

import os
import sys

import re
import tarfile
import gzip
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import audio_bytes_to_array

SOURCE_SAMPLE_RATE = 44100
DATA_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "CORAAL")
SPLITS = ["ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD"]


class CORAALDataset(BaseDataset):
    """Warning, ipa not implemented yet, provides text transcription instead"""

    def __init__(
        self, split="ATL", include_timestamps=False, include_speaker_info=False
    ):
        super().__init__(split, include_timestamps, include_speaker_info)

        if include_timestamps:
            raise NotImplementedError("Timestamps are available but not parsed yet.")

        assert split in SPLITS, f"Split {split} not found in {SPLITS}"

        self.metadata = pd.read_csv(
            os.path.join(DATA_BASE_DIR, f"{split}-metadata.txt"), sep="\t"
        )

    def __len__(self):
        return len(self.metadata)

    def _get_ix(self, ix):
        row = self.metadata.iloc[ix]
        subset = row["CORAAL.Sub"]
        part = int(re.findall(r"\d+", row["Audio.Folder"])[-1])
        filename = os.path.join(".", row["CORAAL.File"])
        assert row["Sampling.Rate"] == f"{SOURCE_SAMPLE_RATE / 1000:.1f} kHz"

        with gzip.GzipFile(
            os.path.join(DATA_BASE_DIR, "transcripts", f"{subset}.tar.gz"), "r"
        ) as gz:
            with tarfile.TarFile(fileobj=gz) as tar:
                with tar.extractfile(filename + ".txt") as txt_file:  # type: ignore
                    transcript = pd.read_csv(txt_file, sep="\t")
                    transcript.set_index("Line", inplace=True)

        with gzip.GzipFile(
            os.path.join(DATA_BASE_DIR, "audio", f"{subset}_{part}.tar.gz"), "r"
        ) as gz:
            with tarfile.TarFile(fileobj=gz) as tar:
                with tar.extractfile(filename + ".wav") as wav_file:  # type: ignore
                    audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)

        if self.include_speaker_info:
            return transcript, audio, row.to_dict()
        else:
            return transcript, audio

    def search_transcript(self, query, flags=re.IGNORECASE):
        for i, row in self.metadata.iterrows():
            subset = row["CORAAL.Sub"]
            filename = os.path.join(".", row["CORAAL.File"])
            assert row["Sampling.Rate"] == f"{SOURCE_SAMPLE_RATE / 1000:.1f} kHz"

            with gzip.GzipFile(
                os.path.join(DATA_BASE_DIR, "transcripts", f"{subset}.tar.gz"), "r"
            ) as gz:
                with tarfile.TarFile(fileobj=gz) as tar:
                    with tar.extractfile(filename + ".txt") as txt_file:  # type: ignore
                        transcript = pd.read_csv(txt_file, sep="\t")
                        transcript.set_index("Line", inplace=True)

                        for line, data in transcript.iterrows():
                            match = re.search(query, data["Content"], flags)
                            if match:
                                before_ctx = min(match.start(), 20)
                                match_len = match.end() - match.start()
                                with_ctx = data["Content"][
                                    match.start() - before_ctx : match.end() + 20
                                ]
                                with_ctx = (
                                    with_ctx[:before_ctx]
                                    + ">>>"
                                    + with_ctx[before_ctx : before_ctx + match_len]
                                    + "<<<"
                                    + with_ctx[before_ctx + match_len :]
                                )
                                yield i, line, data, with_ctx


if __name__ == "__main__":
    data = CORAALDataset(include_speaker_info=True)
    print(len(data))

    example = next(data.search_transcript(" aks "))
    print(example)

    print(data[example[0]])
