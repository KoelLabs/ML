# African American English speech from https://oraal.github.io/coraal

import os
import sys

import re
import tarfile
import gzip
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import audio_bytes_to_array, TARGET_SAMPLE_RATE

SOURCE_SAMPLE_RATE = 44100
DATA_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "CORAAL")
SPLITS = ["ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD"]


class CORAALDataset(BaseDataset):
    """Warning, ipa not implemented yet, provides text transcription instead"""

    def __init__(
        self,
        split="ATL",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        max_samples_per_recording=None,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        if include_timestamps:
            raise NotImplementedError("Timestamps are available but not parsed yet.")

        assert split in SPLITS, f"Split {split} not found in {SPLITS}"

        metadata = pd.read_csv(
            os.path.join(DATA_BASE_DIR, f"{split}-metadata.txt"), sep="\t"
        )

        self.samples = []
        for _, row in metadata.iterrows():
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
                        transcript = transcript[
                            ~transcript["Spkr"].str.contains("_int_")
                            & ~transcript["Content"].str.startswith("(")
                            & ~transcript["Content"].str.contains("/")
                        ]

            for i, (_, trans) in enumerate(transcript.iterrows()):
                text = trans["Content"]
                audio_start = int(trans["StTime"] * TARGET_SAMPLE_RATE)
                audio_end = int(trans["EnTime"] * TARGET_SAMPLE_RATE)

                self.samples.append(
                    (row, subset, part, filename, text, audio_start, audio_end)
                )

                if i + 1 == max_samples_per_recording:
                    break

    def __len__(self):
        return len(self.samples)

    def _get_ix(self, ix):
        row, subset, part, filename, text, audio_start, audio_end = self.samples[ix]

        with gzip.GzipFile(
            os.path.join(DATA_BASE_DIR, "audio", f"{subset}_{part}.tar.gz"), "r"
        ) as gz:
            with tarfile.TarFile(fileobj=gz) as tar:
                with tar.extractfile(filename + ".wav") as wav_file:  # type: ignore
                    audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)
        audio = audio[audio_start:audio_end]

        result = [text, audio]
        if self.include_speaker_info:
            result.append(row.to_dict())
        if self.include_text:
            result[0] = None
            result.append(text)
        return tuple(result)

    def search_transcript(self, query, flags=re.IGNORECASE):
        for i, (_, _, _, _, text, _, _) in enumerate(self.samples):
            match = re.search(query, text, flags)
            if match:
                before_ctx = min(match.start(), 20)
                match_len = match.end() - match.start()
                with_ctx = text[match.start() - before_ctx : match.end() + 20]
                with_ctx = (
                    with_ctx[:before_ctx]
                    + ">>>"
                    + with_ctx[before_ctx : before_ctx + match_len]
                    + "<<<"
                    + with_ctx[before_ctx + match_len :]
                )
                yield i, text[match.start() : match.end()], with_ctx


if __name__ == "__main__":
    data = CORAALDataset(include_text=True)
    print(len(data))

    example = next(data.search_transcript(" aks "))
    print(example)

    print(data[example[0]])
