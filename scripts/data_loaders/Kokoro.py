# Japanese speech from https://github.com/kaiidams/Kokoro-Speech-Dataset

import os
import sys

import zipfile
from misaki import ja

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import audio_bytes_to_wav_array
from forced_alignment.common import group_phonemes

SOURCE_SAMPLE_RATE = 22050
DATA_ZIP = os.path.join(
    os.path.dirname(__file__), "..", "..", ".data", "Kokoro-Speech-Dataset.zip"
)
SUBSTITUTE = {
    "ʦ": "ts",
    "ʨ": "tʃ",
}

g2p = ja.JAG2P()


class KokoroDataset(BaseDataset):
    def __init__(
        self,
        split="meian-by-soseki-natsume",
        include_timestamps=False,
        include_speaker_info=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info)

        if include_timestamps:
            raise ValueError("Timestamps are not available for Kokoro dataset.")

        metadata_zip = zipfile.ZipFile(DATA_ZIP, "r")
        splits = [
            s[: -len(".metadata.txt")]
            for s in metadata_zip.namelist()
            if s.endswith(".metadata.txt")
        ]
        assert split in splits, f"Split {split} not found in {splits}"

        with metadata_zip.open(f"{split}.metadata.txt", "r") as metadata_file:
            self.metadata = metadata_file.read().decode("utf-8").split("\n")
            self.metadata = [
                {
                    "file": x.split("|")[1],  # wav filename
                    "start": x.split("|")[2],
                    "end": x.split("|")[3],
                    "transcription": x.split("|")[4],  # Kanji-kana mixture text
                    "reading": x.split("|")[5],  # Romanized text
                }
                for x in self.metadata
                if len(x.split("|")) > 5
            ]

        metadata_zip.close()

        self.zip = zipfile.ZipFile(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                ".data",
                "Kokoro-Speech-Dataset",
                f"{split}.zip",
            ),
            "r",
        )
        self.files = self.zip.namelist()

    def __del__(self):
        self.zip.close()

    def __len__(self):
        return len(self.metadata)

    def _get_ix(self, ix):
        metadata = self.metadata[ix]
        filename = metadata["file"]
        with self.zip.open(filename) as file:
            audio = audio_bytes_to_wav_array(
                file.read(), format=filename.split(".")[-1]
            )

        transcript = (
            metadata["transcription"]
            .replace(" ", "")
            .replace("。", "")
            .replace("、", "")
            .replace("…", "")
        )
        phonemes, tokens = g2p(transcript)
        phonemes = (
            phonemes.replace(" ", "")
            .replace("“", "")
            .replace("``", "")
            .replace("”", "")
        )
        for key, value in SUBSTITUTE.items():
            phonemes = phonemes.replace(key, value)
        phonemes = group_phonemes(phonemes)
        ipa = "".join(phonemes)

        if self.include_speaker_info:
            speaker = {
                "reader": self.split.split("-by-")[0],
                "author": self.split.split("-by-")[1],
                "transcription": metadata["transcription"],
                "reading": metadata["reading"],
            }
            return ipa, audio, speaker
        else:
            return ipa, audio


if __name__ == "__main__":
    data = KokoroDataset(include_speaker_info=True)

    print(data[0])
