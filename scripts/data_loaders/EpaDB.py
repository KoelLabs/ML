# EpaDB from https://github.com/JazminVidal/EpaDB

import os
import sys

import zipfile
import textgrids

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import audio_bytes_to_array, TARGET_SAMPLE_RATE
from core.codes import epadb2ipa

SOURCE_SAMPLE_RATE = None  # Auto detect because it varies by file

DATA_ZIP = os.path.join(
    os.path.dirname(__file__), "..", "..", ".data", "EpaDB-master.zip"
)


class EpaDBDataset(BaseDataset):
    """
    Valid splits are "train" and "test" which contain scripted utterances spoken by Argentenian spanish speakers
    Two annotations of the dataset are provided (annotation1 and annotation2) when annotations conflict (which are only four samples)
    but only the first annotation is used and we discard the utterances that appear in the second one to remove ambuious annptations.
    """

    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        assert split in ["train", "test"], "Split must be one of 'train' or 'test'"

        self.zip = zipfile.ZipFile(DATA_ZIP, "r")

        self.files = [
            f
            for f in self.zip.namelist()
            if f"/{self.split}/" in f
            and "/annotations_1/" in f
            and f.endswith(".TextGrid")
        ]

    def __del__(self):
        self.zip.close()

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        annotation_path = self.files[ix]
        audio_path = annotation_path.replace("annotations_1", "waveforms").replace(
            ".TextGrid", ".wav"
        )

        # Read wav
        with self.zip.open(audio_path) as wav_file:
            audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)

        # Read annotations from annotator 1
        with self.zip.open(annotation_path) as annotation_file:
            tg = textgrids.TextGrid()
            tg.parse(annotation_file.read())

        phones = tg.interval_tier_to_array("annotation")
        timestamped_phonemes = [
            (
                epadb2ipa(c["label"]),
                int(c["begin"] * TARGET_SAMPLE_RATE),
                int(c["end"] * TARGET_SAMPLE_RATE),
            )
            for c in phones
            if c["label"].lower() not in ["sil", "sp", "spn", "err", ""]
        ]
        ipa = "".join(t[0] for t in timestamped_phonemes)

        outputs = [ipa, audio]
        if self.include_timestamps:
            outputs.append(timestamped_phonemes)
        if self.include_speaker_info:
            outputs.append(
                {"speaker_id": annotation_path.split(os.path.sep)[-1].split("_")[0]}
            )
        if self.include_text:
            words = tg.interval_tier_to_array("words")
            text = " ".join(w["label"] for w in words)
            outputs.append(text)
        return tuple(outputs)


if __name__ == "__main__":
    dataset = EpaDBDataset(
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    )
    print(dataset[0])
