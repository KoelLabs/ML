# Tusom2021: a phonetically transcribed word list from East Tusom, an endangered Tibeto-Burman
# language of Northeast India. One speaker, 2,255 utterances, 51.8 minutes.
# Data: https://github.com/dmort27/tusom2021
# Paper: https://www.isca-archive.org/interspeech_2021/mortensen21_interspeech.html
# License: the MIT License, in the LICENSE file of the data repo.
#
# Download instructions:
#   git clone https://github.com/dmort27/tusom2021 ./.data/Tusom2021
#   git -C ./.data/Tusom2021 checkout e168dfb90f2b03d433742d5ba5ef5e5731686610
#
# TODO: manual verification

import os
import sys

import panphon
import soundfile
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_resample, TARGET_SAMPLE_RATE

DATA_REPO = "https://github.com/dmort27/tusom2021"
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", ".data", "Tusom2021", "data"
)
WAV_DIR = os.path.join(DATA_DIR, "wav")
SPLITS = ("train", "dev", "test", "all")
LABEL_FIELD = "no_tones"


def read_labels(name):
    """The entries of one of the dataset's YAML files, by WAV file name."""
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_wav(path):
    """A WAV file as int16 at 16 kHz."""
    audio, rate = soundfile.read(path, dtype="int16")
    return audio_resample(audio, rate, TARGET_SAMPLE_RATE)


class Tusom2021Dataset(BaseDataset):
    """
    Valid splits are "train", "dev" and "test", from the dataset's own files, or "all".

    The text is the English gloss, since East Tusom has no writing system.
    """

    def __init__(
        self,
        split="all",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        # The paper aligned the words by hand in Praat TextGrids, but the data has no TextGrid file.
        assert not include_timestamps, "Tusom2021 has no phone timestamps"

        labels = read_labels("mapping.yml")
        names = labels if split == "all" else read_labels(f"{split}.yml")
        self.files = sorted(names)
        self.labels = {name: labels[name] for name in self.files}

        # The folder also holds WAV files that mapping.yml does not label. They are not used.
        wavs = [n for n in os.listdir(WAV_DIR) if n.endswith(".wav")]
        self.skipped = {"wav_without_label": sum(n not in labels for n in wavs)}

        feature_table = panphon.FeatureTable()
        self.vocab = {
            phone
            for name in self.files
            for phone in feature_table.segs_safe(self._ipa(name))
        }
        print(
            f"Tusom2021 {split}: {len(self.files)} items, {self.skipped['wav_without_label']}"
            " WAV files with no label skipped"
        )

    def _raw_label(self, name):
        return self.labels[name][LABEL_FIELD]

    def _ipa(self, name):
        # White space separates words in some labels and is no phone. VoxAngeles removes it the
        # same way. str.split also takes out the control character U+001D that is in 6 labels.
        return "".join(self._raw_label(name).split())

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        name = self.files[ix]
        outputs = [self._ipa(name), read_wav(os.path.join(WAV_DIR, name))]
        if self.include_speaker_info:
            outputs.append(
                {
                    "id": name.removesuffix(".wav"),
                    "recording": name.rsplit("-", 1)[0],
                    "speaker": "single female native speaker",
                    "language": "East Tusom",
                    "source": DATA_REPO,
                    "raw_label": self._raw_label(name),
                }
            )
        if self.include_text:
            outputs.append(self.labels[name]["gloss"])
        return tuple(outputs)


if __name__ == "__main__":
    dataset = Tusom2021Dataset(include_speaker_info=True, include_text=True)
    print(len(dataset))
    interactive_flag_samples(dataset)
