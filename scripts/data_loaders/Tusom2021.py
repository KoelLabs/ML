# Tusom2021, a phonetically transcribed word list from East Tusom, an endangered Tibeto-Burman
# language of Northeast India. One speaker, 2,255 utterances, 51.8 minutes.
# Data: https://github.com/dmort27/tusom2021
# Paper: https://www.isca-archive.org/interspeech_2021/mortensen21_interspeech.html
# License: the MIT License, in the LICENSE file of the data repo.
#
# Download instructions:
#   git clone https://github.com/dmort27/tusom2021 ./.data/Tusom2021
#   git -C ./.data/Tusom2021 checkout e168dfb90f2b03d433742d5ba5ef5e5731686610
#
# Tested on Nov 9, 2026, on the real files at that commit. Every item of train, dev, test and all
# was checked in each of the three label forms. No person has listened to the clips yet.
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
# Added to every error about the data files, so that the error says what fixes it.
FIX = (
    f"From the repo root, get the data with git clone {DATA_REPO} ./.data/Tusom2021"
    " and git -C ./.data/Tusom2021 checkout e168dfb90f2b03d433742d5ba5ef5e5731686610,"
    " or repair a damaged copy with git -C ./.data/Tusom2021 checkout -- data"
)
SPLITS = ("train", "dev", "test", "all")
# The value of `tones` and the field of mapping.yml that it picks.
TONE_FIELDS = {"none": "no_tones", "diacritics": "tone_dias", "letters": "tone_letters"}


def read_labels(name):
    """The entries of one of the dataset's YAML files, by WAV file name."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. {FIX}")
    try:
        with open(path, encoding="utf-8") as f:
            entries = yaml.safe_load(f)
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise ValueError(f"Cannot parse {path}. {e}. {FIX}") from e
    if not isinstance(entries, dict):
        raise ValueError(f"{path} is not a YAML mapping. {FIX}")
    for wav, entry in entries.items():
        for field in ("gloss", *TONE_FIELDS.values()):
            text = entry.get(field) if isinstance(entry, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Entry {wav} of {path} has no text in {field}. {FIX}")
    return entries


def read_wav(path):
    """A WAV file as int16 at 16 kHz."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. {FIX}")
    with open(path, "rb") as f:
        header = f.read(12)
    # soundfile would return a shorter clip from a file that was cut short, without any error.
    # The size field of the RIFF header gives it away. It holds the file size minus 8 bytes.
    if int.from_bytes(header[4:8], "little") + 8 != os.path.getsize(path):
        raise ValueError(f"The WAV file {path} is cut short or is not a WAV. {FIX}")
    # soundfile reports the channels and the sample format, so that they can be checked here.
    try:
        with soundfile.SoundFile(path) as f:
            if f.channels != 1 or f.subtype != "PCM_16" or f.frames == 0:
                raise ValueError(
                    f"The WAV file {path} is not mono 16 bit audio with samples. {FIX}"
                )
            audio, rate = f.read(dtype="int16"), f.samplerate
    except soundfile.SoundFileError as e:
        raise ValueError(f"Cannot decode the WAV file {path}. {e}. {FIX}") from e
    return audio_resample(audio, rate, TARGET_SAMPLE_RATE)


class Tusom2021Dataset(BaseDataset):
    """
    Valid splits are "train", "dev" and "test", from the dataset's own files, or "all".

    The text is the English gloss, since East Tusom has no writing system.

    `tones` picks the form of the label. "none" has no tones, "diacritics" writes them as
    combining marks and "letters" as Chao tone letters. The default is "none" because the
    paper's experiments use that form and evaluate.py drops every tone mark before FER.
    """

    def __init__(
        self,
        split="all",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        tones="none",
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        if include_timestamps:
            # The paper aligned the words by hand in Praat, but the data has no TextGrid file.
            raise ValueError("Tusom2021 has no phone timestamps")
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}. Valid splits are {SPLITS}")
        if tones not in TONE_FIELDS:
            raise ValueError(
                f"Unknown tones {tones}. Valid forms are {list(TONE_FIELDS)}"
            )
        self.tones = tones
        self.label_field = TONE_FIELDS[tones]

        labels = read_labels("mapping.yml")
        names = labels if split == "all" else read_labels(f"{split}.yml")
        for name in names:
            if name not in labels:
                raise ValueError(
                    f"{name} is in {split}.yml but has no label in mapping.yml. {FIX}"
                )
        self.files = sorted(names)
        self.labels = {name: labels[name] for name in self.files}

        found = os.listdir(WAV_DIR) if os.path.isdir(WAV_DIR) else []
        wavs = set(n for n in found if n.endswith(".wav"))
        missing = [name for name in self.files if name not in wavs]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of the {len(self.files)} WAV files of split {split} are not in"
                f" {WAV_DIR}, such as {missing[0]}. {FIX}"
            )
        # The folder also holds WAV files that mapping.yml does not label. They are not used.
        self.skipped = {"wav_without_label": sum(n not in labels for n in wavs)}

        # Counted because the IPA differs from the label in mapping.yml for these items.
        self.cleaned = {
            "label_with_white_space": sum(
                self._ipa(name) != self._raw_label(name) for name in self.files
            )
        }

        feature_table = panphon.FeatureTable()
        self.vocab = {
            phone
            for name in self.files
            for phone in feature_table.segs_safe(self._ipa(name))
        }
        print(
            f"Tusom2021 {split}, tones {tones}: {len(self.files)} items,"
            f" white space removed from {self.cleaned['label_with_white_space']} labels,"
            f" {self.skipped['wav_without_label']} WAV files with no label skipped"
        )

    def _raw_label(self, name):
        return self.labels[name][self.label_field]

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
