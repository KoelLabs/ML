# Tusom2021 contains 2,255 labeled clips (51.8 min) of elicited words and short
# phrases from one native speaker of East Tusom, a Tangkhulic variety of Manipur, India.
# The 2004 recordings used a lavalier microphone and contain background noise.
# One linguist transcribed the words and a second reviewed the labels. The
# authors aligned the clips manually in Praat. The paper reports no agreement score.
# The released labels are clip-level IPA with optional tone marks.
# Paper and citation: https://www.isca-archive.org/interspeech_2021/mortensen21_interspeech.html
#
# The upstream dataset and its audio mirror are MIT-licensed.
# https://github.com/dmort27/tusom2021/blob/e168dfb90f2b03d433742d5ba5ef5e5731686610/LICENSE
# https://huggingface.co/datasets/changelinglab/tusom2021-pr
#
# Download the original split files and WAVs:
#   git clone https://github.com/dmort27/tusom2021 .data/Tusom2021
#   git -C .data/Tusom2021 checkout e168dfb90f2b03d433742d5ba5ef5e5731686610

import os
import sys
from pathlib import Path
from unicodedata import normalize

import numpy as np
import panphon
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_file_to_array
from data_loaders.common import BaseDataset, interactive_flag_samples

SOURCE = "https://github.com/dmort27/tusom2021"
DATA_DIR = Path(__file__).resolve().parents[2] / ".data" / "Tusom2021" / "data"
SPLITS = ("train", "dev", "test", "all")
TONE_FIELDS = {"none": "no_tones", "diacritics": "tone_dias", "letters": "tone_letters"}
TONE_DIACRITICS = str.maketrans("", "", "\u0300\u0301\u0302")


def read_mapping(path):
    try:
        with path.open(encoding="utf-8") as file:
            mapping = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing {path}. Download Tusom2021 from {SOURCE}"
        ) from error
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return mapping


class Tusom2021Dataset(BaseDataset):
    """Load an upstream split or all labeled clips.

    The default uses the train split and tone-free IPA. The upstream splits
    group tokens by tone-marked label, so tone-free labels can overlap.
    Optional text is the English gloss. The release has no phone timings.
    """

    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        tones="none",
        *,
        data_dir=DATA_DIR,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)
        if split not in SPLITS:
            raise ValueError(f"Unknown Tusom2021 split {split!r}. Expected {SPLITS}")
        if tones not in TONE_FIELDS:
            raise ValueError(
                f"Unknown Tusom2021 tone form {tones!r}. Expected {tuple(TONE_FIELDS)}"
            )
        if include_timestamps:
            raise ValueError("Tusom2021 has no phone timestamps")

        data_dir = Path(data_dir)
        labels = read_mapping(data_dir / "mapping.yml")
        names = labels if split == "all" else read_mapping(data_dir / f"{split}.yml")
        field = TONE_FIELDS[tones]
        self.items = []
        for name in sorted(names):
            label = labels.get(name)
            if not isinstance(label, dict):
                raise ValueError(f"Missing Tusom2021 label for {name} in mapping.yml")
            raw, gloss = label.get(field), label.get("gloss")
            if (
                not isinstance(raw, str)
                or not isinstance(gloss, str)
                or not gloss.strip()
            ):
                raise ValueError(f"Missing {field} label or English gloss for {name}")
            path = data_dir / "wav" / name
            if path.name != name or path.suffix != ".wav":
                raise ValueError(f"Invalid Tusom2021 WAV name {name!r}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing Tusom2021 WAV {path}")

            ipa = "".join(normalize("NFD", raw).split()).replace("g", "ɡ")
            if tones == "none":
                # Some source no_tones labels have precomposed accented vowels.
                ipa = ipa.translate(TONE_DIACRITICS)
            if not ipa:
                raise ValueError(f"Empty Tusom2021 IPA label for {name}")
            self.items.append((path, ipa, raw, gloss))

        features = panphon.FeatureTable()
        self.vocab = {
            phone for _, ipa, _, _ in self.items for phone in features.segs_safe(ipa)
        }

    def __len__(self):
        return len(self.items)

    def _get_ix(self, ix):
        path, ipa, raw, gloss = self.items[ix]
        audio = audio_file_to_array(path)
        if audio.ndim != 1 or audio.dtype != np.int16 or not audio.size:
            raise ValueError(f"Expected mono 16-bit PCM samples in {path}")
        result = [ipa, audio]
        if self.include_speaker_info:
            result.append(
                {
                    "id": path.stem,
                    "recording": path.stem.rsplit("-", 1)[0],
                    "speaker": "single female native speaker",
                    "language": "East Tusom",
                    "source": SOURCE,
                    "license": "MIT",
                    "raw_label": raw,
                }
            )
        if self.include_text:
            result.append(gloss)
        return tuple(result)


if __name__ == "__main__":
    dataset = Tusom2021Dataset(
        split="all", include_speaker_info=True, include_text=True
    )
    print(len(dataset))
    interactive_flag_samples(dataset)
