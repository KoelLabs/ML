# AISHELL-3 contains about 85 hours of scripted, neutral Mandarin from 218 native
# speakers, mostly with northern or southern accents. Recordings are quiet indoor,
# mono 44.1 kHz PCM. Human annotators transcribed pinyin and tones from the readings,
# including tone changes. The paper gives no annotator count or agreement figures.
# The loader converts pinyin deterministically to broad IPA. There are no phone times.
# Shi et al. (Interspeech 2021): https://doi.org/10.21437/INTERSPEECH.2021-755
#
# AISHELL and OpenSLR list Apache 2.0 for this release. AISHELL's separate membership
# portal describes research-only access.
# https://www.aishelltech.com/aishell_3
# https://www.openslr.org/93/
# https://opendata.aishelltech.com/aishell-3
#
# Extract data_aishell3.tgz from OpenSLR into .data/AISHELL3 so that spk-info.txt,
# train/ and test/ are directly under that directory.

import re
import sys
import wave
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.audio import TARGET_SAMPLE_RATE, audio_resample
from core.codes import TONE2IPA
from data_loaders.common import BaseDataset, interactive_flag_samples

SITE = "https://www.openslr.org/93/"
DATA_DIR = Path(__file__).resolve().parents[2] / ".data" / "AISHELL3"
SPLITS = ("train", "test", "test_new_speakers", "all")
SYLLABLE = re.compile(r"([a-z]+)([1-5])")

# Broad initials and finals follow the University of Washington Mandarin IPA tables.
# https://courses.washington.edu/chin342/ipa/consonants.html
# https://courses.washington.edu/chin342/ipa/finals.html
# Spaces delimit phones for vocab, including tied affricates and syllabic consonants.
INITIALS = {
    "b": "p",
    "p": "pʰ",
    "m": "m",
    "f": "f",
    "d": "t",
    "t": "tʰ",
    "n": "n",
    "l": "l",
    "g": "k",
    "k": "kʰ",
    "h": "x",
    "j": "t͡ɕ",
    "q": "t͡ɕʰ",
    "x": "ɕ",
    "zh": "ʈ͡ʂ",
    "ch": "ʈ͡ʂʰ",
    "sh": "ʂ",
    "r": "ɹ",
    "z": "t͡s",
    "c": "t͡sʰ",
    "s": "s",
}
FINALS = {
    "a": "a",
    "o": "o",
    "e": "ɤ",
    "ai": "a i",
    "ei": "e i",
    "ao": "a u",
    "ou": "o u",
    "an": "a n",
    "en": "ə n",
    "ang": "a ŋ",
    "eng": "ə ŋ",
    "er": "ə ɹ",
    "i": "i",
    "ia": "i a",
    "ie": "i ɛ",
    "iao": "i a u",
    "iou": "i o u",
    "ian": "i ɛ n",
    "in": "i n",
    "iang": "i a ŋ",
    "ing": "i ŋ",
    "iong": "y u ŋ",
    "u": "u",
    "ua": "u a",
    "uo": "u ɔ",
    "uai": "u a i",
    "uei": "u e i",
    "uan": "u a n",
    "uen": "u ə n",
    "uang": "u a ŋ",
    "ong": "u ŋ",
    "ueng": "u ə ŋ",
    "v": "y",
    "ve": "y ɛ",
    "van": "y ɛ n",
    "vn": "y n",
}
SHORT_FINALS = {"iu": "iou", "ui": "uei", "un": "uen"}


@lru_cache
def pinyin_phones(base):
    # Erhua is represented as the base syllable plus a rhotic suffix. Its effect
    # on the rhyme varies by accent. Standalone "er" already has a rhotic final.
    # https://doi.org/10.1515/phon-2022-0036
    if base.endswith("r") and base != "er":
        return (*pinyin_phones(base[:-1]), "ɹ")
    if base == "yo":
        return ("j", "o")

    initial = base[:2] if base[:2] in ("zh", "ch", "sh") else base[:1]
    if initial in INITIALS:
        final = base[len(initial) :]
    else:
        initial, final = "", base
    if initial in ("j", "q", "x"):
        final = final.replace("u", "v", 1) if final.startswith("u") else final
    elif final.startswith("yu"):
        final = "v" + final[2:]
    elif final.startswith("y"):
        final = final[1:] if final.startswith("yi") else "i" + final[1:]
    elif final.startswith("w"):
        final = final[1:] if final.startswith("wu") else "u" + final[1:]
    final = SHORT_FINALS.get(final, final)

    # The unrounded nucleus in zi/ci/si and zhi/chi/shi/ri is syllabic.
    # https://doi.org/10.1017/S0025100303001208
    if final == "i" and initial in ("z", "c", "s"):
        rhyme = "ɹ̩"
    elif final == "i" and initial in ("zh", "ch", "sh", "r"):
        rhyme = "ɻ̩"
    elif final == "o" and initial in ("b", "p", "m", "f"):
        rhyme = FINALS["uo"]
    else:
        rhyme = FINALS[final]
    onset = (INITIALS[initial],) if initial else ()
    return (*onset, *rhyme.split())


def read_content(path):
    """Read the utterance IDs and alternating characters and pinyin from content.txt."""
    utterances = {}
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            name, _, annotation = line.partition("\t")
            tokens = annotation.split()
            if not name.endswith(".wav") or not tokens or len(tokens) % 2:
                raise ValueError(f"Invalid AISHELL-3 transcript at {path}:{number}")
            stem = name[:-4]
            if stem in utterances:
                raise ValueError(f"Duplicate utterance {stem} at {path}:{number}")
            utterances[stem] = ("".join(tokens[::2]), tokens[1::2])
    return utterances


def read_speakers(path):
    speakers = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                speaker, age, gender, accent = (
                    field.strip() for field in line.split("\t")
                )
                speakers[speaker] = (age, gender, accent)
    return speakers


def read_audio(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth()) != (1, 2):
            raise ValueError(f"Expected mono 16-bit PCM in {path}")
        frames, rate = wav.getnframes(), wav.getframerate()
        data = wav.readframes(frames)
    # The WAV data chunk declares the frame count. The outer RIFF size in many
    # AISHELL-3 files overstates the file size.
    if not frames or len(data) != frames * 2:
        raise ValueError(f"Empty or truncated audio in {path}")
    return audio_resample(np.frombuffer(data, dtype="<i2"), rate, TARGET_SAMPLE_RATE)


class AISHELL3Dataset(BaseDataset):
    """Read the archive's train/test folders, or both with split="all".

    The official test folder shares speakers with train. Use "test_new_speakers"
    to evaluate on the 44 test speakers absent from train. Set tones="letters"
    to include Chao letters for the four full tones. Neutral tones are unmarked
    in IPA and kept as tone 5 in the original pinyin in speaker metadata's raw_label.
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
            raise ValueError(
                f"Unknown AISHELL-3 split {split!r}. Expected one of {SPLITS}"
            )
        if tones not in ("none", "letters"):
            raise ValueError(
                f"Unknown tones mode {tones!r}. Expected 'none' or 'letters'"
            )
        if include_timestamps:
            raise ValueError("AISHELL-3 does not provide phone timestamps")

        self.data_dir = Path(data_dir)
        self.speakers = (
            read_speakers(self.data_dir / "spk-info.txt")
            if include_speaker_info
            else None
        )
        self.items = []
        self.vocab = set()
        self.skipped = Counter()

        folders = ("train", "test") if split == "all" else (split,)
        train_speakers = set()
        if split == "test_new_speakers":
            train = read_content(self.data_dir / "train" / "content.txt")
            train_speakers = {stem[:7] for stem in train}
            folders = ("test",)

        for folder in folders:
            content = read_content(self.data_dir / folder / "content.txt")
            for stem, (text, pinyin) in sorted(content.items()):
                if stem[:7] in train_speakers:
                    continue
                phones = []
                for syllable in pinyin:
                    match = SYLLABLE.fullmatch(syllable)
                    if match is None:
                        self.skipped[syllable] += 1
                        break
                    base, tone = match.groups()
                    try:
                        phones.extend(pinyin_phones(base))
                    except KeyError:
                        self.skipped[syllable] += 1
                        break
                    if tones == "letters" and (letter := TONE2IPA["cmn"][tone]):
                        phones.append(letter)
                else:
                    self.vocab.update(phones)
                    self.items.append(
                        (folder, stem, "".join(phones), text, " ".join(pinyin))
                    )

        if self.skipped:
            print(
                f"AISHELL-3 {split}: skipped utterances with unsupported pinyin {dict(self.skipped)}"
            )

    def __len__(self):
        return len(self.items)

    def _get_ix(self, ix):
        folder, stem, ipa, text, pinyin = self.items[ix]
        speaker = stem[:7]
        path = self.data_dir / folder / "wav" / speaker / f"{stem}.wav"
        outputs = [ipa, read_audio(path)]
        if self.include_speaker_info:
            age, gender, accent = self.speakers[speaker]
            outputs.append(
                {
                    "id": stem,
                    "split": folder,
                    "speaker": speaker,
                    "age_group": age,
                    "gender": gender,
                    "accent": accent,
                    "language": "Mandarin",
                    "language_code": "cmn",
                    "glottocode": "mand1415",
                    "source": SITE,
                    "raw_label": pinyin,
                }
            )
        if self.include_text:
            outputs.append(text)
        return tuple(outputs)


if __name__ == "__main__":
    dataset = AISHELL3Dataset(include_speaker_info=True, include_text=True)
    print(len(dataset))
    interactive_flag_samples(dataset)
