# AISHELL-3, Mandarin read speech from 218 speakers, made for speech synthesis. 88,035
# utterances, about 85 hours, with pinyin and tones written by professional annotators.
# Site: https://www.openslr.org/93/
# Paper: Shi, Y., Bu, H., Xu, X., Zhang, S., Li, M. (2021). AISHELL-3: A multi-speaker
#   Mandarin TTS corpus. Interspeech 2021. https://arxiv.org/abs/2010.11567
# License: the sources disagree, Apache 2.0 on OpenSLR and the paper, and the archive has
#   no license line. See the notes, docs/datasets/AISHELL3.md.
#
# The corpus has pinyin, not IPA. The IPA of this loader is made from the pinyin by the
# fixed table in aishell3_pinyin_ipa.json. It is phonemic, and it shows no allophones and
# no accent beyond what the pinyin writes. The archive has no phone times, so there are
# no timestamps.
#
# Download instructions:
#   mkdir -p ./.data/AISHELL3
#   curl -L -o ./.data/AISHELL3/data_aishell3.tgz https://www.openslr.org/resources/93/data_aishell3.tgz
#   tar -xzf ./.data/AISHELL3/data_aishell3.tgz -C ./.data/AISHELL3 --strip-components=1
#   The archive is 19.0 GB. Its size and SHA-256 are in the notes. Afterwards
#   .data/AISHELL3/train and .data/AISHELL3/test each hold content.txt and wav/.
#
# Tested on Nov 11, 2026, on every item of every split, with the real audio.
# No person has listened to the clips yet.
# TODO: manual verification

import io
import json
import os
import re
import sys

import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_resample, TARGET_SAMPLE_RATE

SITE = "https://www.openslr.org/93/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "AISHELL3")
TABLE = os.path.join(os.path.dirname(__file__), "aishell3_pinyin_ipa.json")
# Added to every error about the data, so that the error says what fixes it.
FIX = f"Download {SITE} to .data/AISHELL3/ and unpack it, as the header of AISHELL3.py says"
FOLDERS = ("train", "test")
SPLITS = ("train", "test", "test_new_speakers", "all")
TONES = ("none", "letters")
SYLLABLE = re.compile(r"[a-z]+[1-5]")


def syllable_units(base, table):
    """The IPA units of one pinyin syllable without its tone, or a reason string."""
    if base in table["skip_syllables"]:
        return f"nonstandard:{base}"
    # A final r is erhua, such as dianr. er itself is a syllable.
    if base.endswith("r") and base != "er":
        return "erhua"
    initial = next((i for i in ("zh", "ch", "sh") if base.startswith(i)), "")
    if base[:1] in "yw":
        rest = base[1:]
        if base[0] == "y":
            # yu is ü, yi is i, and every other y is i before the final.
            rest = (
                "v" + rest[1:]
                if rest[:1] == "u"
                else rest if rest[:1] == "i" else "i" + rest
            )
        elif rest[:1] != "u":
            rest = "u" + rest
        final = rest
    else:
        initial = initial or (base[:1] if base[:1] in table["initials"] else "")
        final = base[len(initial) :]
        # ü is written u after j, q and x. AISHELL-3 writes ü as v.
        if initial in ("j", "q", "x") and final[:1] == "u":
            final = "v" + final[1:]
    if final == "i" and initial in ("z", "c", "s", "zh", "ch", "sh", "r"):
        kind = "dental" if initial in ("z", "c", "s") else "retroflex"
        return [table["initials"][initial]["ipa"]] + table["apical"][kind]["units"]
    if final not in table["finals"]:
        return f"unknown_syllable:{base}"
    onset = [table["initials"][initial]["ipa"]] if initial else []
    return onset + table["finals"][final]["units"]


def pinyin_to_ipa(syllables, tones, table):
    """(IPA string, list of IPA units) of a list of pinyin syllables, or a reason."""
    units, previous = [], None
    for syllable in syllables:
        if not SYLLABLE.fullmatch(syllable):
            return f"bad_syllable:{syllable}"
        base, tone = syllable[:-1], syllable[-1]
        part = syllable_units(base, table)
        if isinstance(part, str):
            return part
        units += part
        if tones == "letters":
            letter = table["tones"].get(tone)
            if letter is None:
                # The neutral tone takes its height from the tone before it.
                letter = table["tones"][
                    "neutral_after_4" if previous == "4" else "neutral_other"
                ]
            units.append(letter)
        previous = tone
    return "".join(units), units


def read_content(folder):
    """(file stem, characters, pinyin syllables) of each line of one content.txt."""
    path = os.path.join(DATA_DIR, folder, "content.txt")
    try:
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{path} not found. {FIX}") from e
    lines = []
    # The file has Windows line endings, and 11 lines end with a space.
    for line in re.split(r"\r?\n", text):
        if not line.strip():
            continue
        name, sep, rest = line.strip().partition("\t")
        tokens = rest.split()
        if not (sep and name.endswith(".wav")) or len(tokens) % 2 or len(tokens) == 0:
            raise ValueError(f"Cannot read the line {line!r} of {path}. {FIX}")
        lines.append((name[:-4], "".join(tokens[0::2]), tokens[1::2]))
    return lines


def read_speakers():
    """The (age group, gender, accent) of each speaker, from spk-info.txt."""
    path = os.path.join(DATA_DIR, "spk-info.txt")
    try:
        with open(path, encoding="utf-8") as f:
            rows = [
                r.split("\t")
                for r in f.read().splitlines()
                if r.strip()[:1] != "#" and r.strip()
            ]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{path} not found. {FIX}") from e
    if any(len(r) != 4 for r in rows):
        raise ValueError(f"A line of {path} does not have 4 fields. {FIX}")
    # One gender is written "female " with a space.
    return {r[0]: tuple(x.strip() for x in r[1:]) for r in rows}


def wav_is_whole(data):
    """True if the data chunk of a WAV ends exactly at the end of the file."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    pos = 12
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if data[pos : pos + 4] == b"data":
            return pos + 8 + size in (len(data), len(data) - 1)
        pos += 8 + size + (size & 1)
    return False


def read_wav(path):
    """A WAV file as int16 at 16 kHz."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{path} not found. {FIX}") from e
    # soundfile returns a short clip from a cut file with no error. The size of the data chunk
    # gives such a file away. The RIFF size field is not used, since some files have a wrong one.
    if not wav_is_whole(data):
        raise ValueError(f"The WAV file {path} is cut short or is not a WAV. {FIX}")
    try:
        with soundfile.SoundFile(io.BytesIO(data)) as f:
            if f.channels != 1 or f.subtype != "PCM_16":
                raise ValueError(f"The WAV file {path} is not mono 16 bit PCM. {FIX}")
            audio, rate = f.read(dtype="int16"), f.samplerate
    except soundfile.SoundFileError as e:
        raise ValueError(f"Cannot decode the WAV file {path}. {e}. {FIX}") from e
    if len(audio) == 0:
        raise ValueError(f"The WAV file {path} has no samples. {FIX}")
    return audio_resample(audio, rate, TARGET_SAMPLE_RATE)


class AISHELL3Dataset(BaseDataset):
    """
    Valid splits are "train" and "test", the two folders of the archive, "all", which is
    train then test, and "test_new_speakers", the test utterances of speakers that never
    speak in train. The archive's test folder is not speaker disjoint from train, 170 of
    its 214 speakers are in train too. Items are in file name order within a split, and
    no split uses randomness.

    The IPA comes from the pinyin of content.txt and the table in aishell3_pinyin_ipa.json.
    tones="none" leaves the tones out, as the evaluation does. tones="letters" writes Chao
    tone letters after each syllable. An utterance with erhua, such as dianr, or with a
    syllable the table does not know is skipped and counted in self.skipped.
    There are no timestamps, because the archive has no phone times. The text is the
    characters of content.txt, and raw_label of the speaker info is the pinyin with tones.
    """

    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        tones="none",
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}. Valid splits are {SPLITS}")
        if tones not in TONES:
            raise ValueError(f"Unknown tones {tones}. Valid values are {TONES}")
        if include_timestamps:
            raise ValueError("Timestamps are not available for AISHELL-3, it has none.")
        with open(TABLE, encoding="utf-8") as f:
            table = json.load(f)
        self.tones, self.speakers = tones, read_speakers()
        self.items, self.skipped = [], {}
        parts = {"all": FOLDERS, "test_new_speakers": ("test",)}.get(split, (split,))
        content = {part: read_content(part) for part in parts}
        if split == "test_new_speakers":
            train = {stem[:7] for stem, _, _ in read_content("train")}
            content["test"] = [x for x in content["test"] if x[0][:7] not in train]
        for part in parts:
            on_disk = {}
            for stem, characters, syllables in sorted(content[part]):
                speaker = stem[:7]
                if speaker not in on_disk:
                    folder = os.path.join(DATA_DIR, part, "wav", speaker)
                    if not os.path.isdir(folder):
                        raise ValueError(f"{folder} not found. {FIX}")
                    on_disk[speaker] = set(os.listdir(folder))
                if f"{stem}.wav" not in on_disk[speaker]:
                    raise ValueError(f"{stem}.wav of {part} not found. {FIX}")
                if speaker not in self.speakers:
                    raise ValueError(f"Speaker {speaker} is not in spk-info.txt. {FIX}")
                result = pinyin_to_ipa(syllables, tones, table)
                if isinstance(result, str):
                    self.skipped[result] = self.skipped.get(result, 0) + 1
                    continue
                self.items.append(
                    {
                        "ipa": result[0],
                        "units": result[1],
                        "split": part,
                        "stem": stem,
                        "text": characters,
                        "pinyin": " ".join(syllables),
                    }
                )
        self.vocab = {u for item in self.items for u in item["units"]}
        for item in self.items:
            del item["units"]
        skipped = ", ".join(f"{k} {v}" for k, v in sorted(self.skipped.items()))
        print(
            f"AISHELL-3 {split}: {len(self.items)} items, skipped {skipped or 'none'}"
        )

    def __len__(self):
        return len(self.items)

    def _get_ix(self, ix):
        item = self.items[ix]
        speaker = item["stem"][:7]
        wav = os.path.join(
            DATA_DIR, item["split"], "wav", speaker, item["stem"] + ".wav"
        )
        outputs = [item["ipa"], read_wav(wav)]
        if self.include_speaker_info:
            age, gender, accent = self.speakers[speaker]
            outputs.append(
                {
                    "id": item["stem"],
                    "split": item["split"],
                    "speaker": speaker,
                    "age_group": age,
                    "gender": gender,
                    "accent": accent,
                    "language": "Mandarin",
                    "source": SITE,
                    "raw_label": item["pinyin"],
                }
            )
        if self.include_text:
            outputs.append(item["text"])
        return tuple(outputs)


if __name__ == "__main__":
    dataset = AISHELL3Dataset(include_speaker_info=True, include_text=True)
    print(len(dataset))
    interactive_flag_samples(dataset)
