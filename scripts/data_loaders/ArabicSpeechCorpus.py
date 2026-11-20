# Arabic Speech Corpus contains about 4.1 hours of studio speech from one male
# speaker reading Modern Standard Arabic with a Damascus accent. The material
# includes news-style text and nonsense utterances for diphone coverage.
# The 100 test utterances were recorded separately.
#
# Pronunciation rules generated the phone labels from diacritized text. HTK
# aligned them. About 10% was hand-aligned for the second alignment pass.
# Two experts aligned speech and a third reviewed disagreements.
# Source: https://eprints.soton.ac.uk/409695/ (Halabi, 2016, §§ 2.3.3, 5.1–5.2, 6.4)
#
# The publisher's site lists CC BY 4.0. The archive README names MicroLinkPC
# as the owner, grants free research use, and requires an agreement for commercial use.
# https://en.arabicspeechcorpus.com/
#
# Download arabic-speech-corpus.zip from the publisher's site and keep it zipped
# in .data/ArabicSpeechCorpus/arabic-speech-corpus.zip.

import csv
import io
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import soundfile
import textgrids

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE, audio_resample
from data_loaders.common import BaseDataset, interactive_flag_samples

SITE = "https://en.arabicspeechcorpus.com/"
DATA_ZIP = (
    Path(__file__).resolve().parents[2]
    / ".data"
    / "ArabicSpeechCorpus"
    / "arabic-speech-corpus.zip"
)
FOLDERS = {"train": "arabic-speech-corpus/", "test": "arabic-speech-corpus/test set/"}
LOGGER = logging.getLogger(__name__)

# Halabi (2016), Table 4-7 and § 5.1, and Halabi and Wald (2016), Table 1.
# https://aclanthology.org/L16-1116/
# Section 5.1 identifies u0 as short /u/. AH and Ah denote hamza in the
# pronunciation examples and matching transcript contexts.
CONSONANTS = {phone: phone for phone in "btdrzsfqklmnhwpv"} | {
    "<": "ʔ",
    "AH": "ʔ",
    "Ah": "ʔ",
    "^": "θ",
    "j": "ʒ",
    "J": "d͡ʒ",
    "H": "ħ",
    "x": "x",
    "*": "ð",
    "$": "ʃ",
    "S": "sˤ",
    "D": "dˤ",
    "T": "tˤ",
    "Z": "ðˤ",
    "E": "ʕ",
    "g": "ɣ",
    "y": "j",
}
# The source writes emphatic vowel contexts with capital i/u symbols.
VOWELS = {
    "a": "a",
    "aa": "æː",
    "A": "ɑ",
    "AA": "ɑː",
    "i0": "i",
    "i1": "ɪ",
    "ii0": "iː",
    "ii1": "ɪː",
    "u": "u",
    "u0": "u",
    "u1": "o̞",
    "uu0": "uː",
    "uu1": "o̞ː",
    "-": "ə",
}
VOWELS.update({key.upper(): ipa for key, ipa in VOWELS.items() if key[-1] in "01"})
# The archive uses dist for damaged speech. G is ambiguous in the foreign words
# where it occurs, and the source tables disagree about the symbol.
UNMAPPED = {"dist", "G"}

# Standard Buckwalter orthographic symbols. The archive uses '-' for pauses.
# https://languagelog.ldc.upenn.edu/myl/ldc/morph/buckwalter.html
BUCKWALTER = str.maketrans(
    "'|>&<}AbptvjHxd*rzs$SDTZEg_fqklmnhwYyFNKaui~o\x60{-",
    "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيًٌٍَُِّْٰٱ ",
)


def phone_to_ipa(label):
    if label == "sil":
        return ""
    # Apostrophes on vowels mark stress in the source label.
    vowel = VOWELS.get(label.removesuffix("'"))
    if vowel is not None:
        return vowel
    if label in CONSONANTS:
        return CONSONANTS[label]
    # Doubled consonant symbols denote gemination in the TextGrids.
    half = label[: len(label) // 2]
    if half in CONSONANTS and label == half * 2:
        return CONSONANTS[half] + "ː"
    raise ValueError(f"Unknown Arabic Speech Corpus phone {label!r}")


class ArabicSpeechCorpusDataset(BaseDataset):
    """Load the original train or test split, or both with split="all".

    Audio is mono 16 kHz int16. Phone timings are sample offsets from the start
    of the clip. IPA and timings use the same segment labels. Speaker metadata
    retains the source stress marks and Buckwalter text. Optional text is in
    Arabic script.
    """

    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        *,
        archive=DATA_ZIP,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)
        if split not in (*FOLDERS, "all"):
            raise ValueError(f"Unknown Arabic Speech Corpus split {split!r}")

        self.archive = Path(archive)
        if not self.archive.is_file():
            raise FileNotFoundError(
                f"Download {SITE}arabic-speech-corpus.zip to {self.archive}"
            )
        self.items = []
        self.skipped = Counter()
        with ZipFile(self.archive) as bundle:
            members = set(bundle.namelist())
            for part in FOLDERS if split == "all" else (split,):
                self._read_split(part, bundle, members)
        self.vocab = {
            phone
            for _, _, phones, _, _ in self.items
            for phone, _, _ in phones
            if phone
        }
        if self.skipped:
            LOGGER.warning(
                "Arabic Speech Corpus skipped clips with labels %s", dict(self.skipped)
            )

    def _read_split(self, split, bundle, members):
        folder = FOLDERS[split]
        rows = bundle.read(folder + "orthographic-transcript.txt").decode("utf-8-sig")
        transcripts = dict(
            csv.reader(io.StringIO(rows), delimiter=" ", skipinitialspace=True)
        )
        wavs = {
            name.removeprefix(folder + "wav/")
            for name in members
            if name.startswith(folder + "wav/") and name.endswith(".wav")
        }
        if not wavs or wavs != transcripts.keys():
            raise ValueError(f"WAV filenames and transcript IDs differ in {folder}")

        for name in sorted(wavs):
            stem = name.removesuffix(".wav")
            grid_name = f"{folder}textgrid/{stem}.TextGrid"
            grid = textgrids.TextGrid()
            grid.parse(bundle.read(grid_name).removeprefix(b"\xef\xbb\xbf"))
            intervals = grid["phones"]
            labels = [interval.text.strip() for interval in intervals]
            skipped = UNMAPPED.intersection(labels)
            if skipped:
                self.skipped.update(skipped)
                continue
            try:
                phones = [
                    (
                        phone_to_ipa(label),
                        round(interval.xmin * TARGET_SAMPLE_RATE),
                        round(interval.xmax * TARGET_SAMPLE_RATE),
                    )
                    for label, interval in zip(labels, intervals)
                ]
            except ValueError as error:
                raise ValueError(
                    f"{grid_name} has an invalid phone tier: {error}"
                ) from error
            if not any(phone for phone, _, _ in phones):
                self.skipped["no phones"] += 1
                continue
            if any(not (0 <= start < end) for _, start, end in phones):
                raise ValueError(f"Invalid phone timings in {grid_name}")
            self.items.append(
                (split, stem, phones, " ".join(labels), transcripts[name])
            )

    def __len__(self):
        return len(self.items)

    def _get_ix(self, ix):
        split, stem, phones, raw_label, text = self.items[ix]
        wav = f"{FOLDERS[split]}wav/{stem}.wav"
        with ZipFile(self.archive) as bundle, bundle.open(wav) as file:
            audio, rate = soundfile.read(file, dtype="int16")
        if audio.ndim != 1 or not audio.size:
            raise ValueError(f"Expected a nonempty mono WAV in {wav}")
        audio = audio_resample(audio, rate)
        if max(end for _, _, end in phones) > len(audio):
            raise ValueError(f"Phone timings extend past the end of {wav}")

        result = ["".join(phone for phone, _, _ in phones), audio]
        if self.include_timestamps:
            result.append(list(phones))
        if self.include_speaker_info:
            result.append(
                {
                    "id": f"{split}/{stem}",
                    "split": split,
                    "speaker": "single male speaker, Damascus accent",
                    "language": "Modern Standard Arabic",
                    "source": SITE,
                    "raw_label": raw_label,
                    "buckwalter_text": text,
                }
            )
        if self.include_text:
            result.append(" ".join(text.translate(BUCKWALTER).split()))
        return tuple(result)


if __name__ == "__main__":
    dataset = ArabicSpeechCorpusDataset(include_speaker_info=True, include_text=True)
    print(len(dataset))
    interactive_flag_samples(dataset)
