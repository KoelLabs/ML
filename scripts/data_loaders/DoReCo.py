# DoReCo 2.0 contains over 100 hours across 53 languages and varieties, mostly fieldwork
# narratives and conversations. The Hub export has about 34 hours across 29 languages.
# Its English split has 47 minutes of Southern England speech from two Kent speakers.
# Corpus descriptions and dialect coverage are at https://doreco.huma-num.fr/languages.
#
# Language experts transcribed the recordings. Pronunciations are derived from the
# transcripts. Annotators corrected word boundaries and checked pauses, then MAUS
# aligned the X-SAMPA phones within each word. Four annotators checked 4,125 words
# from six Anal, Resígaro and Vera’a files in a word-level agreement study. The paper
# reports neither a corpus-wide annotator count nor phone-level agreement.
# Alignment and quality control are described at https://aclanthology.org/2020.lrec-1.324/.
#
# Recording conditions vary by corpus. Clips can contain noise, overlapping speakers,
# or speech without phone labels, including disfluencies and code-switching. Audio is
# downmixed to mono and resampled to 16 kHz. Phone offsets index into the returned clip.
#
# Download each language's core annotation ZIP and core WAVs from its DoReCo page.
# Store the ZIP as .data/DoReCo/<glottocode>/doreco_<glottocode>_core_v2.zip and put the
# WAVs (loose or zipped, with their original names) in the same directory.
# The commercial split requires CC BY 4.0 for audio and annotations. Each source
# corpus must be cited separately. Its citation and license are in the bundle's
# dataset-info file and at https://huggingface.co/datasets/KoelLabs/DoReCo.

import logging
import math
import os
import re
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE, audio_resample
from core.codes import xsampa2ipa
from data_loaders.common import BaseDataset, interactive_flag_samples

TOP_DIR = Path(__file__).resolve().parents[2] / ".data" / "DoReCo"
DEFAULT_LANGUAGE = "sout3282"
DEFAULT_SPLIT = "commercial"
CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
LICENSE_STATEMENT = re.compile(
    r"\b(annotation|audio) files are distributed under [^\n]+?\("
    r"https?://creativecommons\.org/licenses/([\w-]+/\d\.\d/)"
)
AUDIO_PADDING = round(0.3 * TARGET_SAMPLE_RATE)
AUDIO_END_TOLERANCE = TARGET_SAMPLE_RATE // 100
LOGGER = logging.getLogger(__name__)


class Corpus(NamedTuple):
    name: str
    doi: str
    annotation_license: str
    audio_license: str


# DoReCo 2.0 catalog. "audio linked" means the source archive sets its own terms.
# fmt: off
CORPORA = {
    "anal1239": Corpus("Anal", "10.34847/nkl.0dbazp8m", "CC BY", "CC BY-NC"),
    "apah1238": Corpus("Yali (Apahapsili)", "10.34847/nkl.9d91nkq2", "CC BY-NC-SA", "CC BY-NC-SA"),
    "arap1274": Corpus("Arapaho", "10.34847/nkl.36f5r1b6", "CC BY", "CC BY"),
    "bain1259": Corpus("Baïnounk Gubëeher", "10.34847/nkl.a332abw8", "CC BY", "CC BY"),
    "beja1238": Corpus("Beja", "10.34847/nkl.edd011t1", "CC BY-NC", "CC BY-NC"),
    "bora1263": Corpus("Bora", "10.34847/nkl.6eaf5laq", "CC BY", "CC BY"),
    "cabe1245": Corpus("Cabécar", "10.34847/nkl.ebc4ra22", "CC BY-NC", "CC BY-NC-ND"),
    "cash1254": Corpus("Cashinahua", "10.34847/nkl.a8f9q2f1", "CC BY", "CC BY"),
    "dolg1241": Corpus("Dolgan", "10.34847/nkl.f09eikq3", "CC BY", "CC BY-NC"),
    "even1259": Corpus("Evenki", "10.34847/nkl.5e0d27cu", "CC BY", "CC BY"),
    "goem1240": Corpus("Goemai", "10.34847/nkl.b93664ml", "CC BY", "audio linked"),
    "goro1270": Corpus("Gorwaa", "10.34847/nkl.a4b4ijj2", "CC BY", "CC BY"),
    "guri1247": Corpus("Gurindji", "10.34847/nkl.ab2d89mx", "CC BY-NC-ND", "audio linked"),
    "hoch1243": Corpus("Hoocąk", "10.34847/nkl.b57f5065", "CC BY-NC-ND", "audio linked"),
    "jeha1242": Corpus("Jahai", "10.34847/nkl.6a71xp0p", "CC BY-NC-SA", "CC BY-NC-ND"),
    "jeju1234": Corpus("Jejuan", "10.34847/nkl.06ebrk38", "CC BY", "CC BY"),
    "kaka1265": Corpus("Kakabe", "10.34847/nkl.d5aeu9t6", "CC BY", "CC BY"),
    "kama1351": Corpus("Kamas", "10.34847/nkl.cdd8177b", "CC BY", "CC BY"),
    "kark1256": Corpus("Tabaq (Karko)", "10.34847/nkl.eea8144j", "CC BY", "audio linked"),
    "komn1238": Corpus("Komnzo", "10.34847/nkl.c5e6dudv", "CC BY", "CC BY"),
    "ligh1234": Corpus("Light Warlpiri", "10.34847/nkl.7452803q", "CC BY-NC", "CC BY-NC"),
    "lowe1385": Corpus("Lower Sorbian", "10.34847/nkl.6c6e4e9k", "CC BY-NC-SA", "audio linked"),
    "movi1243": Corpus("Movima", "10.34847/nkl.da42xf67", "CC BY", "audio linked"),
    "ngal1292": Corpus("Dalabon", "10.34847/nkl.fae299ug", "CC BY", "CC BY"),
    "nisv1234": Corpus("Nisvai", "10.34847/nkl.2801565f", "CC BY-NC-ND", "CC BY-NC-ND"),
    "nngg1234": Corpus("Nǁng", "10.34847/nkl.f6c37fi0", "CC BY", "CC BY"),
    "nort2641": Corpus("Northern Kurdish (Kurmanji)", "10.34847/nkl.ca10ez5t", "CC BY", "CC BY"),
    "nort2875": Corpus("Northern Alta", "10.34847/nkl.efea0b36", "CC BY-NC", "CC BY-NC-ND"),
    "orko1234": Corpus("Fanbyak", "10.34847/nkl.02084446", "CC BY", "CC BY"),
    "pnar1238": Corpus("Pnar", "10.34847/nkl.5ba1062k", "CC BY-NC", "CC BY-NC"),
    "port1286": Corpus("Daakie", "10.34847/nkl.efeav5l9", "CC BY", "CC BY"),
    "resi1247": Corpus("Resígaro", "10.34847/nkl.ffb96lo8", "CC BY", "CC BY"),
    "ruul1235": Corpus("Ruuli", "10.34847/nkl.fde4pp1u", "CC BY", "CC BY"),
    "sadu1234": Corpus("Sadu", "10.34847/nkl.3db4u59d", "CC BY", "CC BY"),
    "sanz1248": Corpus("Sanzhi Dargwa", "10.34847/nkl.81934177", "CC BY", "CC BY"),
    "savo1255": Corpus("Savosavo", "10.34847/nkl.b74d1b33", "CC BY", "CC BY"),
    "sout2856": Corpus("Nafsan (South Efate)", "10.34847/nkl.ba4f760l", "CC BY", "CC BY-NC-SA"),
    "sout3282": Corpus("English (Southern England)", "10.34847/nkl.9c271u5g", "CC BY", "CC BY"),
    "stan1290": Corpus("French (Swiss)", "10.34847/nkl.3520l685", "CC BY-NC-SA", "CC BY-NC-SA"),
    "sumi1235": Corpus("Sümi", "10.34847/nkl.5ad4t01p", "CC BY", "CC BY"),
    "svan1243": Corpus("Svan", "10.34847/nkl.9ba054c3", "CC BY", "CC BY"),
    "taba1259": Corpus("Tabasaran", "10.34847/nkl.ad7f97xr", "CC BY", "CC BY"),
    "teop1238": Corpus("Teop", "10.34847/nkl.9322sdf2", "CC BY", "CC BY"),
    "texi1237": Corpus("Texistepec Popoluca", "10.34847/nkl.c50ck58f", "CC BY", "CC BY"),
    "toto1304": Corpus("Totoli", "10.34847/nkl.c8b6ei29", "CC BY-NC-SA", "audio linked"),
    "trin1278": Corpus("Mojeño Trinitario", "10.34847/nkl.cbc3b4xr", "CC BY", "CC BY"),
    "tsim1256": Corpus("Asimjeeg Datooga", "10.34847/nkl.f77c7m72", "CC BY", "CC BY"),
    "urum1249": Corpus("Urum", "10.34847/nkl.ac166n10", "CC BY-NC", "CC BY-NC-ND"),
    "vera1241": Corpus("Vera'a", "10.34847/nkl.3e2cu8c4", "CC BY", "CC BY"),
    "warl1254": Corpus("Warlpiri", "10.34847/nkl.042dv614", "CC BY-NC", "CC BY-NC"),
    "yong1270": Corpus("Yongning Na", "10.34847/nkl.abe65p95", "CC BY-NC-SA", "CC BY-NC-SA"),
    "yuca1254": Corpus("Yucatec Maya", "10.34847/nkl.9cbb3619", "CC BY-NC", "CC BY-NC-ND"),
    "yura1255": Corpus("Yurakaré", "10.34847/nkl.7ca412wg", "CC BY-NC-ND", "audio linked"),
}
# fmt: on
COMMERCIAL_LANGUAGES = tuple(
    code
    for code, corpus in CORPORA.items()
    if corpus.annotation_license == corpus.audio_license == "CC BY"
)


def is_label(value):
    return not value or value.startswith("<<") or value in ("<p:>", "[INT]")


def annotation_path(root, code):
    folder = root / code
    canonical = folder / f"doreco_{code}_core_v2.zip"
    if canonical.is_file():
        return canonical
    candidates = []
    for path in sorted(folder.glob("*.zip")):
        with ZipFile(path) as bundle:
            if any(name.endswith("_ph.csv") for name in bundle.namelist()):
                candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"Multiple DoReCo annotation bundles for {code}: {candidates}")
    if candidates:
        return candidates[0]
    if code == DEFAULT_LANGUAGE:
        # Also read the flat layout used by the original English loader.
        for name in ("sout3282_annotations_core_v2.zip", "sout3282_annotations_v2.zip"):
            path = root / name
            if path.is_file():
                return path
    return None


def bundle_member(bundle, suffix):
    matches = [name for name in bundle.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one *{suffix} in {bundle.filename}, got {matches}")
    return matches[0]


def wav_files(root, code, annotations):
    folder = root / code
    prefix = f"doreco_{code}_"
    found = {
        path.stem: (None, path)
        for path in sorted(folder.rglob("*"))
        if path.suffix.lower() == ".wav"
        and path.stem.startswith(prefix)
        and path.is_file()
    }
    archives = sorted(folder.rglob("*.zip"))
    if code == DEFAULT_LANGUAGE:
        legacy = root / "sout3282_audio_core_v2.zip"
        if legacy.is_file():
            archives.append(legacy)
    for path in archives:
        if path == annotations:
            continue
        with ZipFile(path) as bundle:
            for name in bundle.namelist():
                member = Path(name)
                if member.suffix.lower() == ".wav" and member.stem.startswith(prefix):
                    found.setdefault(member.stem, (path, name))
    return found


def decode_wav(archive, path):
    """Decode WAV chunks using the container header and the repo's resampler."""
    with ExitStack() as stack:
        source = path
        if archive is not None:
            bundle = stack.enter_context(ZipFile(archive))
            source = stack.enter_context(bundle.open(path))
        with soundfile.SoundFile(source) as wav:
            rate = wav.samplerate
            floating_point = wav.subtype in {"FLOAT", "DOUBLE"}
            dtype = "float32" if floating_point else "int16"
            mono = np.empty(wav.frames, dtype=np.int16)
            offset = 0
            for block in wav.blocks(blocksize=2**20, dtype=dtype, always_2d=True):
                samples = block[:, 0] if wav.channels == 1 else block.mean(axis=1)
                if floating_point:
                    samples = np.clip(samples * 32768, -32768, 32767)
                mono[offset : offset + len(block)] = samples
                offset += len(block)
            if offset != len(mono):
                raise ValueError(f"Truncated DoReCo recording {path}")
    return audio_resample(mono, rate, TARGET_SAMPLE_RATE)


@lru_cache(maxsize=4096)
def phone_to_ipa(phone):
    try:
        ipa = xsampa2ipa(phone)
    except KeyError:
        return None
    # The shared converter still has placeholders for unsupported symbols.
    if (
        not ipa
        or any(c in ipa for c in " ⟨⟩")
        or ipa.startswith("͡")
        or ipa.endswith("͡")
    ):
        return None
    return ipa


@dataclass(frozen=True)
class Utterance:
    language: str
    recording: str
    speaker: str
    ref: str
    bounds: tuple[int, int]
    phones: tuple[tuple[str, int, int], ...]
    text: str


class DoReCoDataset(BaseDataset):
    def __init__(
        self,
        split=DEFAULT_SPLIT,
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        language=DEFAULT_LANGUAGE,
        *,
        root=TOP_DIR,
    ):
        """Load one glottocode, a list, or locally available languages with language="all".

        The commercial split requires CC BY 4.0 audio and annotations. Use split="all"
        for explicit access to other license terms. EN01/EN03 are the English speaker
        splits. Timestamps are sample offsets in the returned mono 16 kHz int16 clip.
        """
        super().__init__(split, include_timestamps, include_speaker_info, include_text)
        if split not in (DEFAULT_SPLIT, "all", "EN01", "EN03"):
            raise ValueError(f"Unknown DoReCo split {split!r}")
        self.root = Path(root)
        self.utterances = []
        self.vocab = set()
        self.skipped = {}
        self.citations = {}
        self._metadata = {}
        self._audio = {}
        self._recording_key = None
        self._recording_audio = None

        discover = language == "all"
        if discover:
            codes = CORPORA if split == "all" else COMMERCIAL_LANGUAGES
        elif isinstance(language, str):
            codes = [language]
        else:
            codes = list(dict.fromkeys(language))
        if not codes:
            raise ValueError("Specify at least one DoReCo language")

        for code in codes:
            if code not in CORPORA:
                raise ValueError(f"Unknown DoReCo glottocode {code!r}")
            corpus = CORPORA[code]
            if split != "all" and code not in COMMERCIAL_LANGUAGES:
                raise ValueError(
                    f"DoReCo {code} is outside the commercial split "
                    f"(audio {corpus.audio_license}, annotations {corpus.annotation_license}). "
                    "Use split='all' for separately authorized use."
                )
            path = annotation_path(self.root, code)
            if path is None:
                if discover:
                    continue
                raise FileNotFoundError(
                    f"Missing DoReCo annotations for {code} in {self.root}. "
                    f"Download from https://doreco.huma-num.fr/languages/{code}"
                )
            files = wav_files(self.root, code, path)
            if discover and not files:
                continue
            if split in ("EN01", "EN03") and code != DEFAULT_LANGUAGE:
                raise ValueError(
                    f"DoReCo split {split!r} is only available for {DEFAULT_LANGUAGE}"
                )
            self._audio[code] = files
            self._read_language(code, corpus, path)
        if discover and not self._audio:
            raise FileNotFoundError(f"No DoReCo corpora with audio in {self.root}")

    def _read_language(self, code, corpus, path):
        with ZipFile(path) as bundle:
            info = bundle.read(bundle_member(bundle, "_dataset-info.txt")).decode(
                "utf-8-sig"
            )
            if self.split != "all" and set(LICENSE_STATEMENT.findall(info)) != {
                ("annotation", "by/4.0/"),
                ("audio", "by/4.0/"),
            }:
                raise ValueError(
                    f"{path} must declare CC BY 4.0 for both annotations and audio"
                )
            _, _, citation = info.partition("should be cited as follows:")
            self.citations[code] = citation.strip().split("\n", 1)[0]
            with bundle.open(bundle_member(bundle, "_metadata.csv")) as f:
                metadata = pd.read_csv(f)
            with bundle.open(bundle_member(bundle, "_ph.csv")) as f:
                phones = pd.read_csv(
                    f,
                    usecols="file speaker ref ph start end wd_ID wd".split(),
                    keep_default_na=False,
                    dtype=dict.fromkeys("speaker ref ph wd_ID wd".split(), str),
                )
        for column in ("start", "end"):
            phones[column] = pd.to_numeric(phones[column], errors="coerce")
        metadata["name"] = f"doreco_{code}_" + metadata["name"]
        names = set(metadata["name"])
        if not set(phones["file"]).issubset(names):
            missing = sorted(set(phones["file"]) - names)
            raise ValueError(f"Missing metadata in {path} for {missing}")
        if self.include_speaker_info:
            self._metadata[code] = {
                row["name"]: dict(
                    row,
                    language=code,
                    language_name=corpus.name,
                    annotation_license=corpus.annotation_license,
                    audio_license=corpus.audio_license,
                    dataset_doi=corpus.doi,
                )
                for row in metadata.to_dict("records")
            }
        if self.split in ("EN01", "EN03"):
            phones = phones[phones["speaker"] == self.split]

        skipped = Counter()
        groups = phones.groupby(["file", "speaker", "ref"], sort=False)
        for (recording, speaker, ref), rows in groups:
            if is_label(ref) or len(rows) <= 10:
                continue
            words, timed = {}, []
            for row in rows.itertuples(index=False):
                if row.wd_ID and not is_label(row.wd):
                    words.setdefault(row.wd_ID, row.wd)
                if is_label(row.ph):
                    continue
                ipa = phone_to_ipa(row.ph)
                if ipa is None:
                    skipped[f"unsupported X-SAMPA {row.ph!r}"] += 1
                    break
                start = row.start * TARGET_SAMPLE_RATE
                end = row.end * TARGET_SAMPLE_RATE
                if not (
                    math.isfinite(start) and math.isfinite(end) and 0 <= start < end
                ):
                    skipped["invalid phone timing"] += 1
                    break
                timed.append((ipa, int(start), int(end)))
            else:
                if not timed:
                    skipped["no phones"] += 1
                    continue
                timed.sort(key=lambda phone: phone[1])
                bounds = timed[0][1], max(phone[2] for phone in timed)
                self.utterances.append(
                    Utterance(
                        code,
                        recording,
                        speaker,
                        ref,
                        bounds,
                        tuple(timed),
                        " ".join(words.values()),
                    )
                )
                self.vocab.update(phone[0] for phone in timed)
        self.skipped[code] = dict(skipped)
        if skipped:
            LOGGER.warning(
                "DoReCo %s skipped %d utterances: %s",
                code,
                sum(skipped.values()),
                dict(skipped),
            )

    @property
    def recordings(self):
        """The source recordings required by the selected utterances."""
        return {(item.language, item.recording) for item in self.utterances}

    def missing_recordings(self):
        return sorted(
            (code, recording)
            for code, recording in self.recordings
            if recording not in self._audio[code]
        )

    def __len__(self):
        return len(self.utterances)

    def _get_ix(self, ix):
        item = self.utterances[ix]
        start, end = item.bounds
        key = (item.language, item.recording)
        if key != self._recording_key:
            location = self._audio[item.language].get(item.recording)
            if location is None:
                raise FileNotFoundError(
                    f"Missing DoReCo recording {item.recording}.wav in {self.root}. "
                    f"Source: https://doreco.huma-num.fr/languages/{item.language}"
                )
            self._recording_audio = decode_wav(*location)
            self._recording_key = key
        recording = self._recording_audio

        if (
            item.phones[-1][1] >= len(recording)
            or end > len(recording) + AUDIO_END_TOLERANCE
        ):
            raise ValueError(
                f"DoReCo {item.ref} ends at {end}, outside "
                f"{item.recording}.wav ({len(recording)} samples)"
            )
        audio = recording[start : end + AUDIO_PADDING].copy()
        ipa = "".join(phone for phone, _, _ in item.phones)
        result = [ipa, audio]
        if self.include_timestamps:
            result.append(
                [
                    (phone, pos - start, min(stop, len(recording)) - start)
                    for phone, pos, stop in item.phones
                ]
            )
        if self.include_speaker_info:
            result.append(
                dict(
                    self._metadata[item.language][item.recording],
                    utt_spk_code=item.speaker,
                    utterance_ref=item.ref,
                )
            )
        if self.include_text:
            result.append(item.text)
        return tuple(result)


if __name__ == "__main__":
    dataset = DoReCoDataset(
        include_timestamps=True,
        include_speaker_info=True,
        include_text=True,
    )
    print(len(dataset))
    interactive_flag_samples(dataset)
