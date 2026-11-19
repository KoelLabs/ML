"""Phone-aligned speech from DoReCo 2.0. See docs/datasets/DoReCo.md for setup."""

import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE, audio_resample
from core.codes import xsampa2ipa
from data_loaders.common import BaseDataset, interactive_flag_samples

TOP_DIR = Path(__file__).resolve().parents[2] / ".data" / "DoReCo"
LANGUAGE_TABLE = Path(__file__).with_name("doreco_language_table.json")
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


def language_table():
    with LANGUAGE_TABLE.open(encoding="utf-8") as f:
        return {entry["glottocode"]: entry for entry in json.load(f)["languages"]}


def is_commercial(entry):
    return entry["annotation_license"] == entry["audio_license"] == "CC BY"


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
    found = {}
    zips = sorted(folder.rglob("*.zip"))
    if code == DEFAULT_LANGUAGE:
        legacy = root / "sout3282_audio_core_v2.zip"
        if legacy.is_file():
            zips.append(legacy)
    for path in zips:
        if path == annotations:
            continue
        with ZipFile(path) as bundle:
            for name in bundle.namelist():
                member = Path(name)
                if member.suffix.lower() == ".wav" and member.stem.startswith(
                    f"doreco_{code}_"
                ):
                    found.setdefault(member.stem, (path, name))
    for path in sorted(folder.rglob("*")):
        if (
            path.suffix.lower() == ".wav"
            and path.stem.startswith(f"doreco_{code}_")
            and path.is_file()
        ):
            found[path.stem] = (None, path)
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
        self._annotation_paths = {}
        self._recording_key = None
        self._recording_audio = None
        table = language_table()

        if language == "all":
            codes = []
            for code, entry in sorted(table.items()):
                if split != "all" and not is_commercial(entry):
                    continue
                path = annotation_path(self.root, code)
                if path is not None and (files := wav_files(self.root, code, path)):
                    self._audio[code] = files
                    self._annotation_paths[code] = path
                    codes.append(code)
            if not codes:
                raise FileNotFoundError(f"No DoReCo corpora with audio in {self.root}")
        elif isinstance(language, str):
            codes = [language]
        else:
            codes = list(dict.fromkeys(language))
        if not codes:
            raise ValueError("Specify at least one DoReCo language")
        if split in ("EN01", "EN03") and codes != [DEFAULT_LANGUAGE]:
            raise ValueError(
                f"DoReCo split {split!r} is only available for {DEFAULT_LANGUAGE}"
            )

        for code in codes:
            if code not in table:
                raise ValueError(f"Unknown DoReCo glottocode {code!r}")
            entry = table[code]
            if split != "all" and not is_commercial(entry):
                raise ValueError(
                    f"DoReCo {code} is outside the commercial split "
                    f"(audio: {entry['audio_license']}; annotations: "
                    f"{entry['annotation_license']}). Use split='all' for separately "
                    "authorized use."
                )
            self._read_language(code, entry)

    def _read_language(self, code, entry):
        path = self._annotation_paths.get(code) or annotation_path(self.root, code)
        if path is None:
            raise FileNotFoundError(
                f"No DoReCo annotations for {code} in {self.root}. "
                f"Source: {entry['language_page']}"
            )
        self._annotation_paths[code] = path
        with ZipFile(path) as bundle:
            info = bundle.read(bundle_member(bundle, "_dataset-info.txt")).decode(
                "utf-8-sig"
            )
            if self.split != "all":
                licenses = defaultdict(set)
                for material, license_slug in LICENSE_STATEMENT.findall(info):
                    licenses[material].add(license_slug)
                if any(
                    licenses[material] != {"by/4.0/"}
                    for material in ("annotation", "audio")
                ):
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
                    usecols=[
                        "file",
                        "speaker",
                        "ref",
                        "ph",
                        "start",
                        "end",
                        "wd_ID",
                        "wd",
                    ],
                    keep_default_na=False,
                    dtype={
                        "speaker": str,
                        "ref": str,
                        "ph": str,
                        "wd_ID": str,
                        "wd": str,
                    },
                )
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
                    language_name=entry["name"],
                    annotation_license=entry["annotation_license"],
                    audio_license=entry["audio_license"],
                    dataset_doi=entry["doi"],
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
                try:
                    start = float(row.start) * TARGET_SAMPLE_RATE
                    end = float(row.end) * TARGET_SAMPLE_RATE
                except (ValueError, TypeError):
                    skipped["invalid phone timing"] += 1
                    break
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
            if recording not in self._wav_index(code)
        )

    def _wav_index(self, code):
        if code not in self._audio:
            self._audio[code] = wav_files(self.root, code, self._annotation_paths[code])
        return self._audio[code]

    def __len__(self):
        return len(self.utterances)

    def _get_ix(self, ix):
        item = self.utterances[ix]
        start, end = item.bounds
        key = (item.language, item.recording)
        if key != self._recording_key:
            location = self._wav_index(item.language).get(item.recording)
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
