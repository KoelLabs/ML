# Arabic Speech Corpus, Modern Standard Arabic read by one male speaker with a Damascus
# accent, made for speech synthesis. 1,813 train and 100 test utterances, with phone
# times.
# Site: https://en.arabicspeechcorpus.com/
# Thesis: Halabi, N. (2016). Modern Standard Arabic phonetics for speech synthesis.
#   University of Southampton. https://eprints.soton.ac.uk/409695/
# License: the sources disagree, CC BY 4.0, CC BY-NC-SA 3.0 and 4.0, and a README in
#   the zip that asks for an agreement for commercial use. See the notes,
#   docs/datasets/ArabicSpeechCorpus.md.
#
# Download instructions:
#   mkdir -p ./.data/ArabicSpeechCorpus
#   curl -L -o ./.data/ArabicSpeechCorpus/arabic-speech-corpus.zip \
#     https://en.arabicspeechcorpus.com/arabic-speech-corpus.zip
#   Keep it zipped. The loader reads the zip as it is, as TIMIT.py does. Its size and
#   SHA-256 are in the notes.
#
# Tested on Nov 10, 2026, on every item of train and test of that zip, with the real audio.
# No person has listened to the clips yet.
# TODO: manual verification

import io
import json
import os
import sys
import zipfile

import soundfile
import textgrids

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_resample, TARGET_SAMPLE_RATE

SITE = "https://en.arabicspeechcorpus.com/"
DATA_ZIP = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    ".data",
    "ArabicSpeechCorpus",
    "arabic-speech-corpus.zip",
)
PHONE_TABLE = os.path.join(
    os.path.dirname(__file__), "arabic_speech_corpus_phones.json"
)
# Added to every error about the data, so that the error says what fixes it.
FIX = (
    f"Download {SITE}arabic-speech-corpus.zip to .data/ArabicSpeechCorpus/ and keep it"
    " zipped, as the header of ArabicSpeechCorpus.py says"
)
# The two folders of the zip. The thesis used "test set" to evaluate the corpus.
FOLDERS = {"train": "arabic-speech-corpus/", "test": "arabic-speech-corpus/test set/"}
SPLITS = ("train", "test", "all")


def phone_to_ipa(label, phones):
    """(IPA, is consonant, is stressed) of a TextGrid label, or None if unknown."""
    stressed = label.endswith("'")
    base = label[:-1] if stressed else label
    entry, length = phones.get(base), ""
    # A doubled consonant symbol, such as dd, is a geminate (Table 4-7 of the thesis).
    half = base[: len(base) // 2]
    if entry is None and base == half * 2:
        if phones.get(half, {}).get("kind") == "consonant":
            entry, length = phones[half], "ː"
    # The apostrophe follows a stressed vowel, so it cannot follow a consonant.
    if entry is None or (stressed and entry["kind"] != "vowel"):
        return None
    return entry["ipa"] + length, entry["kind"] == "consonant", stressed


def with_stress(units):
    """The IPA string, with ˈ at the start of the syllable of each stressed vowel.

    units holds None for a pause. A syllable of Modern Standard Arabic starts with
    exactly one consonant (Broselow, Arabic syllable structure), so ˈ goes before the
    consonant right before the vowel, or before a whole geminate.
    """
    out = []
    for k, unit in enumerate(units):
        if unit is None:
            continue
        out.append(unit[0])
        if unit[2]:
            onset = k > 0 and units[k - 1] is not None and units[k - 1][1]
            out.insert(len(out) - (2 if onset else 1), "ˈ")
    return "".join(out)


def read_wav(name):
    """A WAV file of the zip as int16 at 16 kHz."""
    with zipfile.ZipFile(DATA_ZIP) as z:
        data = z.read(name)
    # soundfile returns a short clip from a cut file with no error. The RIFF size field,
    # the file size minus 8 bytes, gives such a file away.
    if data[:4] != b"RIFF" or int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise ValueError(f"The WAV file {name} is cut short or is not a WAV. {FIX}")
    # Not audio_bytes_to_array of core/audio.py, which expects the samples at byte 44.
    # These files have them at byte 1024, after a bext chunk and others.
    try:
        with soundfile.SoundFile(io.BytesIO(data)) as f:
            if f.channels != 1 or f.subtype != "PCM_16":
                raise ValueError(f"The WAV file {name} is not mono 16 bit PCM. {FIX}")
            audio, rate = f.read(dtype="int16"), f.samplerate
    except soundfile.SoundFileError as e:
        raise ValueError(f"Cannot decode the WAV file {name}. {e}. {FIX}") from e
    if len(audio) == 0:
        raise ValueError(f"The WAV file {name} has no samples. {FIX}")
    return audio_resample(audio, rate, TARGET_SAMPLE_RATE)


def read_phone_tier(z, name):
    """The (label, start, end) of each interval of the phones tier of one TextGrid."""
    data = z.read(name)
    # 7 TextGrids start with a UTF-8 byte order mark, which the parser cannot take.
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    grid = textgrids.TextGrid()
    try:
        grid.parse(data)
    except (TypeError, ValueError, IndexError) as e:
        raise ValueError(f"Cannot parse the TextGrid {name}. {e}. {FIX}") from e
    if "phones" not in grid or len(grid["phones"]) == 0:
        raise ValueError(f"The TextGrid {name} has no phones tier. {FIX}")
    return [(i.text, i.xmin, i.xmax) for i in grid["phones"]]


def read_transcript(z, folder, stems):
    """The orthographic transcript of one folder, in Buckwalter, by file stem."""
    name = folder + "orthographic-transcript.txt"
    try:
        text = z.read(name).decode("utf-8")
    except KeyError as e:
        raise ValueError(f"{name} is not in the zip. {FIX}") from e
    lines = {}
    for line in text.splitlines():
        # A line is "ARA NORM  0002.wav" "the text", with two spaces in the name.
        wav, sep, rest = line.partition('" "')
        stem = wav[1:].removesuffix(".wav")
        if not (sep and wav[:1] == '"' and rest[-1:] == '"') or stem in lines:
            raise ValueError(f"Cannot read the line {line!r} of {name}. {FIX}")
        lines[stem] = rest[:-1]
    if set(lines) != set(stems):
        raise ValueError(f"The lines of {name} do not match its WAV files. {FIX}")
    return lines


def files_in(names, prefix, suffix):
    """The sorted stems of the zip entries in one folder with one extension."""
    return sorted(
        n[len(prefix) : -len(suffix)]
        for n in names
        if n.startswith(prefix) and n.endswith(suffix) and "/" not in n[len(prefix) :]
    )


class ArabicSpeechCorpusDataset(BaseDataset):
    """
    Valid splits are "train" and "test", the two folders of the zip, or "all", which is
    train then test. Items are in file name order within a split.

    The IPA comes from the phones tier of each TextGrid and the table in
    arabic_speech_corpus_phones.json. The IPA has stress marks. The timestamps and the
    vocab do not, so split_utterance_into_multiple gives IPA with no stress.
    An utterance with a phone that has no IPA, such as dist, is skipped and counted in
    self.skipped. raw_label holds the TextGrid labels, with outer spaces removed.

    The text is the orthographic transcript in Buckwalter letters, with "-" for a pause.
    """

    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}. Valid splits are {SPLITS}")
        if not os.path.exists(DATA_ZIP):
            raise FileNotFoundError(f"{DATA_ZIP} not found. {FIX}")
        with open(PHONE_TABLE, encoding="utf-8") as f:
            table = json.load(f)
        self.items, self.skipped, self.cleaned_labels = [], {}, 0
        try:
            z = zipfile.ZipFile(DATA_ZIP)
        except zipfile.BadZipFile as e:
            raise ValueError(f"{DATA_ZIP} is not a readable zip. {e}. {FIX}") from e
        with z:
            names = z.namelist()
            for part in ("train", "test") if split == "all" else (split,):
                folder = FOLDERS[part]
                stems = files_in(names, folder + "wav/", ".wav")
                if files_in(names, folder + "textgrid/", ".TextGrid") != stems:
                    raise ValueError(
                        f"The WAV and TextGrid files of {folder} differ. {FIX}"
                    )
                transcript = read_transcript(z, folder, stems)
                for stem in stems:
                    grid = f"{folder}textgrid/{stem}.TextGrid"
                    item = self._make_item(read_phone_tier(z, grid), grid, table)
                    if item is not None:
                        item.update(split=part, stem=stem, text=transcript[stem])
                        self.items.append(item)

        self.vocab = {p for item in self.items for p, _, _ in item["phones"] if p}
        skipped = ", ".join(f"{k} {v}" for k, v in sorted(self.skipped.items()))
        print(
            f"Arabic Speech Corpus {split}: {len(self.items)} items, skipped"
            f" {skipped or 'none'}, spaces removed from {self.cleaned_labels} labels"
        )

    def _make_item(self, intervals, grid, table):
        """One item from its phone intervals, or None if it has to be skipped."""
        labels = [label.strip() for label, _, _ in intervals]
        if "" in labels:
            raise ValueError(f"The TextGrid {grid} has an empty phone label. {FIX}")
        # 3 labels carry a trailing space, such as "E ".
        self.cleaned_labels += sum(a != b[0] for a, b in zip(labels, intervals))
        phones = table["phones"]
        units = [None if a == "sil" else phone_to_ipa(a, phones) for a in labels]
        missing = [a for a, u in zip(labels, units) if a != "sil" and u is None]
        new = [a for a in missing if a not in table["no_ipa"]]
        if new:
            raise ValueError(
                f"Unknown phone {new[0]!r} in {grid}. Add it to"
                " arabic_speech_corpus_phones.json, in phones or no_ipa, with a source"
            )
        reason = f"no_ipa:{missing[0]}" if missing else None
        if not missing and all(u is None for u in units):
            reason = "no_phones"
        if reason:
            self.skipped[reason] = self.skipped.get(reason, 0) + 1
            return None
        rate = TARGET_SAMPLE_RATE
        return {
            "ipa": with_stress(units),
            # A pause has an empty phone, as split_utterance_into_multiple expects.
            "phones": [
                (u[0] if u else "", round(s * rate), round(e * rate))
                for u, (_, s, e) in zip(units, intervals)
            ],
            "raw_label": " ".join(labels),
        }

    def __len__(self):
        return len(self.items)

    def _get_ix(self, ix):
        item = self.items[ix]
        wav = f"{FOLDERS[item['split']]}wav/{item['stem']}.wav"
        outputs = [item["ipa"], read_wav(wav)]
        if self.include_timestamps:
            # The audio is the whole file, so the TextGrid times need no shift.
            outputs.append(list(item["phones"]))
        if self.include_speaker_info:
            outputs.append(
                {
                    "id": f"{item['split']}/{item['stem']}",
                    "split": item["split"],
                    "speaker": "single male speaker, Damascus accent",
                    "language": "Modern Standard Arabic",
                    "source": SITE,
                    "raw_label": item["raw_label"],
                }
            )
        if self.include_text:
            outputs.append(item["text"])
        return tuple(outputs)


if __name__ == "__main__":
    dataset = ArabicSpeechCorpusDataset(include_speaker_info=True, include_text=True)
    print(len(dataset))
    interactive_flag_samples(dataset)
