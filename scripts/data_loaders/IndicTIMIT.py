# Indic TIMIT, English read speech of Indian speakers, recorded by the SPIRE Lab of the
# Indian Institute of Science. Each speaker read the 2,342 distinct TIMIT sentences. The
# release has one archive per nativity, the speakers' native language, 18 in all.
# Data: https://spiredatasets.ee.iisc.ac.in/indictimitcorpus
# Paper: C. Yarra, R. Aggarwal, A. Rajpal, and P. K. Ghosh, "Indic TIMIT and Indic
#   English lexicon: A speech database of Indian speakers using TIMIT stimuli and a
#   lexicon from their mispronunciations," O-COCOSDA 2019, pp. 1-6,
#   doi: 10.1109/O-COCOSDA46868.2019.9041230.
# Terms: those of the SPIRE Lab's request form on the page above.
#
# Download instructions:
#   Request the data on the page above. It comes as one IISc_IndicTIMIT_<CODE>.tar.gz
#   per nativity, which holds the folder IISc_IndicTIMIT_<CODE>. From the repo root,
#   unpack each one into ./.data/IndicTIMIT, for example:
#       mkdir -p ./.data/IndicTIMIT
#       tar -xzf IISc_IndicTIMIT_ASS.tar.gz -C ./.data/IndicTIMIT
#   The human phone labels go in ./.data/IndicTIMIT/phone_labels. They are not in the
#   18 archives.
#
# The IPA comes only from the human phone labels. The 2019 paper says two linguists
# transcribed 2,342 recordings by hand, one of each sentence, and that the corpus
# provides those transcriptions. Recordings without such a label are left out. The label
# files were not in the archives when this loader was written, so read_phone_labels has
# no parser yet, and the loader stops with an error that says so.
#
# Tested on Nov 9, 2026, on the real files of all 18 nativities, with made up labels in
# place of the human ones. Every recording was read and checked against its transcript
# line and its WAV file. No person has listened yet.
# TODO: manual verification

import io
import os
import re
import sys

import panphon
import soundfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_resample, TARGET_SAMPLE_RATE

SOURCE = "https://spiredatasets.ee.iisc.ac.in/indictimitcorpus"
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".data", "IndicTIMIT")
)
# Added to every error about the data files, so that the error says what fixes it.
FIX = (
    f"Request Indic TIMIT at {SOURCE}, then from the repo root unpack the archive of"
    " each nativity, IISc_IndicTIMIT_<CODE>.tar.gz, into ./.data/IndicTIMIT"
)
# The code of each nativity, as in the archive names, and its native language, region
# and group, as the README on the page above gives them. The README writes MOG for the
# language of MOG, which a 2023 paper of the same lab writes Mog.
NATIVITIES = {
    "ASS": ("Assamese", "East + North-East", "Group-1"),
    "BEN": ("Bengali", "East + North-East", "Group-1"),
    "DIM": ("Dimasa", "East + North-East", "Group-1"),
    "GUJ": ("Gujarati", "West", "Group-3"),
    "HIN": ("Hindi", "North", "Group-2"),
    "KAN": ("Kannada", "South1", "Group-4"),
    "KON": ("Konkani", "West", "Group-3"),
    "MAI": ("Maithili", "East + North-East", "Group-1"),
    "MAL": ("Malwi", "North", "Group-2"),
    "MAR": ("Marathi", "West", "Group-3"),
    "MLY": ("Malayalam", "South2", "Group-5"),
    "MOG": ("Mog", "East + North-East", "Group-1"),
    "NEP": ("Nepali", "East + North-East", "Group-1"),
    "ODI": ("Odia", "East + North-East", "Group-1"),
    "PUN": ("Punjabi", "North", "Group-2"),
    "SAU": ("Saurashtra", "West", "Group-3"),
    "TAM": ("Tamil", "South2", "Group-5"),
    "TEL": ("Telugu", "South1", "Group-4"),
}
# A line of Transcripts_<CODE> is the stem of its WAV file, a space, and the sentence.
# A speaker ID is F or M and a number, as in all 80 speaker folders.
LINE = re.compile(r"IISc_IndicTIMIT_([FM][0-9]+)_([0-9]+) (\S.*)")
STEM = re.compile(r"IISc_IndicTIMIT_([FM][0-9]+)_([0-9]+)")
# Marks that are not phones, which no label may hold after the clean up.
NOT_PHONES = set("~-#0123456789")


def recording_id(recording):
    """The id of a recording, the stem of its WAV file, as the transcripts write it."""
    code, speaker, number, text = recording
    return f"IISc_IndicTIMIT_{speaker}_{number}"


def label_to_ipa(label):
    """The IPA of a label. The 2019 paper tells the annotators to write ∼ between words,
    and the example on its slides has a space between phones. Neither is a phone, and
    VoxAngeles.py removes white space too."""
    return "".join(label.replace("∼", " ").split())


def read_phone_labels():
    """{recording id: label}, every human phone label found, of any nativity.

    The 18 archives of the release have no label files, so their format is not known
    and they have no parser yet. This is the one place the parser goes. It should give
    every label it finds, keyed by the stem of the WAV file, and the dataset keeps those
    of its split. Until then this raises, so that no item gets a label of another kind.
    """
    folder = os.path.join(DATA_DIR, "phone_labels")
    found = []
    if os.path.isdir(folder):
        found = sorted(f for f in os.listdir(folder) if not f.startswith("."))
    if not found:
        raise FileNotFoundError(
            f"No human phone labels in {folder}. This loader returns only the"
            " recordings that two linguists transcribed by hand. The 2019 paper,"
            " section III.A, says there are 2,342, one of each sentence, and that the"
            " corpus provides them. They are not in the 18 archives of the release."
            " Recordings without such a label are left out by design. Ask the SPIRE Lab"
            " for the label files, put them in that folder, and add their parser to"
            " read_phone_labels."
        )
    raise NotImplementedError(
        f"{folder} holds files, such as {found[0]}, but their format was not known when"
        " this loader was written, so read_phone_labels has no parser for them yet"
    )


def read_transcripts(code):
    """The path of Transcripts_<CODE>, and (speaker, number, sentence) for each line."""
    path = os.path.join(DATA_DIR, f"IISc_IndicTIMIT_{code}", f"Transcripts_{code}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. {FIX}")
    try:
        with open(path, encoding="utf-8") as f:
            data = f.read()
    except UnicodeDecodeError as e:
        raise ValueError(f"{path} is not UTF-8 text. {e}. {FIX}") from e
    # All 18 transcripts end with a new line, so one that does not was cut short.
    if not data.endswith("\n"):
        raise ValueError(
            f"{path} is empty or does not end with a new line, so it may be cut short."
            f" {FIX}"
        )
    rows, seen = [], set()
    for k, line in enumerate(data[:-1].split("\n"), 1):
        match = LINE.fullmatch(line)
        if not match:
            raise ValueError(
                f"Line {k} of {path} is not a WAV name, a space and a sentence:"
                f" {line!r}. {FIX}"
            )
        speaker, number, text = match.groups()
        if (speaker, int(number)) in seen:
            raise ValueError(
                f"Line {k} of {path} repeats IISc_IndicTIMIT_{speaker}_{number}. {FIX}"
            )
        seen.add((speaker, int(number)))
        rows.append((speaker, int(number), text.rstrip()))
    return path, rows


def list_wavs(code):
    """(speaker folder, file name) of each WAV file in a nativity's speaker folders."""
    root = os.path.join(DATA_DIR, f"IISc_IndicTIMIT_{code}")
    found = set()
    for speaker in os.listdir(root):
        if os.path.isdir(os.path.join(root, speaker)):
            for name in os.listdir(os.path.join(root, speaker)):
                if name.endswith(".wav") and not name.startswith("."):
                    found.add((speaker, name))
    return found


def read_wav(path):
    """A WAV file as int16 at 16 kHz."""
    # Read once. The RIFF size is checked on these bytes, then they are decoded.
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{path} not found. {FIX}") from e
    # soundfile would return a shorter clip from a file that was cut short, with no
    # error. The size field of the RIFF header gives it away. It holds the file size
    # minus 8.
    if data[:4] != b"RIFF" or int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise ValueError(f"The WAV file {path} is cut short or is not a WAV. {FIX}")
    # soundfile reports the channels and the sample format, so that they can be checked.
    try:
        with soundfile.SoundFile(io.BytesIO(data)) as f:
            if f.channels != 1 or f.subtype != "PCM_16" or f.frames == 0:
                raise ValueError(
                    f"The WAV file {path} is not mono 16 bit audio with samples. {FIX}"
                )
            audio, rate = f.read(dtype="int16"), f.samplerate
    except soundfile.SoundFileError as e:
        # soundfile's own message names the bytes in memory, not the file, so it stays
        # in the cause.
        raise ValueError(f"Cannot decode the WAV file {path}. {FIX}") from e
    return audio_resample(audio, rate, TARGET_SAMPLE_RATE)


class IndicTIMITDataset(BaseDataset):
    """
    Valid splits are "all" or a nativity code from NATIVITIES, such as "BEN" for the
    Bengali speakers. Each speaker is in one nativity. "all" needs all 18 folders.

    Only the recordings with a human phone label are items, and their IPA is that label
    without its spaces and word marks. self.recordings lists every recording, labeled or
    not. The text is the sentence the speaker was asked to read.
    """

    def __init__(
        self,
        split="all",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        if include_timestamps:
            raise ValueError("Indic TIMIT has no phone timestamps")
        if split != "all" and split not in NATIVITIES:
            raise ValueError(
                f"Unknown split {split}. Valid splits are all and {list(NATIVITIES)}"
            )
        codes = list(NATIVITIES) if split == "all" else [split]
        if not os.path.isdir(DATA_DIR):
            raise FileNotFoundError(f"{DATA_DIR} not found. {FIX}")
        absent = [
            f"IISc_IndicTIMIT_{c}"
            for c in codes
            if not os.path.isdir(os.path.join(DATA_DIR, f"IISc_IndicTIMIT_{c}"))
        ]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} of the {len(codes)} nativity folders of split {split}"
                f" are not in {DATA_DIR}: {absent}. Each code is also a split. {FIX}"
            )

        # Every recording, as (nativity, speaker, sentence number, sentence). Each
        # sentence is one string object, shared by the recordings of every speaker.
        shared, owner = {}, {}
        self.recordings, wav_without_line = [], 0
        for code in codes:
            path, rows = read_transcripts(code)
            wavs = list_wavs(code)
            # A copy that was cut off while it was unpacked misses files, so this fails
            # at setup. The names must be exact, as _get_ix opens them.
            names = {(s, f"IISc_IndicTIMIT_{s}_{n}.wav") for s, n, _ in rows}
            missing = sorted(names - wavs)
            if missing:
                s, name = missing[0]
                raise FileNotFoundError(
                    f"{len(missing)} of the {len(rows)} WAV files that {path} lists are"
                    f" not on disk, such as {s}/{name}. {FIX}"
                )
            wav_without_line += len(wavs - names)
            for s, _, _ in rows:
                # A label's id names only the speaker, so a speaker is in one nativity.
                if owner.setdefault(s, code) != code:
                    raise ValueError(
                        f"Speaker {s} has recordings in both {owner[s]} and {code}, so"
                        f" a label for {s} would fit two recordings. {FIX}"
                    )
            self.recordings += sorted(
                (code, s, n, shared.setdefault(text, text)) for s, n, text in rows
            )

        # The labels of this split's speakers. A label of another speaker belongs to
        # another split, or to a speaker who is not in the release, and is counted.
        ids = {recording_id(r) for r in self.recordings}
        self.labels, unknown, other = {}, [], 0
        for key, label in read_phone_labels().items():
            match = STEM.fullmatch(key)
            if not match:
                raise ValueError(
                    f"read_phone_labels gave the id {key!r}, which is not the stem of a"
                    " WAV name such as IISc_IndicTIMIT_F30_1"
                )
            if match.group(1) not in owner:
                other += 1
            elif key not in ids:
                unknown.append(key)
            else:
                self.labels[key] = label
        if unknown:
            raise ValueError(
                f"{len(unknown)} phone labels are for sentences that the transcript of"
                f" their speaker does not list, such as {sorted(unknown)[0]}, so the"
                " label files do not fit the recordings. Check them and"
                " read_phone_labels"
            )
        odd = sorted(
            k
            for k, v in self.labels.items()
            if not label_to_ipa(v) or NOT_PHONES & set(label_to_ipa(v))
        )
        if odd:
            raise ValueError(
                f"{len(odd)} phone labels give no IPA or hold a mark that is not a"
                f" phone, ~, -, # or a digit, such as {odd[0]} {self.labels[odd[0]]!r}."
                " read_phone_labels must turn them into IPA first"
            )
        self.items = [r for r in self.recordings if recording_id(r) in self.labels]
        if not self.items:
            raise ValueError(
                f"None of the {len(self.recordings)} recordings of split {split} has a"
                " human phone label"
            )
        self.skipped = {
            "no_phone_label": len(self.recordings) - len(self.items),
            "wav_without_line": wav_without_line,
            "label_for_other_speaker": other,
        }

        feature_table = panphon.FeatureTable()
        self.vocab = {
            phone for r in self.items for phone in feature_table.segs_safe(self._ipa(r))
        }
        speakers = len({r[1] for r in self.items})
        print(
            f"Indic TIMIT {split}: {len(self.items)} items with a human phone label,"
            f" from {speakers} speaker{'' if speakers == 1 else 's'}. Left out are"
            f" {self.skipped['no_phone_label']} recordings with no label,"
            f" {self.skipped['wav_without_line']} WAV files with no transcript line,"
            f" and {other} labels of speakers outside this split"
        )

    def _ipa(self, recording):
        return label_to_ipa(self.labels[recording_id(recording)])

    def __len__(self):
        return len(self.items)

    def _get_ix(self, ix):
        code, speaker, number, text = self.items[ix]
        stem = recording_id(self.items[ix])
        path = os.path.join(DATA_DIR, f"IISc_IndicTIMIT_{code}", speaker, stem + ".wav")
        outputs = [self._ipa(self.items[ix]), read_wav(path)]
        if self.include_speaker_info:
            language, region, group = NATIVITIES[code]
            outputs.append(
                {
                    "id": stem,
                    "speaker": speaker,
                    "sex": speaker[0],
                    "nativity": code,
                    "native_language": language,
                    "region": region,
                    "group": group,
                    "sentence": number,
                    "raw_label": self.labels[stem],
                    "source": SOURCE,
                }
            )
        if self.include_text:
            outputs.append(text)
        return tuple(outputs)


if __name__ == "__main__":
    dataset = IndicTIMITDataset(include_speaker_info=True, include_text=True)
    print(len(dataset))
    interactive_flag_samples(dataset)
