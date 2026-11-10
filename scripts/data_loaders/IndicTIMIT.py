# Indic TIMIT, English read speech of Indian speakers, recorded by the SPIRE Lab of the Indian Institute of Science.
# Each speaker read the 2,342 distinct TIMIT sentences. The release has one archive per nativity, the speakers'
# native language, 18 in all.
# Data: https://spiredatasets.ee.iisc.ac.in/indictimitcorpus
# Paper: C. Yarra, R. Aggarwal, A. Rajpal, and P. K. Ghosh, "Indic TIMIT and Indic English lexicon: A speech
#   database of Indian speakers using TIMIT stimuli and a lexicon from their mispronunciations," O-COCOSDA 2019,
#   pp. 1-6, doi: 10.1109/O-COCOSDA46868.2019.9041230.
# Terms: those of the SPIRE Lab's request form on the page above.
#
# Download instructions:
#   Request the data on the page above. It comes as one IISc_IndicTIMIT_<CODE>.tar.gz per nativity.
#   From the repo root, unpack each one into ./.data/IndicTIMIT, for example:
#       mkdir -p ./.data/IndicTIMIT
#       tar -xzf IISc_IndicTIMIT_ASS.tar.gz -C ./.data/IndicTIMIT
#   The human phone labels go in ./.data/IndicTIMIT/phone_labels. They are not in the 18 archives.
#
# The IPA comes only from the human phone labels. The 2019 paper says two linguists transcribed 2,342 recordings
# by ear, one of each sentence, and that the corpus provides those transcriptions. Recordings without such a label
# are left out. The label files were not in the archives when this loader was written, so read_phone_labels has no
# parser yet, and the loader stops with an error that says so.
#
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
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "IndicTIMIT")
# Where the human phone labels go. The 18 archives of the release do not hold them.
LABELS_DIR = os.path.join(DATA_DIR, "phone_labels")
# The code of each nativity, as in the archive names, and its native language, region and group,
# as the README on the page above gives them.
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
# A line of Transcripts_<CODE> is the stem of its WAV file, a space, and the sentence, which ends in a space.
LINE = re.compile(r"IISc_IndicTIMIT_([A-Z]+[0-9]+)_([0-9]+) (\S.*)")


def recording_id(recording):
    """The id of a recording, the stem of its WAV file, as Transcripts_<CODE> writes it."""
    code, speaker, number, text = recording
    return f"IISc_IndicTIMIT_{speaker}_{number}"


def read_phone_labels(codes):
    """{recording id: label}, the human phone labels of the recordings of the nativities in codes.

    The release has no label files, so their format is not known and they have no parser yet.
    This is the one place the parser goes. Until then this raises, so that no item gets a label
    of another kind.
    """
    found = sorted(os.listdir(LABELS_DIR)) if os.path.isdir(LABELS_DIR) else []
    if not found:
        raise FileNotFoundError(
            f"No human phone labels in {LABELS_DIR}. This loader returns only the recordings that"
            " two linguists transcribed by ear. The 2019 paper, section III.A, says there are 2,342,"
            " one of each sentence, and that the corpus provides them. They are not in the 18"
            " archives of the release. Recordings without such a label are left out by design."
        )
    raise NotImplementedError(
        f"{LABELS_DIR} holds files, such as {found[0]}, but their format was not known when this"
        " loader was written, so it has no parser for them yet"
    )


def read_transcripts(code):
    """(speaker, sentence number, sentence) for each line of Transcripts_<CODE>."""
    path = os.path.join(DATA_DIR, f"IISc_IndicTIMIT_{code}", f"Transcripts_{code}")
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines[-1] == "":
        lines.pop()
    rows = []
    for line in lines:
        match = LINE.fullmatch(line)
        if not match:
            raise ValueError(f"Cannot read the line {line!r} of {path}")
        speaker, number, text = match.groups()
        rows.append((speaker, int(number), text.rstrip()))
    return rows


def read_wav(path):
    """A WAV file as int16 at 16 kHz."""
    # One read per file, since the files sit in a shared folder where each access costs time.
    with open(path, "rb") as f:
        data = f.read()
    # soundfile would return a shorter clip from a file that was cut short, without any error.
    # The size field of the RIFF header gives it away. It holds the file size minus 8 bytes.
    if data[:4] != b"RIFF" or int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise ValueError(f"The WAV file {path} is cut short or is not a WAV")
    # soundfile reports the channels and the sample format, so that they can be checked here.
    try:
        with soundfile.SoundFile(io.BytesIO(data)) as f:
            if f.channels != 1 or f.subtype != "PCM_16" or f.frames == 0:
                raise ValueError(
                    f"The WAV file {path} is not mono 16 bit audio with samples"
                )
            audio, rate = f.read(dtype="int16"), f.samplerate
    except soundfile.SoundFileError as e:
        raise ValueError(f"Cannot decode the WAV file {path}. {e}") from e
    return audio_resample(audio, rate, TARGET_SAMPLE_RATE)


class IndicTIMITDataset(BaseDataset):
    """
    Valid splits are "all" or a nativity code from NATIVITIES, such as "BEN" for the Bengali speakers.
    Each speaker is in one nativity. "all" needs the folders of all 18.

    Only the recordings with a human phone label are items, and their IPA is that label without its
    spaces and word marks. self.recordings lists every recording, labeled or not. The text is the
    sentence the speaker was asked to read.
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

        # Every recording, as (nativity, speaker, sentence number, sentence). Each sentence is one
        # string object, shared by the recordings of every speaker.
        shared = {}
        self.recordings = []
        for code in codes:
            rows = read_transcripts(code)
            self.recordings += sorted(
                (code, s, n, shared.setdefault(text, text)) for s, n, text in rows
            )
        self.labels = read_phone_labels(codes)
        self.items = [r for r in self.recordings if recording_id(r) in self.labels]
        self.skipped = {"no_phone_label": len(self.recordings) - len(self.items)}

        feature_table = panphon.FeatureTable()
        self.vocab = {
            phone for r in self.items for phone in feature_table.segs_safe(self._ipa(r))
        }
        print(
            f"Indic TIMIT {split}: {len(self.items)} items with a human phone label, from"
            f" {len({r[:2] for r in self.items})} speakers, and"
            f" {self.skipped['no_phone_label']} recordings without one left out"
        )

    def _ipa(self, recording):
        # The 2019 paper tells the annotators to write the word boundaries as ∼, and its example has a
        # space between phones. Neither is a phone. VoxAngeles and Tusom2021 remove white space too.
        return "".join(self.labels[recording_id(recording)].replace("∼", " ").split())

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
