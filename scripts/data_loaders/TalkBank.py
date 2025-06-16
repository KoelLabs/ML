# TalkBank is a project organized by Brian MacWhinney at Carnegie Mellon University to foster fundamental research in the study of human communication with an emphasis on spoken communication. https://talkbank.org/
# Supports 14 research areas including child speech, second language speech, medical conditions, and more.
# E.g.,
# SLABank is dedicated to providing corpora for the study of second language acquisition. https://slabank.talkbank.org/
# Aphasia is dedicated to providing corpora for the study of Aphasia. https://aphasia.talkbank.org/

import os
import sys

import re
import zipfile
import pylangacq

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_bytes_to_wav_array, TARGET_SAMPLE_RATE

TALKBANK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "TalkBank")
DATASETS = [
    f.removesuffix(".zip") for f in os.listdir(TALKBANK_DIR) if f.endswith(".zip")
]
MS_PER_SECOND = 1_000


class TalkBankDataset(BaseDataset):
    def __init__(
        self,
        split="BELC",
        include_timestamps=False,
        include_text=False,
        include_speaker_info=False,
        include_ambiguous_flags=False,
    ):
        super().__init__(
            split,
            include_timestamps,
            include_speaker_info,
            include_text,
            include_ambiguous_flags,
        )

        assert (
            not include_timestamps
        ), "Phoneme-level timestamp not annotated, word level ones exist but are not parsed"
        assert not include_ambiguous_flags, "Ambiguity not annotated"

        assert split in DATASETS, f"Invalid split {split}. Must be one of {DATASETS}"
        self.zip = zipfile.ZipFile(os.path.join(TALKBANK_DIR, f"{split}.zip"))
        files = [
            f
            for f in self.zip.namelist()
            if not f.startswith("__") and not f.endswith(".DS_Store")
        ]
        audios = sorted([f for f in files if f.endswith(".wav")])
        transcripts = sorted([f for f in files if f.endswith("cha")])
        transcripts = [
            t
            for t in transcripts
            if (
                f"{split}/wav/"
                + t.removeprefix(f"{split}/transcriptions/").removesuffix(".cha")
                + ".wav"
            )
            in audios
        ]
        assert len(audios) == len(
            transcripts
        ), f"#audios ({len(audios)}) != #transcripts ({len(transcripts)})"

        strs = []
        for transcript_path in transcripts:
            with self.zip.open(transcript_path) as f:
                strs.append(f.read().decode())
        reader = pylangacq.Reader.from_strs(strs)
        self.utterances = [
            (audio, utterance, header)
            for audio, transcript, header in zip(
                audios, reader.utterances(by_files=True), reader.headers()
            )
            for utterance in transcript  # type: ignore
            if len(re.sub(r"[^a-zA-Z]", "", utterance.tiers[utterance.participant])) > 0
        ]

    def __del__(self):
        if hasattr(self, "isle"):
            self.zip.close()

    def __len__(self):
        return len(self.utterances)

    def _get_ix(self, ix):
        audio_path, utterance, header = self.utterances[ix]
        assert (
            header["Media"].split(",")[0] in audio_path
        ), "Failed to pair audio and transcripts"

        with self.zip.open(audio_path) as f:
            audio = audio_bytes_to_wav_array(f.read(), "wav")
        start, end = utterance.time_marks
        audio = audio[
            int(start * TARGET_SAMPLE_RATE / MS_PER_SECOND) : int(
                end * TARGET_SAMPLE_RATE / MS_PER_SECOND
            )
        ]

        ipa = ""
        timestamped_phonemes = []
        print(utterance.tiers[utterance.participant])
        text = " ".join(
            " ".join(
                re.sub(r"[^a-zA-Z]", "", w.split("@")[0])
                for w in utterance.tiers[utterance.participant].split(" ")
            ).split()
        )

        # --- Return based on config ---
        result = [ipa, audio]
        if self.include_timestamps:
            result.append(timestamped_phonemes)
        if self.include_speaker_info:
            speaker_info = {
                "speaker_id": utterance.participant,
            }
            info = header["Participants"][utterance.participant]
            if "role" in info:
                speaker_info["speaker_type"] = info["role"]
            if "age" in info and len(info["age"].strip()) > 0:
                speaker_info["speaker_age"] = int(info["age"].split(";")[0])
            if "sex" in info:
                speaker_info["speaker_gender"] = info["sex"]
            if "education" in info:
                speaker_info["speaker_education"] = info["education"]
            if "language" in info:
                speaker_info["spoken_language"] = info["language"]
            if "Transcriber" in header:
                speaker_info["annotator_name"] = header["Transcriber"]
            if "Comment" in header:
                speaker_info["comments"] = header["Comment"]
            result.append(speaker_info)
        if self.include_text:
            result.append(text.strip())
        return tuple(result)


if __name__ == "__main__":
    dataset = TalkBankDataset(include_text=True, include_speaker_info=True)
    print(len(dataset))
    interactive_flag_samples(dataset)
