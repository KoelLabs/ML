# L1 American English speech with diverse genders an ages: https://buckeyecorpus.osu.edu/
import os
import sys

import zipfile
from torch.utils.data import ConcatDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import (
    audio_bytes_to_array,
    audio_array_clip,
    TARGET_SAMPLE_RATE,
    audio_array_play,
)
from core.codes import BUCKEYE2IPA

SOURCE_SAMPLE_RATE = 16000
DATA_ZIP = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "buckeye.zip")
REPLACE_INTERVIEWER_SOUNDS_WITH_SILENCE_SECONDS = 1


def all_buckeye_speaker_splits(
    include_timestamps=False,
    include_speaker_info=False,
    include_text=False,
):
    dataset = ConcatDataset(
        BuckeyeDataset(
            split=s,
            include_timestamps=include_timestamps,
            include_speaker_info=include_speaker_info,
            include_text=include_text,
        )
        for s in SPEAKERS.keys()
    )
    setattr(dataset, "split", "BuckeyeAll")
    return dataset


class BuckeyeDataset(BaseDataset):
    def __init__(
        self,
        split="s01",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        assert not include_text, "Text not parsed for Buckeye yet"
        # NOTE: also includes information on laughing etc. which is not parsed

        self.datazip = zipfile.ZipFile(DATA_ZIP, "r")
        speakers = [
            x
            for x in self.datazip.namelist()
            if not x.startswith("__") and x.endswith(".zip")
        ]
        speaker = f"buckeye/{split}.zip"
        assert speaker in speakers, f"Speaker {split} not found in {speakers}"
        self.speaker_zip = zipfile.ZipFile(self.datazip.open(speaker), "r")
        self.conversations = self.speaker_zip.namelist()
        self.conversation_zips = [
            zipfile.ZipFile(self.speaker_zip.open(conversation))
            for conversation in self.conversations
        ]
        self.utterances = [
            (conversation_zip, x.split(".")[0])
            for conversation_zip in self.conversation_zips
            for x in conversation_zip.namelist()
            if not x.startswith("__") and x.endswith(".wav")
        ]

        self.vocab = set(BUCKEYE2IPA.values())

    def __del__(self):
        for conversation_zip in self.conversation_zips:
            conversation_zip.close()
        self.speaker_zip.close()
        self.datazip.close()

    def __len__(self):
        return len(self.utterances)

    def _get_ix(self, ix):
        conversation_zip, utterance = self.utterances[ix]

        with conversation_zip.open(f"{utterance}.wav") as wav_file:
            audio = audio_bytes_to_array(
                wav_file.read(), SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

        with conversation_zip.open(f"{utterance}.phones") as phn_file:
            start = 0
            timestamped_phonemes = []
            accumulated_removal = 0
            for line in phn_file.read().decode("utf-8").split("\n")[9:]:
                if line == "":
                    continue
                line = line.split(";")[0].strip()
                fields = line.split()
                stop = float(fields[0]) - accumulated_removal

                if "B_TRANS" in fields[2].strip().upper():
                    # remove speech until B_TRANS which marks start of transcription
                    assert len(timestamped_phonemes) == 0, "B_TRANS"
                    audio = audio[int(stop * TARGET_SAMPLE_RATE) :]
                    accumulated_removal = stop
                    stop = 0
                elif fields[2].strip().upper() in ["IVER", "VOCNOISE", "SIL", "LAUGH"]:
                    # zero out audio for interviewer
                    audio[
                        int(start * TARGET_SAMPLE_RATE) : int(stop * TARGET_SAMPLE_RATE)
                    ] = 0
                    # reduce zeroed out silence duration to REPLACE_INTERVIEWER_SOUNDS_WITH_SILENCE_SECONDS
                    interviewer_duration = stop - start
                    if (
                        interviewer_duration
                        > REPLACE_INTERVIEWER_SOUNDS_WITH_SILENCE_SECONDS
                    ):
                        extraneous_duration = (
                            stop - start
                        ) - REPLACE_INTERVIEWER_SOUNDS_WITH_SILENCE_SECONDS
                        stop = start + extraneous_duration
                        audio = audio_array_clip(audio, start, stop)
                        accumulated_removal += extraneous_duration

                phone = BUCKEYE2IPA.get(fields[2], "") if len(fields) >= 3 else ""
                timestamped_phonemes.append(
                    (
                        phone,
                        int(start * TARGET_SAMPLE_RATE),
                        int(stop * TARGET_SAMPLE_RATE),
                    )
                )
                start = stop

            ipa = "".join(x[0] for x in timestamped_phonemes)

        result = [ipa, audio]
        if self.include_timestamps:
            result.append(timestamped_phonemes)
        if self.include_speaker_info:
            result.append(SPEAKERS[self.split])
        return tuple(result)


SPEAKERS = {
    "s01": {
        "gender": "female",
        "age": "young",  # age < 40
        "interviewer_gender": "female",
    },
    "s02": {
        "gender": "female",
        "age": "old",  # age > 40
        "interviewer_gender": "male",
    },
    "s03": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s04": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s05": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s06": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s07": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s08": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s09": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s10": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s11": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s12": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s13": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s14": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s15": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s16": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s17": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s18": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s19": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s20": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s21": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s22": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s23": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s24": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s25": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s26": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s27": {
        "gender": "female",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s28": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s29": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s30": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s31": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s32": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s33": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "female",
    },
    "s34": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s35": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s36": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "female",
    },
    "s37": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s38": {
        "gender": "male",
        "age": "old",
        "interviewer_gender": "male",
    },
    "s39": {
        "gender": "female",
        "age": "young",
        "interviewer_gender": "male",
    },
    "s40": {
        "gender": "male",
        "age": "young",
        "interviewer_gender": "female",
    },
}

if __name__ == "__main__":
    dataset = BuckeyeDataset(include_timestamps=True)
    audio = dataset[2][1]
    audio_array_play(audio)
    print(len(dataset))
    interactive_flag_samples(dataset, split_config=(2, 1, 5))
