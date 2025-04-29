# L1 American English speech with diverse genders an ages: https://buckeyecorpus.osu.edu/
import os
import sys

import zipfile
from torch.utils.data import ConcatDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, split_utterance_into_multiple
from core.audio import audio_bytes_to_array, TARGET_SAMPLE_RATE
from core.codes import BUCKEYE2IPA

SOURCE_SAMPLE_RATE = 16000
DATA_ZIP = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "buckeye.zip")


def all_buckeye_speaker_splits(
    include_timestamps=False,
    include_speaker_info=False,
    include_text=False,
    split_config: "tuple[float, float] | None" = (2, 1),
):
    return ConcatDataset(
        BuckeyeDataset(
            split=s,
            include_timestamps=include_timestamps,
            include_speaker_info=include_speaker_info,
            include_text=include_text,
            split_config=split_config,
        )
        for s in SPEAKERS.keys()
    )


def get_utterances(
    speaker_zip,
    conversation,
    split_config: "tuple[float, float] | None" = (2, 1),
):
    with zipfile.ZipFile(speaker_zip.open(conversation)) as utterance_zip:
        all_sections = []
        utterances = [
            x.split(".")[0]
            for x in utterance_zip.namelist()
            if not x.startswith("__") and x.endswith(".wav")
        ]
        for utterance in utterances:
            with utterance_zip.open(f"{utterance}.wav") as wav_file:
                audio = audio_bytes_to_array(
                    wav_file.read(), SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE
                )

            with utterance_zip.open(f"{utterance}.phones") as phn_file:
                start = 0
                timestamped_phonemes = []
                for line in phn_file.read().decode("utf-8").split("\n")[9:]:
                    if line == "":
                        continue
                    line = line.split(";")[0].strip()
                    fields = line.split()
                    stop = float(fields[0])
                    phone = BUCKEYE2IPA.get(fields[2], "") if len(fields) >= 3 else ""
                    timestamped_phonemes.append(
                        (
                            phone,
                            int(start * TARGET_SAMPLE_RATE),
                            int(stop * TARGET_SAMPLE_RATE),
                        )
                    )
                    start = stop

            if split_config is not None:
                all_sections.extend(
                    split_utterance_into_multiple(
                        timestamped_phonemes,
                        audio,
                        split_convos_at_silence_seconds=split_config[0],
                        min_speech_seconds=split_config[1],
                    )
                )
            else:
                ipa = "".join(x[0] for x in timestamped_phonemes)
                all_sections.append((ipa, audio, timestamped_phonemes))

        return all_sections


class BuckeyeDataset(BaseDataset):
    def __init__(
        self,
        split="s40",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        split_config: "tuple[float, float] | None" = (2, 1),
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)
        self.split_config = split_config

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
        self.utterances_per_conversation = [
            len(get_utterances(self.speaker_zip, conversation, self.split_config))
            for conversation in self.conversations
        ]
        self.vocab = set(BUCKEYE2IPA.values())

    def __del__(self):
        self.speaker_zip.close()
        self.datazip.close()

    def __len__(self):
        return sum(self.utterances_per_conversation)

    def _get_ix(self, ix):
        for conversation, utterances in zip(
            self.conversations, self.utterances_per_conversation
        ):
            if ix < utterances:
                break
            ix -= utterances

        sections = get_utterances(self.speaker_zip, conversation, self.split_config)
        ipa, audio, timestamped_phonemes = sections[ix]

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
    dataset = BuckeyeDataset(split_config=None)
    print(len(dataset))
    print(dataset[0])
