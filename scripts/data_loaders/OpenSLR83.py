# https://www.openslr.org/83/
# male and female recordings of English from various dialects of the UK and Ireland.

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_file_to_array

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "OpenSLR83")
UTTERANCES_CSV = os.path.join(DATA_DIR, "line_index_all.csv")
AUDIO_DIR = os.path.join(DATA_DIR, "audios")


class OpenSLR83Dataset(BaseDataset):
    """
    Valid splits are "all", "irish", "midlands", "northern", "scottish", "southern", "welsh"
    """

    def __init__(
        self,
        split="all",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        assert self.include_timestamps == False, "no timestamped phonemes (no phonemes)"

        assert split in [
            "all",
            "irish",
            "midlands",
            "northern",
            "scottish",
            "southern",
            "welsh",
        ], 'Split must be one of "all", "irish", "midlands", "northern", "scottish", "southern", "welsh"'

        self.utterances_df = pd.read_csv(
            UTTERANCES_CSV, header=None, sep=", ", engine="python"
        )
        self.utterances_df.columns = ["id", "file", "transcript"]
        if split != "all":
            self.utterances_df = self.utterances_df[
                self.utterances_df["file"].str.startswith(split[:2])
            ]

    def __len__(self):
        return len(self.utterances_df)

    def _get_ix(self, ix):
        speaker_id, file_name, transcript = self.utterances_df.iloc[ix]
        demographic_identifier = file_name.split("_")[0]
        male = demographic_identifier.endswith("m")
        dialect = demographic_identifier[:2]

        audio = audio_file_to_array(os.path.join(AUDIO_DIR, file_name + ".wav"))

        outputs = [None, audio]
        if self.include_speaker_info:
            outputs.append(
                {
                    "speaker_id": speaker_id,
                    "dialect": {
                        "ir": "Irish English",
                        "we": "Welsh English",
                        "mi": "Midlands English",
                        "no": "Northern English",
                        "sc": "Scottish English",
                        "so": "Southern English",
                    }[dialect],
                    "gender": "male" if male else "female",
                    "recording_id": file_name,
                }
            )
        if self.include_text:
            outputs.append(transcript)
        return tuple(outputs)


if __name__ == "__main__":
    dataset = OpenSLR83Dataset(
        split="all",
        include_timestamps=False,
        include_speaker_info=True,
        include_text=True,
    )
    interactive_flag_samples(dataset)
