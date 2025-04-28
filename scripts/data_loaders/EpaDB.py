# EpaDB from https://github.com/JazminVidal/EpaDB

import os
import sys

import zipfile
import textgrids
import io
import soundfile as sf
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import audio_bytes_to_array
from core.codes import arpabet2ipa
from core.ipa import filter_chars

SOURCE_SAMPLE_RATE = 16000  # Assume 16kHz 

DATA_ZIP = os.path.join(
    os.path.dirname(__file__), "..", "..", ".data", "EpaDB-master.zip"
)



class EpaDBDataset(BaseDataset):
    """
    Valid splits are "train" and "test" which contain scripted utterances spoken by Argentenian spanish speakers
    Two annotations of the dataset are provided (annotation1 and annotation2) when annotations conflict (which are only four samples) 
    but only the first annotation is used and we discard the utterances that appear in the second one to remove ambuious annptations.
    """
    def __init__(
            self, 
            include_timestamps=False, 
            split='train', 
            include_text=False
    ):
        super().__init__(split, include_timestamps, include_text)

        if include_timestamps:
            raise NotImplementedError("Timestamps are available but not parsed yet.")

        self.zip = zipfile.ZipFile(DATA_ZIP, "r")
        files = self.zip.namelist()
        print("First few files inside zip:", files[:10])

        # Collect available utterance ids from annotations_1
        self.files = []
        for file in files:
            if f"/{self.split}/" in file and "/annotations_1/" in file and file.endswith(".TextGrid"):
                utt_id = file.split("/")[-1].split(".")[0]
                self.files.append(file)

        # Collect utterance ids that have duplicates in annotations_2 (to discard)
        self.discard = set()
        for file in files:
            if f"/{self.split}/" in file and "/annotations_2/" in file and file.endswith(".TextGrid"):
                utt_id = file.split("/")[-1].split(".")[0]
                self.discard.add(utt_id)

        # Only keep files not in discard list
        self.files = [f for f in self.files if f.split("/")[-1].split(".")[0] not in self.discard]

    def __del__(self):
        self.zip.close()

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        filepath = self.files[ix]
        utt_id = filepath.split("/")[-1].split(".")[0]

        # Read wav
        with self.zip.open(filepath.replace("annotations_1", "waveforms").replace(".TextGrid", ".wav")) as wav_file:
            audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)

        # Read TextGrid
        with self.zip.open(filepath) as annotation_file:
            tg = textgrids.TextGrid()
            data = annotation_file.read()
            tg.parse(data)

        phones = tg.interval_tier_to_array("annotation")
        arpa = [c["label"] for c in phones if c["label"]]
        arpa = [
            label
            for label in arpa
            if label.lower() not in ["sil", "sp", "spn", "err", ""]
        ]
        ipa = [arpabet2ipa(label) for label in arpa]
        ipa = filter_chars("".join(ipa), filter_type="letters_rmv_tie")

        outputs = [ipa, audio]

        if self.include_text:
            with self.zip.open(filepath.replace("annotations_1", "transcriptions").replace(".TextGrid", ".lab")) as text_file:
                text = text_file.read().decode("utf-8").strip()
            outputs.append(text)

        return tuple(outputs)


if __name__ == "__main__":
    dataset = EpaDBDataset(
        split="train",
        include_timestamps=False,
        include_speaker_info=True,
        include_text=False,
    )
    print(dataset[10])
