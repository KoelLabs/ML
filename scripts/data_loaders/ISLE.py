# ISLE Speech corpus details here: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0083/ 
# paper: http://www.lrec-conf.org/proceedings/lrec2000/pdf/313.pdf

import os
import sys

import zipfile
import textgrids
from torch.utils.data import ConcatDataset


import requests
from contextlib import contextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_bytes_to_array, TARGET_SAMPLE_RATE
from core.codes import isle2ipa, IPA2ARPABET


SOURCE_SAMPLE_RATE = 16_000
DATA_ZIP= os.path.join(os.path.dirname(__file__), "..", "..", ".data", "S0083.zip")

MISSING = []
class ISLE(BaseDataset):
    """
    Each speaker in the SPEAKERS dictionary is a valid split.
    The ISLE non-native speech data consists of 11484 utterances recorded
    by (mostly) intermediate-level German and Italian learners of
    English. ISLEDAT1/ISLEDAT2 contain 23 German sessions, ISLEDAT3/ISLEDAT4 contain 23 Italian sessions.
    """
    def __init__(
        self,
        split="SESS0006",
        include_timestamps=False,
        include_text= False,
        include_speaker_info=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info)
        self.isle = zipfile.ZipFile(DATA_ZIP)
        sub_directories = ["ISLEDAT1", "ISLEDAT2", "ISLEDAT3", "ISLEDAT4"]
        self.files = []

        if split == "all":
            splits =SPEAKERS["GERMAN"] + SPEAKERS["ITALIAN"]
        elif split == "german":
            splits = SPEAKERS["GERMAN"]
        elif split == "italian":
            splits = SPEAKERS["ITALIAN"]
        else:
            splits = [split]
        
        for sub_dir in sub_directories:
            for name in self.isle.namelist():
                for session in splits:
                    if f"{sub_dir}/{session}/CLABS" in name and name.endswith(".LAB"):
                        base = os.path.basename(name).replace(".LAB", "")
                        full_sub_dir = name.split("/")[0] + "/" + sub_dir   
                        self.files.append((full_sub_dir, session, base))


        self.vocab = set(IPA2ARPABET.keys())

    def __del__(self):
        if hasattr(self, "isle"):
            self.isle.close()
    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        sub_dir, session, base = self.files[ix]
        lab_path = f"{sub_dir}/{session}/CLABS/{base}.LAB"
        wav_path = f"{sub_dir}/{session}/WAVS/{base}.WAV"

        # --- Load LAB file ---
        with self.isle.open(lab_path) as f:
            lines = f.read().decode("utf-8").splitlines()

        # --- Extract tier-1 and build timestamped phonemes ---
        timestamped_phonemes = []
        for line in lines:
            if line.strip() == "///":
                break
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"Skipping line with unexpected format: {line}", file=sys.stderr)
                continue
            start_us, end_us, label = parts
            if label in ("sp", "sil"):
                continue
            # phoneme = arpabet2ipa(label)  # or arpabet2ipa(label)
            try:
                phoneme = isle2ipa(label)
            except KeyError:
                MISSING.append((label, lab_path))
                print(f"Skipping unknown phoneme: {label}", file=sys.stderr)
                continue  # or phoneme = "�"
            timestamped_phonemes.append((phoneme, int(start_us), int(end_us)))


        ipa = "".join([x[0] for x in timestamped_phonemes])
        print(f"Processing {base} with IPA: {ipa}")

        # --- Load audio ---
        with self.isle.open(wav_path) as wav_file:
            audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)

        # --- Return based on config ---
        result = [ipa, audio]
        if self.include_timestamps:
            result.append(timestamped_phonemes)
        return tuple(result)
       
        
        

SPEAKERS = {
    "GERMAN": [
        "SESS0006", "SESS0011", "SESS0012", "SESS0015",
        "SESS0020", "SESS0021", "SESS0161", "SESS0162",
        "SESS0163", "SESS0164", "SESS0181", "SESS0182",
        "SESS0183", "SESS0184", "SESS0185", "SESS0186",
        "SESS0187", "SESS0188", "SESS0189", "SESS0190",
        "SESS0191", "SESS0192", "SESS0193",
    ],
    "ITALIAN": [
        "SESS0003", "SESS0040", "SESS0041", "SESS0121",
        "SESS0122", "SESS0123", "SESS0124", "SESS0125",
        "SESS0126", "SESS0127", "SESS0128", "SESS0129",
        "SESS0130", "SESS0131", "SESS0132", "SESS0133",
        "SESS0134", "SESS0135", "SESS0136", "SESS0137",
        "SESS0138", "SESS0139", "SESS0140",
    ]
}


if __name__ == "__main__":
    isle = ISLE(split="german", include_timestamps=True, include_speaker_info=True)
    print(f"Missing phonemes: {set(MISSING)}")