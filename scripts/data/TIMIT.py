from .common import BaseDataset, audio_from_bytes, IPA_SUBSTITUTIONS

import zipfile
from collections import OrderedDict

SOURCE_SAMPLE_RATE = 16000


class TIMITDataset(BaseDataset):
    def __init__(
        self, split="train", include_timestamps=False, include_speaker_info=False
    ):
        super().__init__(split, include_timestamps, include_speaker_info)
        self.zip = zipfile.ZipFile("../.data/TIMIT.zip", "r")
        files = self.zip.namelist()
        self.files = list(
            set(
                map(
                    lambda x: x.split(".")[0],
                    filter(lambda x: x.startswith("data/" + split.upper()), files),
                )
            )
        )

    def __del__(self):
        self.zip.close()

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        filename = self.files[ix]
        speaker_id = filename.split("/")[-2][-4:]
        speaker = SPEAKERS[speaker_id]

        with self.zip.open(filename + ".WAV") as wav_file:
            audio = audio_from_bytes(wav_file.read(), SOURCE_SAMPLE_RATE)

        with self.zip.open(filename + ".PHN") as phn_file:
            timestamped_phonemes = []
            closure_interval_start = None
            for line in phn_file.read().decode("utf-8").split("\n"):
                if line == "":
                    continue
                start, end, phoneme = line.split()
                phoneme = phoneme.upper()

                if closure_interval_start:
                    cl_start, cl_end, cl_phoneme = closure_interval_start
                    if phoneme not in CLOSURE_INTERVALS[cl_phoneme]:
                        ipa_phoneme = TIMIT2IPA[CLOSURE_INTERVALS[cl_phoneme][0]]
                        timestamped_phonemes.append(
                            (ipa_phoneme, int(cl_start), int(cl_end))
                        )
                    else:
                        assert phoneme not in CLOSURE_INTERVALS
                        start = cl_start

                if phoneme in CLOSURE_INTERVALS:
                    closure_interval_start = (start, end, phoneme)
                    continue

                ipa_phoneme = TIMIT2IPA[phoneme]
                timestamped_phonemes.append((ipa_phoneme, int(start), int(end)))

                closure_interval_start = None
        ipa = "".join([x[0] for x in timestamped_phonemes])

        start_signal = timestamped_phonemes.pop(0)
        audio = audio[start_signal[2] :]
        timestamped_phonemes = [
            (x[0], x[1] - start_signal[2], x[2] - start_signal[2])
            for x in timestamped_phonemes
        ]

        result = OrderedDict()
        result["ipa"] = ipa
        result["audio"] = audio
        result["phonemes"] = timestamped_phonemes
        result["speaker"] = speaker
        if not self.include_timestamps:
            del result["phonemes"]
        if not self.include_speaker_info:
            del result["speaker"]
        return tuple(result.values())


# The closure intervals of stops which are distinguished from the stop
# release.  The closure symbols for the stops b,d,g,p,t,k are
# bcl,dcl,gcl,pcl,tck,kcl, respectively.  The closure portions of jh
# and ch, are dcl and tcl.
CLOSURE_INTERVALS = {
    "BCL": ["B"],
    "DCL": ["D", "JH"],
    "GCL": ["G"],
    "PCL": ["P"],
    "TCL": ["T", "CH"],
    "KCL": ["K"],
}
TIMIT2IPA = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AH0': 'ə', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɝ', 'ER0': 'ɚ', 'EY': 'eɪ', 'IH': 'ɪ', 'IH0': 'ɨ', 'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EL': 'l̩', 'EM': 'm̩', 'EN': 'n̩', 'F': 'f', 'G': 'g', 'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'Q': 'ʔ', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w', 'WH': 'ʍ', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', 'AX': 'ə', 'AX-H': 'ə̥', 'AXR': 'ɚ', 'DX': 'ɾ', 'ENG': 'ŋ̍', 'EPI': '', 'HV': 'ɦ', 'H#': '', 'IX': 'ɨ', 'NX': 'ɾ̃', 'PAU': '', 'UX': 'ʉ'}  # fmt: skip
for k in TIMIT2IPA.keys():
    if TIMIT2IPA[k] in IPA_SUBSTITUTIONS:
        TIMIT2IPA[k] = IPA_SUBSTITUTIONS[TIMIT2IPA[k]]

DIALECTS = {
    1: "New England",
    2: "Northern",
    3: "North Midland",
    4: "South Midland",
    5: "Southern",
    6: "New York City",
    7: "Western",
    8: "Army Brat",
}

SPEAKERS = {}
with zipfile.ZipFile("../.data/TIMIT.zip", "r") as zip:
    with zip.open("SPKRINFO.TXT") as file:
        # read as csv with ; indicating comment and double space as separator
        for line in file:
            line = line.decode("utf-8").strip()
            if not line.startswith(";"):
                parts = line.split("  ")
                SPEAKERS[parts[0]] = {
                    "SEX": parts[1],
                    "DIALECT": DIALECTS[int(parts[2])],
                    "SPLIT": parts[3],
                    "RECORDING_DATE": parts[4],
                    "BIRTH_DATE": parts[5],
                    "HEIGHT": parts[6],
                    "RACE": parts[7],
                    "EDUCATION": parts[8],
                }
                if len(parts) == 10:
                    SPEAKERS[parts[0]]["COMMENTS"] = parts[9]
