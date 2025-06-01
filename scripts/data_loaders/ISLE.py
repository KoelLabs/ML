# ISLE Speech corpus details here: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0083/
# paper: http://www.lrec-conf.org/proceedings/lrec2000/pdf/313.pdf

import os
import sys

import zipfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_bytes_to_array, TARGET_SAMPLE_RATE
from core.codes import isle2ipa, IPA2ARPABET


SOURCE_SAMPLE_RATE = 16_000
DATA_ZIP = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "S0083.zip")

MISSING = set()


class ISLE(BaseDataset):
    """
    Each speaker in the SPEAKERS dictionary is a valid split.
    The ISLE non-native speech data consists of 11484 utterances recorded
    by (mostly) intermediate-level German and Italian learners of
    English. ISLEDAT1/ISLEDAT2 contain 23 German sessions, ISLEDAT3/ISLEDAT4 contain 23 Italian sessions.
    there are ambigious phones like "=EH" that are marked with a flag in the ISLE dataset.
    """

    def __init__(
        self,
        split="SESS0006",
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
        self.isle = zipfile.ZipFile(DATA_ZIP)
        sub_directories = ["ISLEDAT1", "ISLEDAT2", "ISLEDAT3", "ISLEDAT4"]
        self.files = []

        if split == "all":
            splits = SPEAKERS["GERMAN"] + SPEAKERS["ITALIAN"]
        elif split == "german":
            splits = SPEAKERS["GERMAN"]
        elif split == "italian":
            splits = SPEAKERS["ITALIAN"]
        else:
            splits = [split]

        for sub_dir in sub_directories:
            for name in self.isle.namelist():
                for session in splits:
                    if f"{sub_dir}/{session}/MIL" in name and name.endswith(".txt"):
                        base = os.path.basename(name).replace(".txt", "")
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
        lab_path = f"{sub_dir}/{session}/MIL/{base}.txt"
        wav_path = f"{sub_dir}/{session}/WAVS/{base}.WAV"

        # --- Load MIL file ---
        with self.isle.open(lab_path) as f:
            lines = f.read().decode("utf-8").splitlines()

        # lang, start, end, word_label, expected_phone, uk_phone, expected_stress, corrected_stress, annotated_phone, *tags
        # Phones: expected (col 5) vs UK-mapped (col 6) vs annotated with native phones using = prefix (col 9)
        # Word labels only on first phone, then "." continuation or "#" for silence

        timestamped_phonemes = []
        ambiguous_flags = []
        text = ""
        for line in lines:

            parts = line.strip().split()
            if len(parts) < 9:
                print(f"Skipping line with insufficient parts: {line}", file=sys.stderr)
                continue
            (
                lang,
                start,
                end,
                word,
                exp_phone,
                uk_phone,
                exp_stress,
                stress,
                non_uk_phone,
                *extra,
            ) = parts
            extra_marks = extra[0] if extra else None
            if self.include_text and word != "." and word != "#" and word != "##":
                text += word + " "
            if "-" in uk_phone:
                for p in uk_phone.split("-"):
                    p_ipa = isle2ipa(p)

                    if "=" in non_uk_phone:
                        ambiguous_flags.append((p_ipa, "T", non_uk_phone))
                    else:
                        ambiguous_flags.append((p_ipa, "F", non_uk_phone))
                    timestamped_phonemes.append((p_ipa, int(start), int(end)))
            elif (
                "." not in uk_phone and "_" not in uk_phone and "BCKGRD" not in uk_phone
            ):
                phone_ipa = isle2ipa(uk_phone)
                # compare non_uk_phone annotations to uk_phone annotations
                if "=" in non_uk_phone:
                    ambiguous_flags.append((phone_ipa, "T", non_uk_phone))
                else:
                    ambiguous_flags.append((phone_ipa, "F", non_uk_phone))

                timestamped_phonemes.append((phone_ipa, int(start), int(end)))

        ipa = "".join([x[0] for x in timestamped_phonemes])

        # --- Load audio ---
        with self.isle.open(wav_path) as wav_file:
            audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)

        # --- Return based on config ---
        result = [ipa, audio]
        if self.include_timestamps:
            result.append(timestamped_phonemes)
        if self.include_speaker_info:
            result.append(
                {
                    "speaker_id": base,
                    "session": session,
                    "sub_dir": sub_dir,
                }
            )
        if self.include_text:
            result.append(text)
        if self.include_ambiguous_flags:
            result.append(ambiguous_flags)
        return tuple(result)


SPEAKERS = {
    "GERMAN": [
        "SESS0006",
        "SESS0011",
        "SESS0012",
        "SESS0015",
        "SESS0020",
        "SESS0021",
        "SESS0161",
        "SESS0162",
        "SESS0163",
        "SESS0164",
        "SESS0181",
        "SESS0182",
        "SESS0183",
        "SESS0184",
        "SESS0185",
        "SESS0186",
        "SESS0187",
        "SESS0188",
        "SESS0189",
        "SESS0190",
        "SESS0191",
        "SESS0192",
        "SESS0193",
    ],
    "ITALIAN": [
        "SESS0003",
        "SESS0040",
        "SESS0041",
        "SESS0121",
        "SESS0122",
        "SESS0123",
        "SESS0124",
        "SESS0125",
        "SESS0126",
        "SESS0127",
        "SESS0128",
        "SESS0129",
        "SESS0130",
        "SESS0131",
        "SESS0132",
        "SESS0133",
        "SESS0134",
        "SESS0135",
        "SESS0136",
        "SESS0137",
        "SESS0138",
        "SESS0139",
        "SESS0140",
    ],
}


if __name__ == "__main__":
    isle = ISLE(
        split="all",
        include_text=True,
        include_speaker_info=True,
        include_ambiguous_flags=True,
    )
    interactive_flag_samples(isle)
