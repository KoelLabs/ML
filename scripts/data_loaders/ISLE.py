# ISLE Speech corpus details here: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0083/
# paper: http://www.lrec-conf.org/proceedings/lrec2000/pdf/313.pdf

import os
import sys

import re
import zipfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_bytes_to_array
from core.codes import isle2ipa, IPA2ISLE

DATA_ZIP = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "S0083.zip")

def _parse(fmt, string):
    """
    !!!!WARNING: currently assumes format specifiers are separated by exactly one space!!!!

    Parses a string following a printf style format specifier into a sequence of matching values.
    Returns None if a perfect match cannot be made.
    """

    # Regex to extract format specifiers
    pattern = re.compile(r'%(-?0?\d*)([sd])')
    matches = pattern.findall(fmt)

    if not matches:
        return None  # No valid format specifiers found

    widths = []
    types = []

    for size, type_char in matches:
        width = int(size.lstrip('0') or '1')  # Handle %s or %d without width (default to 1)
        widths.append(abs(width))
        types.append(type_char)

    # Prepare to parse the input line
    values = []
    index = 0
    total_fields = len(widths)

    for i in range(total_fields):
        if index >= len(string):
            return None  # Not enough characters in line to extract all fields

        width = widths[i]
        while len(string) > index + width and string[index + width] != ' ': # treat format specifier width as minimum like perl, assumes delimiter is space
            width += 1
        field = string[index:index + width]
        if len(field) < width:
            return None  # Not enough characters for this field

        field = field.strip()
        type_char = types[i]

        if type_char == 'd':
            try:
                value = int(field)
            except ValueError:
                return None
        else:
            value = field

        values.append(value)
        index += width

        # Skip space between fields (if any), assuming single space delimiter between format specifiers
        if index < len(string) and string[index] == ' ':
            index += 1

    return values

class ISLEDataset(BaseDataset):
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

        self.vocab = set(IPA2ISLE.keys())

    def __del__(self):
        if hasattr(self, "isle"):
            self.isle.close()

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        sub_dir, session, base = self.files[ix]
        lab_path = f"{sub_dir}/{session}/MIL/{base}.txt"
        wav_path = f"{sub_dir}/{session}/WAVS/{base}.WAV"
        native_language = next(filter(lambda x: session in x[1], SPEAKERS.items()))[0].capitalize()

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
            # ignore BCKGRD lines
            if 'BCKGRD' in line:
                assert line.endswith(" _GARBAGE_                      BCKGRD BCKGRD . . BCKGRD ")

            # must parse the pattern with the extra comment column last
            parts = _parse("%1s %09d %09d %-30s %-3s %-6s %1s %1s %-6s %s", line) or _parse("%1s %09d %09d %-30s %-3s %-6s %1s %1s %-6s", line)
            assert parts is not None, f"Malformed line ({lab_path}, {wav_path}): '{line}'"
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
            assert native_language.startswith(lang)

            extra_marks = extra[0] if extra else None
            if self.include_text and word != "." and word != "#" and word != "##":
                text += word + " "
            if uk_phone == "." or "_" in uk_phone or "BCKGRD" in uk_phone:
                continue

            phones = uk_phone.split("-") if "-" in uk_phone else [uk_phone]
            non_phones = non_uk_phone.split("-") if "-" in non_uk_phone else [non_uk_phone]
            assert stress.upper() in ['P', '.', 'U'], stress
            if stress.upper() == 'P':
                phones[0] += '1'
                non_phones[0] += '1'
            for phone, non_phone in zip(phones, non_phones):
                phone_ipa = isle2ipa(phone)
                is_ambiguous = "=" in non_phone

                if self.include_ambiguous_flags:
                    ambiguous_flags.append(
                        (phone_ipa, is_ambiguous, non_phone)
                    )
                timestamped_phonemes.append((phone_ipa, start, end))

        ipa = "".join([x[0] for x in timestamped_phonemes])

        # --- Load audio ---
        with self.isle.open(wav_path) as wav_file:
            audio = audio_bytes_to_array(wav_file.read())

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
                    "native_language": native_language,
                    'comments': extra_marks,
                }
            )
        if self.include_text:
            result.append(text.strip())
        if self.include_ambiguous_flags:
            result.append(ambiguous_flags)
        return tuple(result)


SPEAKERS = {
    "GERMAN": ["SESS0006","SESS0011","SESS0012","SESS0015","SESS0020","SESS0021","SESS0161","SESS0162","SESS0163","SESS0164","SESS0181","SESS0182","SESS0183","SESS0184","SESS0185","SESS0186","SESS0187","SESS0188","SESS0189","SESS0190","SESS0191","SESS0192","SESS0193"], # fmt: skip
    "ITALIAN": ["SESS0003","SESS0040","SESS0041","SESS0121","SESS0122","SESS0123","SESS0124","SESS0125","SESS0126","SESS0127","SESS0128","SESS0129","SESS0130","SESS0131","SESS0132","SESS0133","SESS0134","SESS0135","SESS0136","SESS0137","SESS0138","SESS0139","SESS0140"], # fmt: skip
}


if __name__ == "__main__":
    isle = ISLEDataset(
        split="all",
        include_text=True,
        include_speaker_info=True,
        include_ambiguous_flags=True,
    )
    print(len(isle))
    interactive_flag_samples(isle)
