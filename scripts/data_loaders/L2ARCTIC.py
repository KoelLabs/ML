# L2 English Speech from https://psi.engr.tamu.edu/l2-arctic-corpus/

import os
import sys

import zipfile
import textgrids
from torch.utils.data import ConcatDataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_bytes_to_array, TARGET_SAMPLE_RATE
from core.codes import arpabet2ipa, IPA2ARPABET

SOURCE_SAMPLE_RATE = 44100
DATA_ZIP = os.path.join(
    os.path.dirname(__file__), "..", "..", ".data", "l2arctic_release_v5.0.zip"
)


def all_arctic_speaker_splits(
    include_timestamps=False,
    include_speaker_info=False,
    include_text=False,
):
    dataset = ConcatDataset(
        L2ArcticDataset(
            split=s,
            include_timestamps=include_timestamps,
            include_speaker_info=include_speaker_info,
            include_text=include_text,
        )
        for s in SPEAKERS.keys()
    )
    setattr(dataset, "split", "L2ArcticScripted")
    return dataset


def L2Symbol2Phoneme(c):
    c = c.split(",")[1].strip() if "," in c else c
    c = (
        c.split(" ")[0]
        .replace("*", "")  # nearest english phoneme has been annotated
        .replace(")", "")  # typo
        .replace("_", "")  # typo
        .replace("`", "")  # typo
        .replace("8", "")  # typo
        .replace("!", "")  # typo
        .upper()
    )
    if c.lower() in [
        "sil",
        "err",
        "sp",
        "spn",
    ]:
        return ""

    return arpabet2ipa(c)


class L2ArcticDataset(BaseDataset):
    """
    Each speaker in the SPEAKERS dictionary is a valid split.
    These splits involve short scripted utterances of English (3.6 seconds on average) spoken by non-native speakers.
    The suitcase_corpus is a subset of L2-Arctic featuring non-native English speech in free-form/non-scripted conversations.
    This is also a valid split.
    """

    def __init__(
        self,
        split="ABA",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        self.arctic = zipfile.ZipFile(DATA_ZIP)
        self.suitcase = self.arctic.open(f"{split}.zip")
        self.zip = zipfile.ZipFile(self.suitcase)
        files = self.zip.namelist()
        self.files = list(
            map(
                lambda x: x.split(".")[0].split("/")[-1],
                filter(
                    lambda x: x.startswith(f"{split}/annotation/")
                    and x.endswith(".TextGrid"),
                    files,
                ),
            )
        )
        self.vocab = set(IPA2ARPABET.keys())

    def __del__(self):
        self.zip.close()
        self.suitcase.close()
        self.arctic.close()

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        filename = self.files[ix]
        with self.zip.open(f"{self.split}/wav/{filename}.wav") as wav_file:
            audio = audio_bytes_to_array(wav_file.read(), SOURCE_SAMPLE_RATE)

        with self.zip.open(
            f"{self.split}/annotation/{filename}.TextGrid"
        ) as annotation_file:
            tg = textgrids.TextGrid()
            data = annotation_file.read()
            # If two lines look like this:
            #     text = "oʊ, ɔ,
            # oʊ, ə, "
            # patch them to be one line:
            #     text = "oʊ, ɔ, oʊ, ə, "
            patched_data = []
            for line in data.decode("utf-8").split("\n"):
                if (
                    line == 'oʊ, ə, " '
                    or line == '" '
                    or line == 's" '
                    or (line.count('"') == 1 and line.endswith('" '))
                ):
                    patched_data[-1] = patched_data[-1].strip() + line
                else:
                    patched_data.append(line)
            data = "\n".join(patched_data).encode("utf-8")
            try:
                tg.parse(data)
            except Exception as e:
                print(data.decode("utf-8"))
                raise e

            if self.include_text:
                words = tg.interval_tier_to_array("words")
                text = " ".join(c["label"] for c in words if c["label"])

            arpa = tg.interval_tier_to_array("phones")
            timestamped_phonemes = [
                (
                    L2Symbol2Phoneme(c["label"]),
                    int(c["begin"] * TARGET_SAMPLE_RATE),
                    int(c["end"] * TARGET_SAMPLE_RATE),
                )
                for c in arpa
                if c["label"]
            ]
            # print(timestamped_phonemes)
            ipa = "".join([x[0] for x in timestamped_phonemes])

        result = [ipa, audio]
        if self.include_timestamps:
            result.append(timestamped_phonemes)
        if self.include_speaker_info:
            speaker_id = self.split if self.split != "suitcase_corpus" else filename
            speaker = SPEAKERS[speaker_id.upper()]
            speaker["id"] = speaker_id
            result.append(speaker)
        if self.include_text:
            result.append(text)
        return tuple(result)


SPEAKERS = {
    "ABA": {
        "gender": "M",
        "native-language": "Arabic",
    },
    "SKA": {
        "gender": "F",
        "native-language": "Arabic",
    },
    "YBAA": {
        "gender": "M",
        "native-language": "Arabic",
    },
    "ZHAA": {
        "gender": "F",
        "native-language": "Arabic",
    },
    "BWC": {
        "gender": "M",
        "native-language": "Chinese",
    },
    "LXC": {
        "gender": "F",
        "native-language": "Chinese",
    },
    "NCC": {
        "gender": "F",
        "native-language": "Chinese",
    },
    "TXHC": {
        "gender": "M",
        "native-language": "Chinese",
    },
    "ASI": {
        "gender": "M",
        "native-language": "Hindi",
    },
    "RRBI": {
        "gender": "M",
        "native-language": "Hindi",
    },
    "SVBI": {
        "gender": "F",
        "native-language": "Hindi",
    },
    "TNI": {
        "gender": "F",
        "native-language": "Hindi",
    },
    "HJK": {
        "gender": "F",
        "native-language": "Korean",
    },
    "HKK": {
        "gender": "M",
        "native-language": "Korean",
    },
    "YDCK": {
        "gender": "F",
        "native-language": "Korean",
    },
    "YKWK": {
        "gender": "M",
        "native-language": "Korean",
    },
    "EBVS": {
        "gender": "M",
        "native-language": "Spanish",
    },
    "ERMS": {
        "gender": "M",
        "native-language": "Spanish",
    },
    "MBMPS": {
        "gender": "F",
        "native-language": "Spanish",
    },
    "NJS": {
        "gender": "F",
        "native-language": "Spanish",
    },
    "HQTV": {
        "gender": "M",
        "native-language": "Vietnamese",
    },
    "PNV": {
        "gender": "F",
        "native-language": "Vietnamese",
    },
    "THV": {
        "gender": "F",
        "native-language": "Vietnamese",
    },
    "TLV": {
        "gender": "M",
        "native-language": "Vietnamese",
    },
}

if __name__ == "__main__":
    suitcase = L2ArcticDataset(
        "suitcase_corpus",
        include_speaker_info=True,
        include_text=False,
        include_timestamps=True,
    )
    scripted = all_arctic_speaker_splits(
        include_speaker_info=True,
        include_timestamps=True,
    )
    interactive_flag_samples(scripted)
    interactive_flag_samples(suitcase)
