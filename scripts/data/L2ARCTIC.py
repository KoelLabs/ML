from .common import BaseDataset, audio_from_bytes, IPA_SUBSTITUTIONS

import zipfile
import textgrids
from torch.utils.data import ConcatDataset

SOURCE_SAMPLE_RATE = 44100


def all_arctic_speaker_splits(include_timestamps=False, include_speaker_info=False):
    return ConcatDataset(
        L2ArcticDataset(
            split=s,
            include_timestamps=include_timestamps,
            include_speaker_info=include_speaker_info,
        )
        for s in SPEAKERS.keys()
    )


class L2ArcticDataset(BaseDataset):
    """
    Each speaker in the SPEAKERS dictionary is a valid split.
    These splits involve short scripted utterances of English (3.6 seconds on average) spoken by non-native speakers.
    The suitcase_corpus is a subset of L2-Arctic featuring non-native English speech in free-form/non-scripted conversations.
    This is also a valid split.
    """

    def __init__(
        self, split="ABA", include_timestamps=False, include_speaker_info=False
    ):
        super().__init__(split, include_timestamps, include_speaker_info)

        if include_timestamps:
            raise NotImplementedError("Timestamps are available but not parsed yet.")

        self.arctic = zipfile.ZipFile("../.data/l2arctic_release_v5.0.zip", "r")
        self.suitcase = self.arctic.open(f"{split}.zip")
        self.zip = zipfile.ZipFile(self.suitcase, "r")
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

    def __del__(self):
        self.zip.close()
        self.suitcase.close()
        self.arctic.close()

    def __len__(self):
        return len(self.files)

    def _get_ix(self, ix):
        filename = self.files[ix]
        with self.zip.open(f"{self.split}/wav/{filename}.wav") as wav_file:
            audio = audio_from_bytes(wav_file.read(), SOURCE_SAMPLE_RATE)

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

            arpa = tg.interval_tier_to_array("phones")
            arpa = [c["label"] for c in arpa]
            arpa = [c.split(",")[1].strip() if "," in c else c for c in arpa]
            # remove numbers (since they indicate stress)
            arpa = ["".join([c for c in p if not c.isdigit()]) for p in arpa]
            arpa = [c.split(" ")[0] for c in arpa]
            ipa = "".join([ARPABET2IPA.get(c, "") for c in arpa])

        if self.include_speaker_info:
            speaker_id = self.split if self.split != "suitcase_corpus" else filename
            speaker = SPEAKERS[speaker_id.upper()]
            return ipa, audio, speaker
        else:
            return ipa, audio


ARPABET2IPA = {'AA':'ɑ','AE':'æ','AH':'ʌ','AH0':'ə','AO':'ɔ','AW':'aʊ','AY':'aɪ','EH':'ɛ','ER':'ɝ','ER0':'ɚ','EY':'eɪ','IH':'ɪ','IH0':'ɨ','IY':'i','OW':'oʊ','OY':'ɔɪ','UH':'ʊ','UW':'u','B':'b','CH':'tʃ','D':'d','DH':'ð','EL':'l̩','EM':'m̩','EN':'n̩','F':'f','G':'ɡ','HH':'h','JH':'dʒ','K':'k','L':'l','M':'m','N':'n','NG':'ŋ','P':'p','Q':'ʔ','R':'ɹ','S':'s','SH':'ʃ','T':'t','TH':'θ','V':'v','W':'w','WH':'ʍ','Y':'j','Z':'z','ZH':'ʒ'}  # fmt: skip
for k in ARPABET2IPA.keys():
    if ARPABET2IPA[k] in IPA_SUBSTITUTIONS:
        ARPABET2IPA[k] = IPA_SUBSTITUTIONS[ARPABET2IPA[k]]

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
