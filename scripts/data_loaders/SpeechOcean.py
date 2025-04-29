# Native Mandarin speakers in English, half children/half adults: https://github.com/jimbozhang/speechocean762

import os
import sys

from datasets import load_dataset, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import TARGET_SAMPLE_RATE
from core.codes import arpabet2ipa, IPA2ARPABET


HUGGING_FACE_ID = "mispeech/speechocean762"


class SpeechOceanDataset(BaseDataset):
    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        # filters
        discard_heavy_accents=False,  # if True, discard samples marked as "heavy accent"
        discard_approximate_labels=False,  # if True, discard samples with uncertain sound labels
        discard_unknown_labels=True,  # if True, discard samples where not all sounds are discernible
    ):
        """
        Valid splits are "train" and "test"
        """
        super().__init__(split, include_timestamps, include_speaker_info, include_text)

        assert not include_timestamps, "No timestamp information"

        self.dataset: Dataset = load_dataset("mispeech/speechocean762", split=split)  # type: ignore

        remove_ixs = []
        for ix, sample in enumerate(self.dataset):  # type: ignore
            sample: dict
            for w in sample["words"]:
                for mis in w["mispronunciations"]:
                    w["phones"][mis["index"]] = mis["pronounced-phone"]
                    if discard_approximate_labels and mis["pronounced-phone"].endswith(
                        "*"
                    ):
                        remove_ixs.append(ix)
                        break
                    if discard_unknown_labels and mis["pronounced-phone"] == "<unk>":
                        remove_ixs.append(ix)
                        break
                else:
                    break
                if discard_heavy_accents and min(w["phones-accuracy"]) < 1.0:
                    remove_ixs.append(ix)
                    break
        if len(remove_ixs) > 0:
            self.dataset = self.dataset.filter(
                lambda _, ix: ix not in remove_ixs, with_indices=True
            )

        self.vocab = set(IPA2ARPABET.keys())

    def __len__(self):
        return len(self.dataset)

    def _get_ix(self, ix):
        sample = self.dataset[ix]

        audio = sample["audio"]["array"]
        assert (
            sample["audio"]["sampling_rate"] == TARGET_SAMPLE_RATE
        ), "Please call the resample util to match target sample rate"

        words = sample["words"]
        phones = [arpabet2ipa(p) for w in words for p in w["phones"]]
        ipa = "".join(phones)

        result = [ipa, audio]
        if self.include_speaker_info:
            result.append(
                {
                    "speaker_id": sample["speaker"],
                    "gender": sample["gender"],
                    "age": sample["age"],
                    "accuracy": sample[
                        "accuracy"
                    ],  # how good the overall pronunciation is, 0-10
                    "completeness": sample[
                        "completeness"
                    ],  # fraction of "good" pronounciation words, 0.0-1.0
                    "fluency": sample["fluency"],  # how few pauses/stammering, 0-10
                    "prosodic": sample[
                        "prosodic"
                    ],  # correctness of intonation and cadence, 0-10
                }
            )
        if self.include_text:
            result.append(sample["text"])
        return tuple(result)


if __name__ == "__main__":
    dataset = SpeechOceanDataset(include_speaker_info=True, include_text=True)
    print(len(dataset))
    print(dataset[0])
