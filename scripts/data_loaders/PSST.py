# Post-stroke aphasia English speech from https://aphasia.talkbank.org/derived/RaPID/
# Paper: https://aclanthology.org/2022.rapid-1.6.pdf

import os
import sys

import requests
from contextlib import contextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset, interactive_flag_samples
from core.audio import audio_file_to_array, TARGET_SAMPLE_RATE
from core.codes import arpabet2ipa, IPA2ARPABET


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "psst-data")


@contextmanager
def no_internet():
    old_get, old_post = requests.get, requests.post

    def throw_http_error(*args, **kwargs):
        raise requests.HTTPError("Simulated no internet")

    requests.get = requests.post = throw_http_error
    yield
    requests.get, requests.post = old_get, old_post


class PSSTDataset(BaseDataset):
    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        force_offline=False,
    ):
        super().__init__(split, include_timestamps, include_speaker_info)

        if include_timestamps:
            raise NotImplementedError("Timestamps are not available for PSST.")

        if (
            force_offline
        ):  # often, the server is not available, so we need to force fully local mode
            with no_internet():
                import psstdata

                data = psstdata.load(local_dir=DATA_DIR, version_id="local")
        else:
            import psstdata

            data = psstdata.load(local_dir=DATA_DIR)

        if split == "train":
            self.utterances = data.train
        elif split == "valid":
            self.utterances = data.valid
        elif split == "test":
            self.utterances = data.test
        else:
            raise ValueError(f"Unknown split: {split}")

        self.vocab = set(IPA2ARPABET.keys())

    def __len__(self):
        return len(self.utterances)

    def _get_ix(self, ix):
        utterance = self.utterances[ix]

        ipa = arpabet2ipa(utterance.transcript.replace("<spn>", "").replace("<sil>", ""))  # type: ignore
        audio = audio_file_to_array(utterance.filename_absolute)  # type: ignore

        # cropping bad samples
        if self.split == "test":
            if ix == 309:  # sample contains un-annotated speaker
                # crop after 1.4 seconds
                start = int(1.4 * TARGET_SAMPLE_RATE)
                audio = audio[start:]
                ipa = "oʊoʊoʊpʌnʌɑpɝeɪtɪŋʌm"
            if ix == 50:
                # preserve first 1.8 seconds
                end = int(1.8 * TARGET_SAMPLE_RATE)
                audio = audio[:end]

        if self.include_speaker_info:
            return (
                ipa,
                audio,
                {
                    "utterance_id": utterance.utterance_id,  # type: ignore
                    "test": utterance.test,  # type: ignore
                    "session": utterance.session,  # type: ignore
                    "text_prompt": utterance.prompt,  # type: ignore
                    "correct": utterance.correctness,  # type: ignore
                    "aq_index": utterance.aq_index,  # type: ignore
                    "filename": utterance.filename_absolute,  # type: ignore
                },
            )
        else:
            return ipa, audio


if __name__ == "__main__":
    train = PSSTDataset(include_speaker_info=True, force_offline=True)
    interactive_flag_samples(train)
    valid = PSSTDataset(split="valid", include_speaker_info=True, force_offline=True)
    interactive_flag_samples(valid)
    test = PSSTDataset(split="test", include_speaker_info=True, force_offline=True)
    interactive_flag_samples(test)
