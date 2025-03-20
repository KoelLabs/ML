# Post-stroke aphasia English speech from https://aphasia.talkbank.org/derived/RaPID/

import os
import sys

import requests
from contextlib import contextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from core.audio import audio_file_to_array
from core.codes import arpabet2ipa

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
    def __init__(self, split="train", include_timestamps=False, include_speaker_info=False, force_offline=False):
        super().__init__(split, include_timestamps, include_speaker_info)

        if include_timestamps:
            raise NotImplementedError("Timestamps are available but not parsed yet.")

        if force_offline: # often, the server is not available, so we need to force fully local mode
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

    def __len__(self):
        return len(self.utterances)

    def _get_ix(self, ix):
        utterance = self.utterances[ix]
        ipa = arpabet2ipa(utterance.transcript.replace("<spn>", "").replace("<sil>", ""))
        audio = audio_file_to_array(utterance.filename_absolute)
        if self.include_speaker_info:
            return ipa, audio, {
                "utterance_id": utterance.utterance_id,
                "test": utterance.test,
                "session": utterance.session,
                "text_prompt": utterance.prompt,
                "correct": utterance.correctness,
                "aq_index": utterance.aq_index,
            }
        else:
            return ipa, audio

if __name__ == '__main__':
    dataset = PSSTDataset(include_speaker_info=True, force_offline=True)
    print(len(dataset))
    print(dataset[0])
