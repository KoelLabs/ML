import os
import sys

from torch.utils.data import Dataset
from abc import abstractmethod, ABCMeta
from collections import Iterable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE


def show_sample(sample):
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display

    ipa, audio, *metadata = sample

    print("IPA:", ipa)
    if metadata:
        print("Metadata:", *metadata)
    plt.plot(audio)
    plt.show()
    display(Audio(audio, rate=TARGET_SAMPLE_RATE))


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        include_g2p=False,
        g2p_filter_type="letters_rmv_tie",
    ):
        self.split = split
        self.include_timestamps = include_timestamps
        self.include_speaker_info = include_speaker_info
        self.include_text = include_text
        self.include_g2p = include_g2p
        self.g2p_filter_type = g2p_filter_type

    def __len__(self):
        return 0

    @abstractmethod
    def _get_ix(self, filename):
        pass

    def __getitem__(self, index):
        if isinstance(index, Iterable):
            return [self._get_ix(ix) for ix in index]
        elif isinstance(index, slice):
            return [self._get_ix(ix) for ix in range(*index.indices(len(self)))]
        else:
            return self._get_ix(index)
