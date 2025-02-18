from torch.utils.data import Dataset
from abc import abstractmethod, ABCMeta
from collections import Iterable
import numpy as np

WAV_HEADER_SIZE = 44
TARGET_SAMPLE_RATE = 16000


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


def audio_from_bytes(data, src_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    audio = np.frombuffer(data, dtype=np.int16)[WAV_HEADER_SIZE // 2 :]
    if src_sample_rate != target_sample_rate:
        audio = np.interp(
            np.linspace(
                0,
                len(audio),
                int(len(audio) * target_sample_rate / src_sample_rate),
            ),
            np.arange(len(audio)),
            audio,
        ).astype(np.int16)
    return audio


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self, split="train", include_timestamps=False, include_speaker_info=False
    ):
        self.split = split
        self.include_timestamps = include_timestamps
        self.include_speaker_info = include_speaker_info

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
