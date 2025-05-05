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
        self.vocab: "set | None" = None

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


def split_utterance_into_multiple(
    timestamped_phonemes,
    audio,
    split_convos_at_silence_seconds: float = 2,
    min_speech_seconds: float = 1,
):
    """
    Uses the timestamped phonemes to determine the silence and speech sections.
    Splits when silence is greater than split_convos_at_silence_seconds seconds
    and there is speech for min_speech_seconds seconds.
    """
    # go through timestamped phonemes and identify silence and speech sections
    sections = []
    prev = "nothing"
    sil_start, sil_end = 0, 0
    speech_start, speech_end = 0, 0
    speech_phones = []
    for phone, start, stop in timestamped_phonemes:
        if phone == "":
            if prev != "":
                sil_start = start
            sil_end = stop
        significant_silence = False
        if prev == "" and phone != "":
            silence = (sil_end - sil_start) / TARGET_SAMPLE_RATE
            if silence > split_convos_at_silence_seconds:
                speech = (speech_end - speech_start) / TARGET_SAMPLE_RATE
                # print('speech', speech)
                # print('silence', silence)
                if min_speech_seconds < speech:
                    significant_silence = True
                    sections.append(
                        (
                            "".join([x[0] for x in speech_phones]),
                            audio[speech_start:speech_end],
                            [
                                (p, s - speech_start, e - speech_start)
                                for p, s, e in speech_phones
                            ],
                        )
                    )

        if phone != "":
            if significant_silence:
                speech_start = start
                speech_phones = []
            speech_end = stop
            speech_phones.append((phone, start, stop))

        prev = phone

    return sections
