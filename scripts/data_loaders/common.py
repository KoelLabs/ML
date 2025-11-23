import os
import sys

from torch.utils.data import Dataset
from abc import abstractmethod, ABCMeta
from collections import Iterable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE


def show_sample(sample, skip_plot=False, format_metadata=lambda x: x):
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display

    ipa, audio, *metadata = sample

    print("IPA:", ipa)
    if metadata:
        print("Metadata:", *format_metadata(metadata))
    if not skip_plot:
        plt.plot(audio)
        plt.show()
    display(Audio(audio, rate=TARGET_SAMPLE_RATE))


def show_hf_sample(sample, skip_plot=False):
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display

    sample = sample.copy()
    ipa = sample.pop("ipa")
    audio = sample.pop("audio")

    print("IPA:", ipa)
    print("Metadata:", sample)
    if not skip_plot:
        plt.plot(audio["array"])
        plt.show()
    display(Audio(audio["array"], rate=audio["sampling_rate"]))


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        split="train",
        include_timestamps=False,
        include_speaker_info=False,
        include_text=False,
        include_ambiguous_flags=False,  # true if a phone has different interpretations
    ):
        self.split = split
        self.include_timestamps = include_timestamps
        self.include_speaker_info = include_speaker_info
        self.include_text = include_text
        self.include_ambiguous_flags = include_ambiguous_flags
        self.vocab: "set | None" = None

    def __len__(self):
        return 0

    @abstractmethod
    def _get_ix(self, ix):
        pass

    def _get_ix_safe(self, ix):
        if ix >= len(self) or ix < 0:
            raise IndexError(f"Index {ix} out of bounds for dataset")
        try:
            return self._get_ix(ix)
        except IndexError as e:
            # Re-throw index errors for valid indices as a different type to avoid silently failing
            raise AssertionError(f"Sample {ix} has an internal index error", e)

    def __getitem__(self, index):
        if isinstance(index, Iterable):
            return [self._get_ix_safe(ix) for ix in index]
        elif isinstance(index, slice):
            return [self._get_ix_safe(ix) for ix in range(*index.indices(len(self)))]
        else:
            return self._get_ix_safe(index)


def split_utterance_into_multiple(
    timestamped_phonemes,
    audio,
    split_convos_at_silence_seconds: float = 2,
    min_speech_seconds: float = 1,
    min_ipa_length: int = 5,
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
                if min_speech_seconds < speech and len(speech_phones) >= min_ipa_length:
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


def split_iterator(dataset, split_config: "tuple[float, float, int]"):
    i = 0
    for sample in dataset:
        for subsample in split_utterance_into_multiple(
            sample[2], sample[1], split_config[0], split_config[1], split_config[2]
        ):
            yield i, subsample
            i += 1


def interactive_flag_samples(
    dataset,
    progress_file=os.path.join(
        os.path.dirname(__file__), "..", "..", ".data", "interactive_progress.txt"
    ),
    split_config: "None | tuple[float, float, int]" = None,
):
    import json
    from core.audio import (
        audio_resample,
        audio_array_pitchshift,
        audio_array_play,
        audio_array_to_wav_file,
        TARGET_SAMPLE_RATE,
    )

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {}

    progress_key = f"{dataset.__class__.__name__}_{dataset.split}"
    if split_config is not None:
        progress_key += f"_{split_config}"
    if not progress_key in progress:
        progress[progress_key] = {"marked": [], "cur_ix": 0}

    speedup_factor = float(input("Enter speedup factor as float: "))

    marked = progress[progress_key]["marked"]
    data_iter = (
        enumerate(dataset)
        if split_config is None
        else split_iterator(dataset, split_config)
    )
    for i, sample in data_iter:
        if i < progress[progress_key]["cur_ix"]:
            continue

        print("Sample", i, sample)
        audio = audio_array_pitchshift(
            audio_resample(
                sample[1], TARGET_SAMPLE_RATE, int(TARGET_SAMPLE_RATE / speedup_factor)
            ),
            1 / speedup_factor,
        )
        inp = "r"
        while "r" in inp:
            audio_array_play(audio)
            inp = (
                input(
                    "Enter for good, n + enter for bad, r + enter for replay, s + path + enter to save audio file: "
                )
                .lower()
                .strip()
            )
            if inp.startswith("s"):
                path = inp.removeprefix("s").strip() or "test.wav"
                assert len(path) > 4
                audio_array_to_wav_file(audio, path)
                inp = "r"
            elif "n" in inp:
                marked.append(i)

        progress[progress_key]["cur_ix"] = i + 1
        with open(progress_file, "w") as f:
            json.dump(progress, f)

    print("Marked", marked)
    return marked
