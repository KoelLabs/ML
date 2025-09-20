import os
import sys
import glob
import re

# go back two directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.core.audio import audio_bytes_to_array
from scripts.data_loaders.common import BaseDataset

SPEAKERS = {
    "aew": {"sex": "male", "lang": "US English", "accent": "US"},
    "ahw": {"sex": "male", "lang": "US English", "accent": "German"},
    "aup": {"sex": "male", "lang": "US English", "accent": "Indian"},
    "awb": {"sex": "male", "lang": "US English", "accent": "Scottish"},
    "axb": {"sex": "female", "lang": "US English", "accent": "Indian"},
    "bdl": {"sex": "male", "lang": "US English", "accent": "US"},
    "clb": {"sex": "female", "lang": "US English", "accent": "US"},
    "eey": {"sex": "female", "lang": "US English", "accent": "US"},
    "fem": {"sex": "male", "lang": "US English", "accent": "Irish"},
    "gka": {"sex": "male", "lang": "US English", "accent": "Indian"},
    "jmk": {"sex": "male", "lang": "US English", "accent": "Canadian"},
    "ksp": {"sex": "male", "lang": "US English", "accent": "Indian"},
    "ljm": {"sex": "female", "lang": "US English", "accent": "US"},
    "lnh": {"sex": "female", "lang": "US English", "accent": "US"},
    "rms": {"sex": "male", "lang": "US English", "accent": "US"},
    "rxr": {"sex": "male", "lang": "US English", "accent": "Dutch"},
    "slp": {"sex": "female", "lang": "US English", "accent": "Indian"},
    "slt": {"sex": "female", "lang": "US English", "accent": "US"},
}


class L1ArcticDataset(BaseDataset):
    """
    Dataset class for CMU ARCTIC corpus that loads audio data and corresponding text
    """

    def __init__(
        self,
        data_dir=".data/CMU_ARCTIC",
        include_speaker_info=False,
        include_text=True,
        speaker_list=None,
    ):
        self.data_dir = data_dir
        self.include_speaker_info = include_speaker_info
        self.include_text = include_text

        # Use specific speakers or all available ones
        self.speaker_list = speaker_list if speaker_list else list(SPEAKERS.keys())

        # Build data index
        self._build_index()

    def _build_index(self):
        """Build an index of all audio files and their corresponding text"""
        self.data_samples = []

        # Process each speaker directory
        for speaker in self.speaker_list:
            speaker_dir = os.path.join(self.data_dir, f"cmu_us_{speaker}_arctic")

            # Skip if speaker directory doesn't exist
            if not os.path.exists(speaker_dir):
                print(f"Warning: Speaker directory {speaker_dir} not found. Skipping.")
                continue

            # Load the text file containing utterance information
            text_file = os.path.join(speaker_dir, "etc", "txt.done.data")
            if not os.path.exists(text_file):
                print(
                    f"Warning: Text file {text_file} not found. Skipping speaker {speaker}."
                )
                continue

            # Parse text data file
            utterance_texts = {}
            with open(text_file, "r", encoding="utf-8") as f:
                for line in f:
                    # Format is: ( arctic_a0001 "Text of the utterance." )
                    match = re.match(r'\(\s*(\S+)\s+"(.+)"\s*\)', line.strip())
                    if match:
                        utterance_id, text = match.groups()
                        utterance_texts[utterance_id] = text

            # Find all wav files for this speaker
            wav_dir = os.path.join(speaker_dir, "wav")
            if not os.path.exists(wav_dir):
                print(
                    f"Warning: Wav directory {wav_dir} not found. Skipping speaker {speaker}."
                )
                continue

            wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))[
                :TOTAL_UTTERANCES_PER_SPEAKER
            ]
            # Add each wav file and its corresponding text to the index
            for wav_path in wav_files:
                filename = os.path.basename(wav_path)
                utterance_id = os.path.splitext(filename)[0]

                # Only include samples that have text (skip those without text)
                if utterance_id in utterance_texts:
                    self.data_samples.append(
                        {
                            "wav_path": wav_path,
                            "text": utterance_texts[utterance_id],
                            "speaker": speaker,
                            "utterance_id": utterance_id,
                        }
                    )
                else:
                    print(
                        f"Warning: No text found for {utterance_id} of speaker {speaker} - skipping this sample"
                    )

        print(
            f"Loaded {len(self.data_samples)} samples from {len(self.speaker_list)} speakers"
        )

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data_samples)

    def _get_ix(self, idx):
        """Get a sample from the dataset"""
        sample = self.data_samples[idx]

        with open(sample["wav_path"], "rb") as f:
            audio = audio_bytes_to_array(f.read())

        result = [None, audio]
        if self.include_text:
            result.append(sample["text"])
        if self.include_speaker_info:
            speaker_info = SPEAKERS[sample["speaker"]]
            result.append(speaker_info)
            result.append(sample["speaker"])

        return tuple(result)


# Example usage
if __name__ == "__main__":
    # Create the dataset with all speakers
    dataset = L1ArcticDataset(
        data_dir=".data/CMU_ARCTIC", include_speaker_info=True, include_text=True
    )

    # Get the first sample
    sample = dataset[0]

    # Print information about the sample
    _, audio, text, speaker_info = sample  # type: ignore
    print(f"Audio shape: {audio.shape}")  # type: ignore
    print(f"Speaker info: {speaker_info}")
    print(f"Text: {text}")

    # Example of getting a specific speaker
    bdl_dataset = L1ArcticDataset(
        data_dir=".data/CMU_ARCTIC",
        include_speaker_info=True,
        include_text=True,
        speaker_list=["bdl"],
    )

    # print(f"\nLoaded {len(bdl_dataset)} samples for speaker 'bdl'")

    if len(bdl_dataset) > 0:
        bdl_sample = bdl_dataset[0]
        _, bdl_audio, bdl_text, bdl_speaker_info = bdl_sample  # type: ignore
        print(f"BDL speaker info: {bdl_speaker_info}")
        print(f"BDL text sample: {bdl_text}")
