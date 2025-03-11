# meta data (csv with transcript) located: ../../data/.repos/Kokoro-Speech-Dataset/output/metadata.csv
# wav paths located: ../../data/.repos/Kokoro-Speech-Dataset/output/wavs
# split each line in meta data by the pipe character "|" and store the first part in the "file" column and the second part in the "transcript" column, third part can be ignored

# then convert the japanese part of the transcript to phonemes using misaki package 

import os
import sys

from .common import BaseDataset

import zipfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_resample
from core.codes import parse_timit

SOURCE_SAMPLE_RATE = 22050


# resample the audio to 16000 Hz

class KokoraDataset(BaseDataset):
    """
    Metadata is provided in metadata.csv. This file consists of one record per line, delimited by the pipe character (0x7c). The fields are:

    ID: this is the name of the corresponding .wav file
    Transcription: Kanji-kana mixture text spoken by the reader (UTF-8)
    Reading: Romanized text spoken by the reader (UTF-8)

    We specifically use the tiny dataset size
    Total clips: 308
    Min duration: 3.030 secs
    Max duration: 8.092 secs
    Mean duration: 4.695 secs
    Total duration: 00:24:05
    """