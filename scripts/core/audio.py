#!/usr/bin/env python3

# Audio processing utilities
# Convert between audio formats, play audio, record audio, etc.

import ffmpeg
import sys
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import requests
import os
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.load_secrets import load_secrets

load_secrets()

WAV_HEADER_SIZE = 44
TARGET_SAMPLE_RATE = 16000


def audio_convert(input_path, output_path, output_sample_rate=TARGET_SAMPLE_RATE):
    ffmpeg.input(input_path).output(output_path, ar=output_sample_rate).run()


def audio_bytes_to_wav_array(bytes, format, output_sample_rate=TARGET_SAMPLE_RATE):
    wav_bytes = (
        ffmpeg.input("pipe:0", format=format)
        .output("pipe:1", format="wav", ar=output_sample_rate)
        .run(input=bytes, capture_stdout=True)
    )
    return audio_bytes_to_array(wav_bytes[0], output_sample_rate)


def audio_array_to_bytes(array, sample_rate=TARGET_SAMPLE_RATE):
    with BytesIO() as f:
        wavfile.write(f, sample_rate, array)
        return f.getvalue()


def audio_array_to_wav_file(
    input_array, output_path, output_sample_rate=TARGET_SAMPLE_RATE
):
    wavfile.write(output_path, output_sample_rate, input_array)


def audio_array_pitchshift(
    x: np.ndarray, pitch_ratio: float, mode: str = "tune", window_size: int = 2048
) -> np.ndarray:
    """
    Pitch-shift (or time-stretch) a 16-bit single-channel signal in memory.
    Adapted from https://github.com/haoyu987/phasevocoder

    Parameters
    ----------
    x           : np.ndarray[int16]
                  Input PCM signal (shape (N,), dtype=int16).
    pitch_ratio : float
                  Ratio >1.0 shifts up, <1.0 shifts down.
    mode        : {'tune','stretch'}
                  'tune'  = pitch-shift (with resampling)
                  'stretch' = time-stretch only
    window_size : int
                  FFT window size (power of 2), default 2048.

    Returns
    -------
    y : np.ndarray[int16]
        Output PCM signal, dtype=int16.
    """
    assert x.dtype == np.int16 and x.ndim == 1

    def resample(signal):
        output_Length = int((len(signal) - 1) / pitch_ratio)
        output = np.zeros(output_Length)
        for i in range(output_Length - 1):
            x = float(i * pitch_ratio)
            ix = int(np.floor(x))
            dx = x - ix
            output[i] = signal[ix] * (1.0 - dx) + signal[ix + 1] * dx
        return output

    def locate_peaks(signal):
        # function to find peaks
        # a peak is any sample which is greater than its two nearest neighbours
        index = 0
        k = 2
        indices = []
        while k < len(signal) - 2:
            seg = signal[k - 2 : k + 3]
            if np.amax(seg) < 150:
                k = k + 2
            else:
                if seg.argmax() == 2:
                    indices.append(k)
                    index = index + 1
                    k = k + 2
            k += 1
        return indices

    # hop sizes
    synthesis_hop = window_size // 4
    analysis_hop = int(synthesis_hop / pitch_ratio)

    # zero‑pad so that last frame is complete
    n_in = len(x)
    n_pad = (window_size - (n_in - window_size) % analysis_hop) % analysis_hop
    x_padded = np.concatenate([x.astype(float), np.zeros(n_pad)])

    # prepare STFT buffers
    win = np.hanning(window_size)
    num_bins = window_size // 2 + 1
    last_phase = np.zeros(num_bins)
    accum_phase = np.zeros(num_bins)
    expected_phi = (
        np.linspace(0, num_bins - 1, num_bins) * 2 * np.pi * analysis_hop / window_size
    )

    # output length estimate
    est_frames = 1 + (len(x_padded) - window_size) // analysis_hop
    y_len = est_frames * synthesis_hop + window_size
    y = np.zeros(y_len)

    read_pt = 0
    write_pt = 0
    AmpMax = 2**15 - 1

    # initial dummy peaks
    pk_indices = list(range(num_bins))

    while read_pt + window_size <= len(x_padded):
        frame = x_padded[read_pt : read_pt + window_size] * win
        # FFT
        X = np.fft.rfft(frame)
        mag = np.abs(X)
        phase = np.angle(X)

        # phase difference
        delta = phase - last_phase
        last_phase = phase.copy()
        delta -= expected_phi
        delta = np.unwrap(delta)

        # accumulate true phase
        accum_phase += (delta + expected_phi) * (synthesis_hop / analysis_hop)

        # region‑of‑influence re‑assignment around peaks
        # compute rotation at peaks
        rotation = accum_phase[pk_indices] - phase[pk_indices]
        # for each region between peaks, set accum_phase
        starts = [0] + [
            (pk_indices[i] + pk_indices[i + 1]) // 2 for i in range(len(pk_indices) - 1)
        ]
        ends = [
            (pk_indices[i] + pk_indices[i + 1]) // 2 for i in range(len(pk_indices) - 1)
        ] + [num_bins]
        for r, pk in enumerate(pk_indices):
            # for bins in region around this peak
            for b in range(starts[r], ends[r]):
                accum_phase[b] = rotation[r] + phase[b]

        # detect new peaks for next iteration
        new_pk = locate_peaks(mag)
        pk_indices = new_pk if new_pk else [1]

        # inverse FFT with modified phase
        Y = mag * (np.cos(accum_phase) + 1j * np.sin(accum_phase))
        y_frame = np.fft.irfft(Y)

        # overlap‑add
        y[write_pt : write_pt + window_size] += y_frame * win

        read_pt += analysis_hop
        write_pt += synthesis_hop

    # trim to actual written
    y = y[:write_pt]

    # clip and convert back to int16
    y = np.clip(y, -AmpMax, AmpMax)
    y = y.astype(np.int16)

    if mode == "tune":
        # final resample to correct pitch
        y = resample(y).astype(np.int16)
    elif mode != "stretch":
        raise ValueError("mode must be 'tune' or 'stretch'")

    return y


def audio_resample(array, src_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    if src_sample_rate == target_sample_rate:
        return array
    return np.interp(
        np.linspace(
            0,
            len(array),
            int(len(array) * target_sample_rate / src_sample_rate),
        ),
        np.arange(len(array)),
        array,
    ).astype(np.int16)


def audio_bytes_to_array(
    data, src_sample_rate=None, target_sample_rate=TARGET_SAMPLE_RATE
):
    # TODO: rename to make clear this requires WAV format
    assert data[:4] == b"RIFF", "Not a WAV file, first 4 bytes are not RIFF: " + data[
        :4
    ].decode("utf-8")
    if src_sample_rate == None:
        # read 32 bit integer from bytes 25-28 in header
        src_sample_rate = int.from_bytes(data[24:28], byteorder="little")
    # read bits per sample from bytes 35-36 in header
    bits_per_sample = int.from_bytes(data[34:36], byteorder="little")
    dtype = np.int16 if bits_per_sample == 16 else np.int32
    # read number of channels from bytes 23-24 in header
    num_channels = int.from_bytes(data[22:24], byteorder="little")
    data = data[WAV_HEADER_SIZE:]
    audio = np.frombuffer(data, dtype=dtype).astype(np.int16)
    # average in chunks of num_channels
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels)
        audio = np.mean(audio, axis=1).astype(np.int16)
    audio = audio_resample(audio, src_sample_rate, target_sample_rate)
    return audio


def audio_dual_channel_to_mono(input_array):
    if input_array.ndim == 2 and input_array.shape[1] == 2:
        return np.mean(input_array, axis=1).astype(np.int16)
    return input_array


def audio_file_to_array(input_path, desired_sample_rate=TARGET_SAMPLE_RATE):
    rate, data = wavfile.read(input_path)
    data = audio_dual_channel_to_mono(data)
    data = audio_resample(data, rate, desired_sample_rate)
    return data


def audio_array_play(input_array, sample_rate=TARGET_SAMPLE_RATE):
    sd.play(input_array, sample_rate)
    sd.wait()


def audio_wav_file_play(input_path, start_sec=None, end_sec=None):
    print(start_sec, end_sec)
    rate, data = wavfile.read(input_path)
    start = int(float(start_sec) * rate) if start_sec else 0
    end = int(float(end_sec) * rate) if end_sec else len(data)
    data = data[start:end]
    audio_array_play(data, rate)


def audio_wav_file_crop(input_path, start_sec, end_sec, output_path):
    rate, data = wavfile.read(input_path)
    start = int(float(start_sec) * rate)
    end = int(float(end_sec) * rate)
    data = data[start:end]
    audio_array_to_wav_file(data, output_path, rate)


def audio_record_to_array(output_sample_rate=TARGET_SAMPLE_RATE):
    print("Recording, please speak and press Ctrl+C when done")
    samples = np.array([], dtype=np.int16)
    try:
        with sd.InputStream(
            channels=1, dtype="int16", samplerate=output_sample_rate
        ) as s:
            while True:
                sample, _ = s.read(output_sample_rate)
                samples = np.append(samples, sample.reshape(-1))
    except KeyboardInterrupt:
        print("Recording stopped")
    return samples


def audio_record_to_file(output_path, output_sample_rate=TARGET_SAMPLE_RATE):
    samples = audio_record_to_array(output_sample_rate)
    audio_array_to_wav_file(samples, output_path, output_sample_rate)


def audio_file_from_text(text, output_path):
    assert os.environ.get("GOOEY_API_KEY"), "GOOEY_API_KEY environment variable not set"

    response = requests.post(
        "https://api.gooey.ai/v2/TextToSpeech",
        headers={
            "Authorization": "bearer " + os.environ["GOOEY_API_KEY"],
        },
        json={
            "text_prompt": text,
            "tts_provider": "AZURE_TTS",
            "azure_voice_name": "en-US-KaiNeural",
        },
    )
    assert response.ok, response.content

    result = response.json()
    audio_url = result["output"]["audio_url"]
    response = requests.get(audio_url)
    with open(output_path, "wb") as f:
        f.write(response.content)


def audio_array_float64_to_int16(audio_data_float64):
    """Converts a float64 numpy array to int16."""

    # Normalize to the range -1.0 to 1.0 (if needed)
    max_value = np.max(np.abs(audio_data_float64))
    if max_value > 1.0:
        audio_data_float64 = audio_data_float64 / max_value

    # Scale to int16 range
    audio_data_int16 = (audio_data_float64 * 32767).astype(np.int16)

    return audio_data_int16


def main(args):
    if args[0] == "record":
        audio_record_to_file(args[1])
    elif args[0] == "convert":
        if len(args) > 3:
            audio_convert(args[1], args[2], int(args[3]))
        else:
            audio_convert(args[1], args[2])
    elif args[0] == "play":
        start, end = None, None
        if len(args) > 2:
            start, end = args[2].split(":")
            start, end = float(start), float(end)
        audio_wav_file_play(args[1], start, end)
    elif args[0] == "crop":
        audio_wav_file_crop(args[1], float(args[2]), float(args[3]), args[4])
    elif args[0] == "text":
        audio_file_from_text(args[1], args[2])
    elif args[0] == "speed":
        speed_factor = float(args[2])
        audio = audio_file_to_array(args[1])
        audio_shifted = audio_array_pitchshift(audio, 1 / speed_factor, mode="stretch")
        if args[3] == "mic":
            audio_array_play(audio_shifted)
        else:
            audio_array_to_wav_file(audio_shifted, args[3])
    else:
        print("Invalid command")
        print("Usage: python ./scripts/core/audio.py record <output_wav_path>")
        print("Usage: python ./scripts/core/audio.py play <input_wav_path> [start:end]")
        print(
            "Usage: python ./scripts/core/audio.py convert <input_path> <output_path>"
        )
        print(
            "Usage: python ./scripts/core/audio.py convert <input_path> <output_path> <output_sample_rate>"
        )
        print("Usage: python ./scripts/core/audio.py text <text> <output_wav_path>")
        print(
            "Usage: python ./scripts/core/audio.py crop <input_wav_path> <start> <end> <output_wav_path>"
        )
        print(
            "Usage: python ./scripts/core/audio.py speed <input_wav_path> <speed_factor> <output_wav_path>"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
