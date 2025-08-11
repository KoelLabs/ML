import numpy as np

from queue import Queue
from threading import Thread, Event

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import audio_file_to_array, audio_stream_microphone, TARGET_SAMPLE_RATE

import time
from typing import Union, Literal

TRANSCRIPT_T = list[tuple[str, float, float]]
SEND_EVENT_T = tuple[float, TRANSCRIPT_T]
RECEIVE_EVENT_T = tuple[float, int]


class Source:
    def __init__(self, receive):
        self._receive = receive
        self.send_log: list[SEND_EVENT_T] = []
        self.receive_log: list[RECEIVE_EVENT_T] = []
        self.waiting_time = 0

    def receive(
        self,
        format: Union[Literal["int16"], Literal["float32"], Literal["bytes"]] = "bytes",
    ) -> Union[np.ndarray, Literal["stop"]]:
        value, waiting_time = self._receive()
        self.waiting_time += waiting_time
        if isinstance(value, str) and value == "stop":
            return value
        self.receive_log.append((time.perf_counter(), len(value)))
        if format == "float32":
            return value.astype(np.float32) / 32768  # type: ignore
        if format == "bytes":
            return (value.astype(np.float32) / 32768).tobytes()  # type: ignore
        return value  # type: ignore

    def send(self, transcript: TRANSCRIPT_T):
        self.send_log.append((time.perf_counter(), transcript.copy()))

    def evaluate(self):
        # Account for cheated time, i.e., we allow consuming the stream (in the case of file/array sources) faster than the realtime playback of the audio
        # (this is to make evals faster), so we need to add back in the time that would be spent waiting for audio in a realtime streaming scenario (say from the mic)
        adjusted_receive_log = []
        adjusted_send_log = [list(t) for t in self.send_log]
        audio_time_sec = 0
        time_sec = 0
        offset_sec = 0
        for raw_time_sec, samples in self.receive_log:
            audio_time_sec += samples / TARGET_SAMPLE_RATE
            time_sec = (
                raw_time_sec
                - self.receive_log[0][0]
                + self.receive_log[0][1] / TARGET_SAMPLE_RATE
                + offset_sec
            )
            if audio_time_sec > time_sec:
                off = audio_time_sec - time_sec
                for i, (event_time_sec, _) in enumerate(self.send_log):
                    if event_time_sec >= raw_time_sec:
                        adjusted_send_log[i][0] += off  # type: ignore
                offset_sec += off
            adjusted_receive_log.append((raw_time_sec + offset_sec, samples))
        self.receive_log = adjusted_receive_log
        self.send_log = [tuple(t) for t in adjusted_send_log]  # type: ignore

        # Compute stats
        audio_duration = sum(s for _, s in self.receive_log) / TARGET_SAMPLE_RATE
        stream_start = (
            self.receive_log[0][0] - self.receive_log[0][1] / TARGET_SAMPLE_RATE
        )
        stream_end = self.receive_log[-1][0]
        stream_duration = stream_end - stream_start
        computation_time = stream_duration - self.waiting_time - offset_sec

        warmup_latency = self.send_log[0][0] - stream_start

        average_first_guess_latency = 0
        prev_transcript_time_sec = 0
        for event_time_sec, transcript in self.send_log:
            stream_time_sec = event_time_sec - stream_start
            transcript_time_sec = transcript[-1][2]
            new_transcript_time_sec = transcript_time_sec - prev_transcript_time_sec
            latency = stream_time_sec - transcript_time_sec
            average_first_guess_latency += latency * new_transcript_time_sec
            prev_transcript_time_sec = transcript_time_sec
        average_first_guess_latency /= audio_duration

        average_final_guess_latency = 0
        for i, (p, _, audio_time_sec) in enumerate(self.send_log[-1][1]):
            final_time_sec = None
            for event_time_sec, transcript in self.send_log:
                if len(transcript) - 1 < i or transcript[i][0] != p:
                    final_time_sec = None
                elif final_time_sec is None:
                    final_time_sec = event_time_sec - stream_start
            assert final_time_sec is not None
            average_final_guess_latency += final_time_sec - audio_time_sec
        average_final_guess_latency /= len(self.send_log[-1][1])

        return (
            audio_duration,
            stream_duration,
            computation_time,
            warmup_latency,
            average_first_guess_latency,
            average_final_guess_latency,
        )


def run_source_from_callback_provider(callback_provider, timeout=None):
    q = Queue()
    stop_event = Event()

    def task():
        callback_provider(q.put, stop_event)
        q.put("stop")

    def receive() -> tuple[np.ndarray, float]:
        try:
            start = time.perf_counter()
            value = q.get(block=True, timeout=timeout)
            waiting_time = time.perf_counter() - start
            return value, waiting_time
        except KeyboardInterrupt:
            stop_event.set()
            return np.array([], dtype=np.int16), 0

    Thread(target=task, daemon=True).start()
    return Source(receive)


def run_microphone_source(
    block_size=512, sample_rate=TARGET_SAMPLE_RATE, max_listen_seconds=60, timeout=None
):
    def sleep_until_interrupted(stop_event):
        for _ in range(max_listen_seconds):
            if stop_event.is_set():
                return
            time.sleep(1)

    return run_source_from_callback_provider(
        lambda c, e: audio_stream_microphone(
            on_block=c,
            block_size=block_size,
            sample_rate=sample_rate,
            timeout=lambda: sleep_until_interrupted(e),
        ),
        timeout=timeout,
    )


def run_array_source(
    wav_array: np.ndarray, block_size=512, timeout=None, slow_down_to_realtime=False
):
    start = time.time() if slow_down_to_realtime else 0

    def chunker(callback, stop_event):
        for i in range(0, len(wav_array), block_size):
            time_sec = (i + block_size) / TARGET_SAMPLE_RATE
            while slow_down_to_realtime and time.time() - start < time_sec:
                time.sleep(time_sec - time.time() + start)
            callback(wav_array[i : i + block_size])
            if stop_event.is_set():
                return

    return run_source_from_callback_provider(chunker, timeout=timeout)


def run_file_source(
    wav_path: str,
    sample_rate=TARGET_SAMPLE_RATE,
    block_size=512,
    timeout=None,
    slow_down_to_realtime=False,
):
    return run_array_source(
        audio_file_to_array(wav_path, sample_rate),  # type: ignore
        block_size=block_size,
        timeout=timeout,
        slow_down_to_realtime=slow_down_to_realtime,
    )
