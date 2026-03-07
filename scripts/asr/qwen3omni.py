# Docs: https://github.com/QwenLM/Qwen3-Omni
# ASR Example Code: https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/speech_recognition.ipynb

import os, sys

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"


def _load_model_processor():
    model = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=max(torch.cuda.device_count(), 1),
        limit_mm_per_prompt={"image": 1, "video": 3, "audio": 3},
        max_num_seqs=1,
        max_model_len=32768,
        max_num_batched_tokens=8192,
        seed=1234,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def _run_model(
    model: LLM, processor: Qwen3OmniMoeProcessor, messages, use_audio_in_video=True
):
    sampling_params = SamplingParams(
        temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(  # type: ignore
        messages, use_audio_in_video=use_audio_in_video
    )
    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
    }
    if images is not None:
        inputs["multi_modal_data"]["image"] = images
    if videos is not None:
        inputs["multi_modal_data"]["video"] = videos
    if audios is not None:
        inputs["multi_modal_data"]["audio"] = audios
    outputs = model.generate(inputs, sampling_params=sampling_params)  # type: ignore
    response = outputs[0].outputs[0].text
    return response, None


model = None
processor = None


def qwen_transcribe_from_file(input_path: str):
    assert model and processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": input_path},
                {"type": "text", "text": "Transcribe the English audio into text."},
            ],
        }
    ]
    response, _ = _run_model(model=model, messages=messages, processor=processor)
    return response


def main(args):
    global model, processor

    model, processor = _load_model_processor()

    if len(args) < 1:
        print("Usage: python ./scripts/asr/qwen3omni.py <audio file>")
        print("Usage: python ./scripts/asr/qwen3omni.py mic")
        return

    input_path = args[0]
    if input_path == "mic":
        pass
    else:
        print(qwen_transcribe_from_file(input_path))


if __name__ == "__main__":
    main(sys.argv[1:])
