import torch

# set espeak library path for macOS
import sys

if sys.platform == "darwin":
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    _ESPEAK_LIBRARY = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib"
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)


def transcribe_batch(batch, model, processor):
    input_values = (
        processor(
            [x[1] for x in batch],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return [processor.decode(ids) for ids in predicted_ids]


def transcribe_batch_filtered(batch, model, processor, vocab):
    input_values = (
        processor(
            [x[1] for x in batch],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        .input_values.type(torch.float32)
        .to(model.device)
    )
    with torch.no_grad():
        logits = model(input_values).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # probabilities[:, :, processor.tokenizer.pad_token_id] /= 1.2
    probabilities[:, :, processor.tokenizer.pad_token_id] = probabilities[
        :, :, processor.tokenizer.pad_token_id
    ] * (probabilities[:, :, processor.tokenizer.pad_token_id] > 0.5)

    # filter out unwanted tokens
    target_vocab = set("".join(vocab.values()))
    for t in set(processor.tokenizer.vocab.keys()).difference(target_vocab):
        if t in processor.tokenizer.special_tokens_map.values():
            continue
        probabilities[:, :, processor.tokenizer.vocab[t]] = 0

    predicted_ids = torch.argmax(probabilities, dim=-1)
    return [processor.decode(ids) for ids in predicted_ids]
