import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE
from core.ipa import remove_length_diacritics, remove_tones_and_stress
from core.codes import ALL_ANNOTATED_IPA_SYMBOLS, string2symbols
from data_loaders.common import show_hf_sample

import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
from datasets import (
    Dataset,
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    interleave_datasets,
)
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)


# ===========================================================================
# ============================= Data Processing =============================
def load_datasets() -> list[Dataset]:
    return [
        load_dataset("KoelLabs/DoReCo")["train"],  # type: ignore
        load_dataset("KoelLabs/EpaDB")["train"],  # type: ignore
        load_dataset("KoelLabs/L2Arctic")["scripted"],  # type: ignore
        load_dataset("KoelLabs/L2ArcticSpontaneousSplit")["train"],  # type: ignore
        load_dataset("KoelLabs/Buckeye")["train"],  # type: ignore
        load_dataset("KoelLabs/PSST")["train"],  # type: ignore
        load_dataset("KoelLabs/PSST")["valid"],  # type: ignore
        load_dataset("KoelLabs/SpeechOceanNoTH")["train"],  # type: ignore
        load_dataset("KoelLabs/TIMIT")["train"],  # type: ignore
    ]


def inspect_samples(datasets: list[Dataset], skip_plot=False):
    for ds in datasets:
        print(ds.info.dataset_name)
        show_hf_sample(ds[0], skip_plot=skip_plot)


def plot_dataset_distributions(datasets: list[Dataset]):
    audio_lengths = []
    ipa_lengths = []
    labels = []

    for ds in datasets:
        # Calculate audio durations
        durations = [x["audio"]["array"].shape[0] / x["audio"]["sampling_rate"] for x in ds]  # type: ignore
        audio_lengths.append(durations)
        # Calculate IPA string lengths
        ipa_len = [len(x["ipa"]) for x in ds]  # type: ignore
        ipa_lengths.append(ipa_len)
        labels.append(ds.info.dataset_name)

    ncols = 3  # Number of columns per row
    nrows = int(np.ceil(len(datasets) / ncols))

    # Audio Lengths Plot
    fig1, axs1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    fig1.suptitle("Audio Length Distributions (seconds)", fontsize=18)
    axs1 = axs1.flatten()  # Make indexing easy
    for idx, (lengths, label) in enumerate(zip(audio_lengths, labels)):
        axs1[idx].hist(lengths, bins=50)
        axs1[idx].set_title(f"{label}")
        axs1[idx].set_xlabel("Seconds")
        axs1[idx].set_ylabel("Count")
    for j in range(idx + 1, len(axs1)):
        fig1.delaxes(axs1[j])  # Hide any unused subplots
    fig1.tight_layout()
    plt.show()

    # IPA Lengths Plot
    fig2, axs2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    fig2.suptitle("IPA Transcription Length Distributions (characters)", fontsize=18)
    axs2 = axs2.flatten()
    for idx, (lengths, label) in enumerate(zip(ipa_lengths, labels)):
        axs2[idx].hist(lengths, bins=50)
        axs2[idx].set_title(f"{label}")
        axs2[idx].set_xlabel("Characters")
        axs2[idx].set_ylabel("Count")
    for j in range(idx + 1, len(axs2)):
        fig2.delaxes(axs2[j])
    fig2.tight_layout()
    plt.show()

    for lengths, label in zip(audio_lengths, labels):
        print(label, "speech minutes: ", sum(lengths) / 60)
    print()
    print(
        "Total audio hours:", sum(s for s in (sum(a) for a in audio_lengths)) / 60 / 60
    )
    print(
        "Max audio sample minutes:",
        max(s for s in (max(a) for a in audio_lengths)) / 60,
    )


def combine_datasets(datasets: list[Dataset], sample_probabilities=None, seed=42):
    columns = ["ipa", "audio"]
    datasets = list(
        map(
            lambda x: x.remove_columns(
                [col for col in x.column_names if col not in columns]
            ),
            datasets,
        )
    )
    if sample_probabilities is None:
        return concatenate_datasets(datasets)
    else:
        # e.g., sample randomly according to quality:
        # sample_probabilities = [0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2]
        return interleave_datasets(
            datasets,
            probabilities=sample_probabilities,
            seed=seed,
            stopping_strategy="all_exhausted",
        )


def is_not_empty(row):
    return (
        row.get("audio") is not None
        and row.get("ipa") is not None
        and len(row["ipa"]) > 0
        and row["audio"].get("array") is not None
        and len(row["audio"]["array"]) > 0
    )


def process_row(processor: Wav2Vec2Processor, rows):
    # model expects audio to be float32 @ 16 kHz
    all_input_values = []
    all_labels = []
    for ipa, audio in zip(rows["ipa"], rows["audio"]):
        assert audio["sampling_rate"] == TARGET_SAMPLE_RATE
        audio = audio["array"].astype(np.float32, copy=False)
        inputs = processor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors=None)  # type: ignore
        input_values = np.asarray(inputs["input_values"][0], dtype=np.float32)
        all_input_values.append(input_values)

        ipa = remove_length_diacritics(remove_tones_and_stress(ipa.replace(" ", "")))
        labels = processor(text=ipa).input_ids
        all_labels.append(labels)
    return {
        "input_values": all_input_values,
        "labels": all_labels,
    }


def process_dataset(
    combined_ds: Dataset, processor: Wav2Vec2Processor, num_proc: int, seed: int
):
    processed_ds = combined_ds.filter(is_not_empty, num_proc=num_proc).shuffle(
        seed=seed
    )
    return processed_ds.map(
        lambda r: process_row(processor, r),
        num_proc=num_proc,
        batch_size=1,
        batched=True,
    )


def get_or_create_processed_dataset(
    path: str,
    combined_ds: Dataset,
    processor: Wav2Vec2Processor,
    num_proc: int,
    seed: int,
):
    if os.path.exists(path):
        processed_ds = load_from_disk(path)
    else:
        processed_ds = process_dataset(combined_ds, processor, num_proc, seed)
        processed_ds.save_to_disk(path)
    return processed_ds


# ===========================================================================
# =============================== Vocabulary ================================
def identify_dataset_vocab(combined_ds: Dataset):
    """Identify a set of symbols that covers the dataset, use the same groupings as the annotators so we don't make up ones that don't exist"""
    symbol_uses = {k: [] for k in ALL_ANNOTATED_IPA_SYMBOLS}  # type: ignore

    def uses_only_symbols(string):
        return string in ALL_ANNOTATED_IPA_SYMBOLS or all(string2symbols(string, ALL_ANNOTATED_IPA_SYMBOLS)[1])  # type: ignore

    def reduce_uses(ipa, idx):
        ipa = remove_length_diacritics(remove_tones_and_stress(ipa.replace(" ", "")))
        assert uses_only_symbols(
            ipa
        ), f"Dataset contains unaccounted for symbols: {ipa}"
        for symbol in symbol_uses.keys():
            if symbol in ipa:
                symbol_uses[symbol].append(idx)

    combined_ds.map(reduce_uses, input_columns="ipa", with_indices=True)

    return set(k for k, v in symbol_uses.items() if len(v) > 0 and k), symbol_uses


def identify_model_vocab(model_id):
    old_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
    model_vocab = old_tokenizer.get_vocab()
    special_tokens = {
        k: model_vocab[k] for k in old_tokenizer.special_tokens_map.values()
    }
    return model_vocab, special_tokens


def align_dataset_with_model_vocab(
    dataset_vocab: set[str], model_vocab: dict[str, int], special_tokens: dict[str, int]
):
    """
    Reconcile model vocab with dataset vocab:
        - Pair each dataset token with corresponding token ids in model_vocab
        - If there is an exact match, we go by that, otherwise, we break down by part
    """

    def flatten_list(nested_list):
        flattened = []
        for item in nested_list:
            if isinstance(item, list):
                flattened.extend(flatten_list(item))  # Recursively flatten sublists
            else:
                flattened.append(item)
        return flattened

    unmatched = set(special_tokens.keys()) | dataset_vocab.copy()
    unmatched -= set(special_tokens.keys())

    matched_tokens = {}
    for s in unmatched:
        if s in model_vocab:
            matched_tokens[s] = model_vocab[s]
    unmatched -= set(matched_tokens.keys())

    partial_matched_tokens = {}
    weird_symbols = {
        "̥": [v for k, v in model_vocab.items() if "̥" in k],
        "̃": [v for k, v in model_vocab.items() if "̃" in k],
        "̍": [v for k, v in model_vocab.items() if "̍" in k],
    }
    combined_model_vocab = model_vocab.copy() | weird_symbols
    for s in unmatched:
        if (
            s == "ĩ" or s == "ĩ"
        ):  # edge case, this one doesn't split properly when looping through characters
            parts = ["ɪ", "̃"] if s == "ĩ" else ["i", "̃"]
            matched = [p in combined_model_vocab.keys() for p in parts]
        elif s == "ə̥":
            partial_matched_tokens[s] = [
                combined_model_vocab["ə"],
                combined_model_vocab["h"],
            ]
            continue
        else:
            parts, matched = string2symbols(s, combined_model_vocab.keys())
        if all(matched):
            partial_matched_tokens[s] = flatten_list(
                [combined_model_vocab[p] for p in parts]
            )
    unmatched -= set(partial_matched_tokens.keys())

    return matched_tokens, partial_matched_tokens, unmatched


def save_model_with_updated_vocab(
    save_dir: str,
    old_model_id,
    special_tokens,
    matched_tokens,
    partial_matched_tokens,
    seed=42,
):
    # Renumber the token sequence
    token_to_old_id = special_tokens | matched_tokens | partial_matched_tokens
    token_sequence = sorted(token_to_old_id.keys(), key=lambda x: token_to_old_id[x][0] if isinstance(token_to_old_id[x], list) else token_to_old_id[x])  # type: ignore
    vocab_dict = {tok: idx for idx, tok in enumerate(token_sequence)}

    # Save vocab.json
    os.makedirs(save_dir, exist_ok=True)
    vocab_json = os.path.join(save_dir, "vocab.json")

    # Save tokenizer
    with open(vocab_json, "w") as f:
        f.write(json.dumps(vocab_dict, ensure_ascii=False, indent=2))
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_json,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(save_dir)

    # Save updated processor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(old_model_id)
    Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    ).save_pretrained(save_dir)

    # Transfer model weights: average weights if multiple matches, otherwise just transfer them, unmatched tokens will get randomly initialized
    # load old weights
    model = Wav2Vec2ForCTC.from_pretrained(old_model_id)
    old_W = model.lm_head.weight.data.clone()  # type: ignore
    old_b = model.lm_head.bias.data.clone()  # type: ignore

    # shrink lm_head & update vocab_size
    in_feats = model.lm_head.in_features  # type: ignore
    model.lm_head = torch.nn.Linear(in_feats, len(vocab_dict), bias=True)  # type: ignore
    model.config.vocab_size = len(vocab_dict)
    torch.nn.init.normal_(model.lm_head.weight, std=0.02)

    # set new weights
    new_W, new_B = model.lm_head.weight.data, model.lm_head.bias.data
    rng = torch.Generator().manual_seed(seed)
    jitter = lambda v, s=0.01: v + torch.empty_like(v).uniform_(-s, s, generator=rng)

    # copy / average / jitter weights
    for tok, new_id in vocab_dict.items():
        old_ids = token_to_old_id[tok]

        if isinstance(old_ids, int):
            # single perfect match - copy weights
            new_W[new_id] = old_W[old_ids]
            new_B[new_id] = old_b[old_ids]
        else:
            if len(old_ids) == 1:
                # single imperfect match - jitter
                new_W[new_id] = jitter(old_W[old_ids[0]])
                new_B[new_id] = old_b[old_ids[0]]
            else:
                # multiple matches - average
                weights = [0.5] + [
                    0.5 / (len(old_ids) - 1) for _ in range(len(old_ids) - 1)
                ]
                new_W[new_id] = torch.stack(
                    [old_W[i] * w for i, w in zip(old_ids, weights)]
                ).mean(0)
                new_B[new_id] = torch.stack(
                    [old_b[i] * w for i, w in zip(old_ids, weights)]
                ).mean(0)

    # Save updated model
    model.save_pretrained(save_dir)

    print("Model weight transfer complete")
    print(f"   old head rows : {old_W.size(0)}")
    print(f"   new head rows : {new_W.size(0)}")

    print("🚀  Condensed checkpoint saved to", save_dir)

    return vocab_dict


# ===========================================================================
# ================================ Training =================================
def prepare_model_and_processor(
    model_dir, freeze_feature_extractor=True
) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model = Wav2Vec2ForCTC.from_pretrained(
            model_dir, attn_implementation="flash_attention_2"
        )
    else:
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)

    # Freeze the CNN encoder
    if freeze_feature_extractor:
        for p in model.wav2vec2.feature_extractor.parameters():  # type: ignore
            p.requires_grad = False

    return model, processor  # type: ignore


@dataclass
class DataCollatorCTCWithPadding:
    """
    Dynamically pad *both* the acoustic inputs and the target
    label sequences for CTC training.

    Notes
    -----
    • Speech features are padded with 0.0 (default supplied by
      `Wav2Vec2Processor.pad`).

    • Label padding indices are replaced with -100, so that
      `nn.CTCLoss` ignores them.
    """

    processor: Wav2Vec2Processor
    padding: "bool | str" = True  # "longest" | "max_length" | True
    pad_to_multiple_of: "int | None" = None  # e.g. 8 for tensor cores

    def __call__(
        self, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        # ─── 1) separate source and target sequences ──────────────────────────────
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        # ─── 2) pad acoustic inputs ──────────────────────────────────────────────
        acoustic_batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # ─── 3) pad label sequences (using target processor branch) ──────────────
        with self.processor.as_target_processor():
            label_batch = self.processor.pad(
                {"input_ids": labels},
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

        # ─── 4) replace label padding with -100 so CTCLoss ignores it ────────────
        acoustic_batch["labels"] = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )

        return acoustic_batch
