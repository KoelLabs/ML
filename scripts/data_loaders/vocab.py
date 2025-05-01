# Utils for handling vocab

import os
import sys

from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_loaders.common import BaseDataset
from forced_alignment.common import group_phonemes


def parse_vocab_by_character(dataset: BaseDataset, transform=lambda x: x):
    """Naive, each character is its own vocab token"""
    vocab = set()
    for sample in dataset:
        vocab |= set(transform(sample[0]))  # type: ignore
    return vocab - {""}


def parse_vocab_by_groups(dataset: BaseDataset, transform=lambda x: x):
    """Group characters into tokens by prefix minimum"""
    vocab = set()
    for sample in dataset:
        ipa = transform(sample[0])  # type: ignore
        if "̄" in ipa:
            raise ValueError("Warning: we use this IPA as a temporary marker for ŋ̍")
        if "ŋ̍" in ipa:
            ipa = ipa.replace("ŋ̍", "̄")
        vocab |= set([transform(x.replace("̄", "ŋ̍")) for x in group_phonemes(ipa)])
    return vocab - {""}

def get_unmatched_groups():
    grouped_phoneme_vocab = get_vocab_superset_fast(data, simplify_ipa)
    matched_tokens, _ = parse_vocab_aligned_with_model(
        data, model_vocab, simplify_ipa
    )
    
    unmatched_new_tokens_grouped = set()
    unmatched_seen_tokens_grouped = set()
    for p in grouped_phoneme_vocab:
        seen = False
        if p not in matched_tokens.keys():
            for c in p: 
                if c in matched_tokens.keys(): # if a subtoken matches, we do not consider it unmatched
                    unmatched_seen_tokens_grouped.add(p)
                    seen = True
                    break   
            if not seen: 
                unmatched_new_tokens_grouped.add(p)
    

    return [unmatched_new_tokens_grouped, unmatched_seen_tokens_grouped]

def get_vocab_superset_fast(
    dataset: BaseDataset, transform=lambda x: x, fallback=parse_vocab_by_groups
):
    """Use the code.py information to deduce a vocab superset if possible, otherwise fall back to fallback"""
    if dataset.vocab is not None:
        return set(transform(x) for x in dataset.vocab) - {""}
    else:
        return fallback(dataset, transform) - {""}


def parse_vocab_aligned_with_model(
    dataset: BaseDataset,
    model_vocab: dict,
    transform=lambda x: x,
    include_list_for_partial_matches=True,
    keep_symbols_paired=True,
    fallback=parse_vocab_by_groups,
):
    """
    Identify which dataset tokens can be mapped directy to a model token id (transfer weights directly),
    which appear as a subtoken of some existing model tokens (possibly average weights),
    and which do not have any good matches (new random weights).

    The direct maps are returned as keys in matched_tokens that map to an int token id.
    The subtoken maps are returned as keys in matched_tokens that map to a list of int token ids.
    The unmatched tokens are returned in unmatched_tokens.

    With keep_symbols_paired, for the subtoken maps, you will need to decide whether they are necessary to keep
    With include_list_for_partial_matches, no smart grouping is done on unmatched_tokens. You may want to use one of the other vocab methods for this
    """
    model_vocab_items = sorted(model_vocab.items(), key=lambda x: len(x[0]))
    model_vocab = OrderedDict(model_vocab_items)

    matched_tokens = {}
    unmatched_tokens = set()
    for sample in dataset:
        ipa: str = transform(sample[0])  # type: ignore
        for k, v in model_vocab.items():
            k = transform(k)
            if not k:
                continue
            if k in ipa:
                matched_tokens[k] = v
                if not keep_symbols_paired:
                    ipa = ipa.replace(k, "")
        if len(ipa) == 0:
            continue
        if include_list_for_partial_matches:
            for c in ipa:
                if c in matched_tokens or c in unmatched_tokens:
                    continue
                matches = []
                for k, v in model_vocab.items():
                    if c in transform(k):
                        matches.append(v)
                if len(matches) > 0:
                    matched_tokens[c] = matches
                else:
                    unmatched_tokens.add(c)
        else:
            unmatched_tokens |= fallback([(ipa, None)], transform=transform)

    # remove redundant tokens introduced by keep_symbols_paired, this is a simplification
    to_remove = set()
    for k, v in matched_tokens.items():
        k = transform(k)
        if len(k) < 2:
            continue
        redundant = True
        for c in k:
            if c not in matched_tokens.keys() or type(matched_tokens[c]) != int:
                redundant = False
                break
        if redundant:
            to_remove.add(k)
    for k in to_remove:
        del matched_tokens[k]

    return matched_tokens, unmatched_tokens


# Example code:
if __name__ == "__main__":
    from data_loaders.TIMIT import TIMITDataset
    from data_loaders.PSST import PSSTDataset
    from data_loaders.L2ARCTIC import L2ArcticDataset
    from data_loaders.DoReCo import DoReCoDataset
    from data_loaders.EpaDB import EpaDBDataset
    from data_loaders.SpeechOcean import SpeechOceanDataset
    from data_loaders.OSUBuckeye import BuckeyeDataset

    from core.ipa import simplify_ipa

    # for dataloader in [
    #     TIMITDataset,
    #     PSSTDataset,
    #     L2ArcticDataset,
    #     DoReCoDataset,
    #     EpaDBDataset,
    #     SpeechOceanDataset,
    #     BuckeyeDataset,
    # ]:
    #     data = (
    #         dataloader(force_offline=True)  # type: ignore
    #         if dataloader.__qualname__ == "PSSTDataset"
    #         else dataloader()
    #     )
    #     print(len(data))
    #     print(
    #         f"Fast Vocab Raw ({dataloader.__qualname__}):",
    #         get_vocab_superset_fast(data),
    #     )
    #     print(
    #         f"Fast Vocab Filtered ({dataloader.__qualname__}):",
    #         get_vocab_superset_fast(data, simplify_ipa),
    #     )

    print("------")

    data = EpaDBDataset()
    # print("Raw Vocab:", parse_vocab_by_character(data))
    # print("Filtered Vocab:", parse_vocab_by_character(data, simplify_ipa))

    # print("------")

    # print("Raw Vocab by Group:", parse_vocab_by_groups(data))
    # print("Filtered Vocab by Group:", parse_vocab_by_groups(data, simplify_ipa))

    # print("------")

    import ipa_transcription.wav2vec2  # import this for the espeak patch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    tokenizor = processor.tokenizer
    model_vocab = tokenizor.get_vocab()

    print("Facebook Vocab:", model_vocab.keys())

    matched_tokens, unmatched_tokens = parse_vocab_aligned_with_model(
        data, model_vocab, simplify_ipa
    )
    print("Matched Vocab:", matched_tokens)
    print("Unmatched Vocab:", unmatched_tokens)
    print("get unmatched groups:", get_unmatched_groups())