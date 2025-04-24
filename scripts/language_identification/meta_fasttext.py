#!/usr/bin/env python3

import sys
import fasttext
from langcodes import standardize_tag
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin"
)
model = fasttext.load_model(model_path)


def facebook_identify_language(text: str, top_k=1) -> str:
    label, probability = model.predict(text, k=top_k)
    tag = label[0].replace("__label__", "")
    return standardize_tag(tag), probability.item()  # type: ignore


def main(args):
    print(facebook_identify_language(" ".join(args)))


if __name__ == "__main__":
    # Usage: python ./scripts/language_identification/fasttext.py <text>
    main(sys.argv[1:])
