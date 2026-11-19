"""Build and publish DoReCo language splits on the existing gated Hugging Face repo."""

from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Features, Value
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download

from core.audio import TARGET_SAMPLE_RATE, audio_array_to_bytes
from data_loaders.DoReCo import (
    CC_BY_4,
    TOP_DIR,
    DoReCoDataset,
    is_commercial,
    language_table,
)

REPO_ID = "KoelLabs/DoReCo"
CONFIG_NAME = "multilingual"
FEATURES = Features(
    {
        "audio": Audio(sampling_rate=TARGET_SAMPLE_RATE),
        "ipa": Value("string"),
        "text": Value("string"),
        "language": Value("string"),
        "recording": Value("string"),
        "speaker_code": Value("string"),
        "utterance_ref": Value("string"),
        "phones": [
            {
                "ipa": Value("string"),
                "start_sample": Value("int32"),
                "end_sample": Value("int32"),
            }
        ],
    }
)


def _rows(language, root):
    dataset = DoReCoDataset(
        language=language,
        root=root,
        include_timestamps=True,
        include_speaker_info=True,
        include_text=True,
    )
    for ipa, audio, phones, metadata, text in dataset:
        yield {
            "audio": {"bytes": audio_array_to_bytes(audio), "path": None},
            "ipa": ipa,
            "text": text,
            "language": language,
            "recording": metadata["name"],
            "speaker_code": metadata["utt_spk_code"],
            "utterance_ref": metadata["utterance_ref"],
            "phones": [
                {"ipa": phone, "start_sample": start, "end_sample": end}
                for phone, start, end in phones
            ],
        }


def prepare_split(language, root, output):
    """Write a complete language split as Parquet."""
    rows = _rows(language, root)
    schema = FEATURES.arrow_schema
    with TemporaryDirectory(dir=output.parent) as cache:
        path = Path(cache) / output.name
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            while batch := list(islice(rows, 64)):
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        path.replace(output)


def _hub_path(language):
    return f"multilingual/{language}.parquet"


def _update_card(card, corpora):
    configs = card.data.get("configs", [])
    card.data.configs = [
        config for config in configs if config["config_name"] != CONFIG_NAME
    ] + [
        {
            "config_name": CONFIG_NAME,
            "data_files": [
                {"split": language, "path": _hub_path(language)} for language in corpora
            ],
        }
    ]
    info = card.data.get("dataset_info")
    if isinstance(info, dict):
        card.data.dataset_info = [dict(info, config_name="default")]
    card.data.license = "cc-by-4.0"
    card.data.pretty_name = "DoReCo (Koel Labs)"
    card.data.multilinguality = "multilingual"
    card.data.size_categories = ["10K<n<100K"]
    card.data.language = None
    card.data.tags = ["speech", "IPA"]

    entries = []
    citations = []
    for language, (entry, citation, count) in corpora.items():
        entries.append(
            f"| `{language}` | {entry['name']} | {count:,} | "
            f"[CC BY 4.0]({CC_BY_4}) | [CC BY 4.0]({CC_BY_4}) | "
            f"[Source and citation](https://doi.org/{entry['doi']}) |"
        )
        citations.append(f"- **`{language}`** {citation}")
    split_table = "\n".join(entries)
    references = "\n".join(citations)
    return DatasetCard(f"""---
{card.data.to_yaml()}
---

# DoReCo (Koel Labs)

[DoReCo 2.0](https://doreco.huma-num.fr/) contains time-aligned speech and
annotations across 53 languages. The `multilingual` configuration contains a
split named for each exported language's Glottolog code. The original
Southern England English release remains available as `default/train`.

## Usage

Request access on the dataset page and sign in with a Hugging Face token.

```python
from datasets import load_dataset

sumi = load_dataset("KoelLabs/DoReCo", "multilingual", split="sumi1235")
legacy_english = load_dataset("KoelLabs/DoReCo", "default", split="train")
```

## Processing

The [Koel Labs DoReCo loader](https://github.com/KoelLabs/ML/blob/main/scripts/data_loaders/DoReCo.py)
segments core recordings into annotation units, converts audio to mono at
16 kHz, and converts phone labels from X-SAMPA to IPA. The `phones` array
gives each IPA label with start and end sample offsets within the clip.
The `recording` and `utterance_ref` fields identify the source. The labels
reflect the source annotations and alignment. Some clips contain speech
without phone labels or overlapping speakers.

## Language splits and licenses

| Split | Language | Clips | Audio | Annotations | Source |
| --- | --- | ---: | --- | --- | --- |
{split_table}

The original `default/train` split uses the English (Southern England)
source and citation listed under `sout3282`. Credit the creators of every
source corpus you use and indicate modifications, as required by
[CC BY 4.0]({CC_BY_4}).

## Source citations

{references}
""")


def export_doreco(root=TOP_DIR):
    """Validate all source recordings, then publish all splits and their card together."""
    root = Path(root)
    corpora = {}
    missing = []
    for code, entry in language_table().items():
        if not is_commercial(entry):
            continue
        dataset = DoReCoDataset(language=code, root=root)
        missing.extend(name for _, name in dataset.missing_recordings())
        citation = dataset.citations[code]
        if not citation or not len(dataset):
            raise ValueError(f"Missing citation or utterances for DoReCo {code}")
        corpora[code] = entry, citation, len(dataset)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} DoReCo recordings in {root}, including "
            + ", ".join(missing[:5])
        )
    with TemporaryDirectory() as staging:
        staging = Path(staging)
        for code in corpora:
            print(f"Preparing DoReCo split {code}", flush=True)
            prepare_split(code, root, staging / f"{code}.parquet")
        return publish_doreco(corpora, staging)


def publish_doreco(corpora, staging):
    api = HfApi()
    repo = api.dataset_info(REPO_ID)
    if not repo.gated:
        raise ValueError(f"Expected an existing gated dataset at {REPO_ID}")
    path = hf_hub_download(
        "KoelLabs/DoReCo", "README.md", repo_type="dataset", revision=repo.sha
    )
    card = _update_card(DatasetCard.load(path), corpora)
    operations = [
        CommitOperationAdd(
            path_in_repo=_hub_path(code),
            path_or_fileobj=staging / f"{code}.parquet",
        )
        for code in corpora
    ]
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card).encode())
    )
    return api.create_commit(
        REPO_ID,
        operations=operations,
        repo_type="dataset",
        commit_message="Publish DoReCo multilingual splits with source attribution",
        parent_commit=repo.sha,
        num_threads=8,
    )
