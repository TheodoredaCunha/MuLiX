from typing import Dict, List, Optional, Sequence
import json

from datasets import DatasetDict
from tqdm.auto import tqdm

from .classifier import build_zero_shot_classifier, LABELS, LABEL_TO_ID
from .data_loader import load_musicbench_dataset
from .utils import split_caption_sentences


def _validate_dataset(dataset: DatasetDict) -> None:
    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split.")

    required_columns = {"main_caption", "alt_caption"}
    missing = required_columns - set(dataset["train"].column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def classify_caption_sentences_fast(
    split_captions: Sequence[Sequence[str]],
    classifier_pipe,
    batch_size: int = 64,
    labels: Sequence[str] = LABELS,
    label_to_id: Dict[str, int] = LABEL_TO_ID,
) -> List[List[int]]:
    flat_sentences: List[str] = []
    structure: List[int] = []

    for caption in split_captions:
        cleaned = [sentence.strip() for sentence in caption if sentence.strip()]
        structure.append(len(cleaned))
        flat_sentences.extend(cleaned)

    flat_labels: List[int] = []
    for i in tqdm(range(0, len(flat_sentences), batch_size), desc="Classifying sentences"):
        batch = flat_sentences[i : i + batch_size]
        results = classifier_pipe(batch, labels, multi_label=False)
        for result in results:
            top_label = result["labels"][0]
            flat_labels.append(label_to_id[top_label])

    classified: List[List[int]] = []
    idx = 0
    for length in tqdm(structure, desc="Reconstructing captions"):
        classified.append(flat_labels[idx : idx + length])
        idx += length

    return classified


def generate_caption_labels(
    data_file: Optional[str] = None,
    batch_size: int = 128,
    device: int = 0,
    save_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, List[List[int]]]:
    dataset = load_musicbench_dataset(data_file)
    _validate_dataset(dataset)

    main_captions = dataset["train"]["main_caption"]
    alt_captions = dataset["train"]["alt_caption"]

    split_main_captions = split_caption_sentences(main_captions)
    split_alt_captions = split_caption_sentences(alt_captions)

    classifier_pipe = build_zero_shot_classifier(device=device)

    main_caption_classes = classify_caption_sentences_fast(
        split_main_captions,
        classifier_pipe,
        batch_size=batch_size,
    )
    alt_caption_classes = classify_caption_sentences_fast(
        split_alt_captions,
        classifier_pipe,
        batch_size=batch_size,
    )

    labels = {
        "main_caption": main_caption_classes,
        "alt_caption": alt_caption_classes,
    }

    if save_paths:
        if "main_caption" in save_paths:
            with open(save_paths["main_caption"], "w", encoding="utf-8") as f:
                json.dump(main_caption_classes, f)
        if "alt_caption" in save_paths:
            with open(save_paths["alt_caption"], "w", encoding="utf-8") as f:
                json.dump(alt_caption_classes, f)

    return labels
