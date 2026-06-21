from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json
import random

from datasets import DatasetDict
from tqdm.auto import tqdm

from .classifier import (
    build_qwen_classifier,
    classify_by_rules,
    classify_sentences_batch,
    LABELS,
    LABEL_TO_ID,
)
from .data_loader import load_musicbench_dataset
from .utils import split_caption_sentences


def _validate_dataset(dataset: DatasetDict) -> None:
    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split.")

    required_columns = {"main_caption", "alt_caption"}
    missing = required_columns - set(dataset["train"].column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def verify_label_alignment(
    caption_name: str,
    original_captions: Sequence[str],
    split_captions: Sequence[Sequence[str]],
    caption_classes: Sequence[Sequence[Sequence[int]]],
    skip_mismatched_samples: bool = False,
) -> None:
    """Verify that each caption's sentence count matches its label count."""
    assert len(original_captions) == len(split_captions), (
        f"Original caption count and split caption count differ for {caption_name}."
    )
    assert len(split_captions) == len(caption_classes), (
        f"Caption group count and class group count differ for {caption_name}."
    )

    mismatched_indices = []
    for idx, (orig, split_sentences, labels) in enumerate(zip(original_captions, split_captions, caption_classes)):
        if len(split_sentences) != len(labels):
            mismatched_indices.append(idx)
            print(f"[verify_label_alignment] MISMATCH in {caption_name} at index {idx}:")
            print("  Original caption:", orig)
            print("  Split sentences:", split_sentences)
            print("  Labels:", labels)
            print("  sentence count:", len(split_sentences), "label group count:", len(labels))
            print()
            if skip_mismatched_samples:
                caption_classes[idx] = [[] for _ in split_sentences]

    if mismatched_indices and not skip_mismatched_samples:
        raise AssertionError(
            f"Label alignment failed for {caption_name} at indices {mismatched_indices}."
        )


def classify_caption_sentences_multilabel(
    split_captions: Sequence[Sequence[str]],
    classifier_pipe,
    batch_size: int = 8,
) -> List[List[List[int]]]:
    """
    Classify caption sentences with multi-label support.
    
    Args:
        split_captions: 2D list of sentences [[sent1, sent2], [sent3]]
        classifier_pipe: Qwen classifier pipe from build_qwen_classifier()
        batch_size: Batch size for processing
    
    Returns:
        2D list where each sentence has a list of label indices [[[0, 2], [1]], [[3, 4]]]
    """
    flat_sentences: List[str] = []
    structure: List[int] = []

    # Flatten captions while tracking structure
    for caption in split_captions:
        cleaned = [sentence.strip() for sentence in caption if sentence.strip()]
        structure.append(len(cleaned))
        flat_sentences.extend(cleaned)

    # Apply rule-based classification first
    flat_labels: List[List[int]] = []
    unresolved_sentences: List[str] = []
    unresolved_indices: List[int] = []

    for idx, sentence in enumerate(flat_sentences):
        rule_label = classify_by_rules(sentence)
        if rule_label is not None:
            flat_labels.append([rule_label])
        else:
            flat_labels.append([])
            unresolved_sentences.append(sentence)
            unresolved_indices.append(idx)

    # Zero-shot classify only unresolved sentences
    if unresolved_sentences:
        with tqdm(total=len(unresolved_sentences), desc="Zero-shot classifying") as pbar:
            for i in range(0, len(unresolved_sentences), batch_size):
                batch = unresolved_sentences[i : i + batch_size]
                batch_results = classify_sentences_batch(batch, classifier_pipe, batch_size=len(batch))
                for j, labels in enumerate(batch_results):
                    flat_labels[unresolved_indices[i + j]] = labels
                pbar.update(len(batch))

    # Reconstruct 2D structure
    classified: List[List[List[int]]] = []
    idx = 0
    for length in tqdm(structure, desc="Reconstructing captions"):
        caption_labels = flat_labels[idx : idx + length]
        classified.append(caption_labels)
        idx += length

    return classified


def log_label_distribution(caption_classes: Sequence[Sequence[Sequence[int]]], label_type: str) -> None:
    """Print distribution statistics and sample sentences for each label."""
    flat_labels = [label for caption in caption_classes for sentence_labels in caption for label in sentence_labels]
    total = len(flat_labels)
    counts = Counter(flat_labels)

    print(f"\nLabel distribution for {label_type}:")
    for label_idx, label_name in enumerate(LABELS):
        count = counts.get(label_idx, 0)
        pct = (count / total * 100) if total else 0.0
        print(f"  {label_idx}: {count} ({pct:.1f}%) - {label_name}")


def get_label_samples(
    split_captions: Sequence[Sequence[str]],
    caption_classes: Sequence[Sequence[Sequence[int]]],
    max_samples: int = 10,
) -> Dict[int, List[str]]:
    examples = defaultdict(list)
    for caption_sentences, caption_labels in zip(split_captions, caption_classes):
        for sentence, labels in zip(caption_sentences, caption_labels):
            for label in labels:
                if len(examples[label]) < max_samples * 5:
                    examples[label].append(sentence)
    samples = {}
    for label_idx in range(len(LABELS)):
        items = examples.get(label_idx, [])
        samples[label_idx] = random.sample(items, min(len(items), max_samples)) if items else []
    return samples


def print_label_samples(samples: Dict[int, List[str]], label_type: str) -> None:
    print(f"\nRandom samples by label for {label_type}:")
    for label_idx, label_name in enumerate(LABELS):
        print(f"\nLabel {label_idx} ({label_name}) - {len(samples.get(label_idx, []))} examples")
        for sentence in samples.get(label_idx, []):
            print(f"  - {sentence}")


def generate_caption_labels(
    data_file: Optional[str] = None,
    batch_size: int = 8,
    device: int = 0,
    save_paths: Optional[Dict[str, str]] = None,
    skip_mismatched_samples: bool = False,
) -> Dict[str, List[List[List[int]]]]:
    """
    Generate multi-label classifications for music captions using Qwen2.5-7B.
    
    Args:
        data_file: Path to dataset file
        batch_size: Batch size for model inference
        device: GPU device index (or -1 for CPU)
        save_paths: Custom paths for saving results
    
    Returns:
        Dict with main_caption and alt_caption multi-label classifications
    """
    dataset = load_musicbench_dataset(data_file)
    _validate_dataset(dataset)

    main_captions = dataset["train"]["main_caption"]
    alt_captions = dataset["train"]["alt_caption"]

    split_main_captions = split_caption_sentences(main_captions)
    split_alt_captions = split_caption_sentences(alt_captions)

    print("Loading Qwen2.5-7B-Instruct model...")
    classifier_pipe = build_qwen_classifier(device=device)

    print("Classifying main captions...")
    main_caption_classes = classify_caption_sentences_multilabel(
        split_main_captions,
        classifier_pipe,
        batch_size=batch_size,
    )
    verify_label_alignment(
        "main_caption",
        main_captions,
        split_main_captions,
        main_caption_classes,
        skip_mismatched_samples=skip_mismatched_samples,
    )

    print("Classifying alt captions...")
    alt_caption_classes = classify_caption_sentences_multilabel(
        split_alt_captions,
        classifier_pipe,
        batch_size=batch_size,
    )
    verify_label_alignment(
        "alt_caption",
        alt_captions,
        split_alt_captions,
        alt_caption_classes,
        skip_mismatched_samples=skip_mismatched_samples,
    )

    labels = {
        "main_caption": main_caption_classes,
        "alt_caption": alt_caption_classes,
    }

    log_label_distribution(main_caption_classes, "main_caption")
    log_label_distribution(alt_caption_classes, "alt_caption")

    main_samples = get_label_samples(split_main_captions, main_caption_classes)
    alt_samples = get_label_samples(split_alt_captions, alt_caption_classes)

    print_label_samples(main_samples, "main_caption")
    print_label_samples(alt_samples, "alt_caption")

    labels_dir = Path(__file__).resolve().parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    default_save_paths = {
        "main_caption": str(labels_dir / "main_caption_classes.json"),
        "alt_caption": str(labels_dir / "alt_caption_classes.json"),
    }

    if save_paths is None:
        save_paths = default_save_paths
    else:
        save_paths = {**default_save_paths, **save_paths}

    print(f"Saving main_caption labels to {save_paths['main_caption']}")
    assert len(split_main_captions) == len(main_caption_classes), (
        "Main caption sentence groups and label groups must match before saving."
    )
    for idx, (sentences, labels) in enumerate(zip(split_main_captions, main_caption_classes)):
        assert len(sentences) == len(labels), (
            f"Main caption group {idx} has {len(sentences)} sentences but {len(labels)} label groups."
        )

    with open(save_paths["main_caption"], "w", encoding="utf-8") as f:
        json.dump(main_caption_classes, f, indent=2)
    
    print(f"Saving alt_caption labels to {save_paths['alt_caption']}")
    assert len(split_alt_captions) == len(alt_caption_classes), (
        "Alt caption sentence groups and label groups must match before saving."
    )
    for idx, (sentences, labels) in enumerate(zip(split_alt_captions, alt_caption_classes)):
        assert len(sentences) == len(labels), (
            f"Alt caption group {idx} has {len(sentences)} sentences but {len(labels)} label groups."
        )

    with open(save_paths["alt_caption"], "w", encoding="utf-8") as f:
        json.dump(alt_caption_classes, f, indent=2)

    print("Classification complete!")
    return labels
