from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json

from datasets import DatasetDict
from tqdm.auto import tqdm

from .classifier import build_qwen_classifier, classify_sentences_batch, LABELS, LABEL_TO_ID
from .data_loader import load_musicbench_dataset
from .utils import split_caption_sentences


def _validate_dataset(dataset: DatasetDict) -> None:
    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split.")

    required_columns = {"main_caption", "alt_caption"}
    missing = required_columns - set(dataset["train"].column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


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

    # Classify all sentences in batches
    flat_labels: List[List[int]] = []
    total_batches = (len(flat_sentences) + batch_size - 1) // batch_size
    
    with tqdm(total=len(flat_sentences), desc="Classifying sentences") as pbar:
        for i in range(0, len(flat_sentences), batch_size):
            batch = flat_sentences[i : i + batch_size]
            batch_results = classify_sentences_batch(batch, classifier_pipe, batch_size=len(batch))
            flat_labels.extend(batch_results)
            pbar.update(len(batch))

    # Reconstruct 2D structure
    classified: List[List[List[int]]] = []
    idx = 0
    for length in tqdm(structure, desc="Reconstructing captions"):
        caption_labels = flat_labels[idx : idx + length]
        classified.append(caption_labels)
        idx += length

    return classified


def generate_caption_labels(
    data_file: Optional[str] = None,
    batch_size: int = 8,
    device: int = 0,
    save_paths: Optional[Dict[str, str]] = None,
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
    
    print("Classifying alt captions...")
    alt_caption_classes = classify_caption_sentences_multilabel(
        split_alt_captions,
        classifier_pipe,
        batch_size=batch_size,
    )

    labels = {
        "main_caption": main_caption_classes,
        "alt_caption": alt_caption_classes,
    }

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
    with open(save_paths["main_caption"], "w", encoding="utf-8") as f:
        json.dump(main_caption_classes, f, indent=2)
    
    print(f"Saving alt_caption labels to {save_paths['alt_caption']}")
    with open(save_paths["alt_caption"], "w", encoding="utf-8") as f:
        json.dump(alt_caption_classes, f, indent=2)

    print("Classification complete!")
    return labels
