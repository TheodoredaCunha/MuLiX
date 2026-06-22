import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Union

from .io_utils import resolve_project_path
from .prototypes import LABEL_NAMES, PLOT_LABELS

PathLike = Union[str, Path]


def count_labels(classes: Iterable[List[int]]) -> Counter:
    return Counter(label for caption in classes for label in caption)


def save_distribution_plots(
    main_classes: List[List[int]],
    alt_classes: List[List[int]],
    output_dir: PathLike,
) -> List[Path]:
    import matplotlib.pyplot as plt

    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_counts = count_labels(main_classes)
    alt_counts = count_labels(alt_classes)
    labels = list(range(6))
    names = [PLOT_LABELS[label] for label in labels]
    paths = []

    paths.append(
        _save_grouped_bar(
            plt,
            names,
            [main_counts.get(label, 0) for label in labels],
            [alt_counts.get(label, 0) for label in labels],
            "Curriculum Label Distribution",
            "Frequency",
            output_dir / "curriculum_label_distribution.png",
        )
    )

    main_total = sum(main_counts.values()) or 1
    alt_total = sum(alt_counts.values()) or 1
    paths.append(
        _save_grouped_bar(
            plt,
            names,
            [main_counts.get(label, 0) / main_total for label in labels],
            [alt_counts.get(label, 0) / alt_total for label in labels],
            "Normalized Curriculum Distribution",
            "Percentage",
            output_dir / "normalized_curriculum_distribution.png",
        )
    )

    return paths


def _save_grouped_bar(plt, labels, main_values, alt_values, title, ylabel, path):
    x = range(len(labels))
    width = 0.38

    plt.figure(figsize=(14, 6))
    plt.bar([idx - width / 2 for idx in x], main_values, width, label="main")
    plt.bar([idx + width / 2 for idx in x], alt_values, width, label="alt")
    plt.xticks(list(x), labels, rotation=20)
    plt.title(title)
    plt.xlabel("Curriculum Class")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_label_examples(
    split_main_captions: List[List[str]],
    main_classes: List[List[int]],
    split_alt_captions: List[List[str]],
    alt_classes: List[List[int]],
    output_dir: PathLike,
    examples_per_label: int = 10,
    seed: int = 0,
) -> Path:
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "curriculum_label_examples.txt"

    rng = random.Random(seed)
    main_examples = _group_examples(split_main_captions, main_classes)
    alt_examples = _group_examples(split_alt_captions, alt_classes)
    lines = []

    for label in range(6):
        lines.append("=" * 60)
        lines.append(f"LABEL {label}: {LABEL_NAMES[label]}")
        lines.append("")
        lines.append("MAIN CAPTION EXAMPLES:")
        lines.extend(_sample_lines(main_examples[label], examples_per_label, rng))
        lines.append("")
        lines.append("ALT CAPTION EXAMPLES:")
        lines.extend(_sample_lines(alt_examples[label], examples_per_label, rng))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _group_examples(split_captions, classes) -> Dict[int, List[str]]:
    examples = defaultdict(list)
    for caption_sentences, caption_labels in zip(split_captions, classes):
        for sentence, label in zip(caption_sentences, caption_labels):
            examples[label].append(sentence)
    return examples


def _sample_lines(examples, sample_size, rng):
    if len(examples) == 0:
        return ["(none)"]
    return [
        f"- {example}"
        for example in rng.sample(examples, min(sample_size, len(examples)))
    ]
