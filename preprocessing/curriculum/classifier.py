from typing import Dict, List, Optional

from transformers import pipeline

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

LABELS: List[str] = [
    "surface acoustic property",
    "structural harmonic property",
    "style or genre descriptor",
    "semantic or emotional descriptor",
]

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}


def build_zero_shot_classifier(model_name: Optional[str] = None, device: int = 0):
    return pipeline(
        "zero-shot-classification",
        model=model_name or DEFAULT_MODEL,
        device=device,
    )
