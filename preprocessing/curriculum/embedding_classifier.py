from typing import TYPE_CHECKING, Dict, Iterable, List

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def build_prototype_embeddings(
    embedding_model: "SentenceTransformer",
    prototypes: Dict[int, List[str]],
) -> Dict[int, np.ndarray]:
    prototype_embeddings = {}

    for label, prototype_sentences in prototypes.items():
        embeddings = embedding_model.encode(prototype_sentences)
        prototype_embeddings[label] = _normalize(np.mean(embeddings, axis=0))

    return prototype_embeddings


def classify_caption_sentences(
    split_captions: Iterable[List[str]],
    embedding_model: "SentenceTransformer",
    prototype_embeddings: Dict[int, np.ndarray],
) -> List[List[int]]:
    classified = []

    for caption in tqdm(split_captions, desc="Classifying captions"):
        if len(caption) == 0:
            classified.append([])
            continue

        sentence_embeddings = embedding_model.encode(caption)
        caption_labels = []

        for sentence_embedding in sentence_embeddings:
            sentence_embedding = _normalize(sentence_embedding)
            similarities = {
                label: float(np.dot(sentence_embedding, prototype_embedding))
                for label, prototype_embedding in prototype_embeddings.items()
            }
            caption_labels.append(max(similarities, key=similarities.get))

        classified.append(caption_labels)

    return classified
