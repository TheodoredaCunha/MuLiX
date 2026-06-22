import re
from typing import Iterable, List

_SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)\.(?!\d)")


def split_caption_sentences(captions: Iterable[str]) -> List[List[str]]:
    return [
        [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(caption or "")
            if sentence.strip()
        ]
        for caption in captions
    ]
