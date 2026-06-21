import re
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r'(?<!\d)\.|\.(?!\d)')


def split_caption_sentences(captions: List[str]) -> List[List[str]]:
    """Split captions into sentences without breaking numeric decimals."""
    return [
        [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(caption) if sentence.strip()]
        for caption in captions
    ]
