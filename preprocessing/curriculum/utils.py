from typing import List


def split_caption_sentences(captions: List[str]) -> List[List[str]]:
    return [
        [sentence.strip() for sentence in caption.split(".") if sentence.strip()]
        for caption in captions
    ]
