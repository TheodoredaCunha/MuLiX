import re
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r'(?<!\d)\.(?!\d)')
_SUSPICIOUS_FRAGMENT_RE = re.compile(r'^(0\b|0\s|0\s*bpm|0\s*beats)', re.IGNORECASE)


def split_caption_sentences(captions: List[str]) -> List[List[str]]:
    """Split captions into sentences without breaking numeric decimals or version-like numbers."""
    grouped_sentences: List[List[str]] = []

    for caption in captions:
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(caption) if sentence.strip()]
        grouped_sentences.append(sentences)

        suspicious = [sentence for sentence in sentences if _SUSPICIOUS_FRAGMENT_RE.search(sentence)]
        if suspicious:
            print("[split_caption_sentences] Suspicious caption split detected:")
            print("  Original caption:", caption)
            print("  Split sentences:", sentences)
            print("  Suspicious fragments:", suspicious)

    return grouped_sentences


if __name__ == "__main__":
    test_caption = (
        "This 120.0 beats per minute song has a chord sequence of Dbm7b5/B. "
        "It's perfect for a TV series soundtrack."
    )

    result = split_caption_sentences([test_caption])
    print("Split result:", result)

    expected = [
        [
            "This 120.0 beats per minute song has a chord sequence of Dbm7b5/B",
            "It's perfect for a TV series soundtrack",
        ]
    ]
    assert result == expected, f"Expected {expected}, got {result}"
    print("Sentence splitting test passed.")
