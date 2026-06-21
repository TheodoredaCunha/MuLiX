#!/usr/bin/env python3
"""
Test script for Qwen2.5-7B-Instruct multi-label music classifier.
Demonstrates the new classification pipeline with sample data.
"""

from classifier import build_qwen_classifier, classify_sentences_batch, LABELS
import json


def test_single_sentence_classification():
    """Test classification of individual sentences."""
    print("=" * 70)
    print("TEST 1: Single Sentence Classification")
    print("=" * 70)
    
    test_sentences = [
        "soft piano melody with sustained notes",
        "heavy drums with distorted electric guitar",
        "minor key progression creating a melancholic atmosphere",
        "upbeat tempo with groovy bass line",
        "orchestral strings and woodwinds in a baroque style",
        "ambient soundscape with reverb effects",
    ]
    
    print("Loading Qwen2.5-7B-Instruct model...")
    classifier = build_qwen_classifier(device=0)
    
    print("\nClassifying test sentences:\n")
    results = classify_sentences_batch(test_sentences, classifier, batch_size=2)
    
    for sentence, labels in zip(test_sentences, results):
        label_names = [LABELS[i] for i in labels]
        print(f"Sentence: {sentence}")
        print(f"  Labels: {labels}")
        print(f"  Categories: {label_names}")
        print()


def test_multilabel_structure():
    """Test that multi-label structure is preserved correctly."""
    print("=" * 70)
    print("TEST 2: Multi-label Structure (2D List)")
    print("=" * 70)
    
    # Simulate split captions (2D structure)
    split_captions = [
        ["slow piano melody", "sad emotional atmosphere"],
        ["heavy drums", "minor key progression"],
        ["orchestral strings", "baroque style"],
    ]
    
    # Flatten structure
    flat_sentences = []
    structure = []
    for caption in split_captions:
        cleaned = [s.strip() for s in caption if s.strip()]
        structure.append(len(cleaned))
        flat_sentences.extend(cleaned)
    
    print("Loading Qwen2.5-7B-Instruct model...")
    classifier = build_qwen_classifier(device=0)
    
    print(f"\nFlattened sentences: {flat_sentences}")
    print(f"Structure: {structure}")
    
    # Classify
    flat_results = classify_sentences_batch(flat_sentences, classifier, batch_size=2)
    
    # Reconstruct 2D structure
    classified = []
    idx = 0
    for length in structure:
        caption_labels = flat_results[idx : idx + length]
        classified.append(caption_labels)
        idx += length
    
    print("\nClassification results (2D structure):")
    print(json.dumps(classified, indent=2))
    
    return classified


def test_empty_sentences():
    """Test handling of empty sentences."""
    print("=" * 70)
    print("TEST 3: Empty Sentence Handling")
    print("=" * 70)
    
    test_sentences = [
        "upbeat tempo",
        "",  # Empty sentence
        "melancholic atmosphere",
        "   ",  # Whitespace only
        "string quartet performance",
    ]
    
    print("Loading Qwen2.5-7B-Instruct model...")
    classifier = build_qwen_classifier(device=0)
    
    print("\nClassifying with empty sentences:\n")
    results = classify_sentences_batch(test_sentences, classifier, batch_size=2)
    
    for sentence, labels in zip(test_sentences, results):
        print(f"Sentence: '{sentence}'")
        print(f"  Labels: {labels}")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("QWEN2.5-7B MULTI-LABEL MUSIC CLASSIFIER TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_single_sentence_classification()
        test_multilabel_structure()
        test_empty_sentences()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
