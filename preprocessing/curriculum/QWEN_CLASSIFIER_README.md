# Qwen2.5-7B-Instruct Multi-Label Classifier

## Overview

This directory contains an upgraded music sentence classification pipeline that replaces the zero-shot DeBERTa classifier with **Qwen2.5-7B-Instruct**, a state-of-the-art language model fine-tuned for instruction following.

### Key Improvements

- **Multi-label Classification**: Each sentence can now be assigned multiple labels simultaneously (not just one)
- **Better Semantic Understanding**: Qwen2.5 provides deeper semantic understanding of music descriptions
- **Flexible Output**: JSON-based output parsing with robust error handling
- **GPU Optimized**: Full GPU support with efficient batching and float16 precision
- **Progress Tracking**: tqdm progress bars for all long-running operations

## Architecture

### Files

- **`classifier.py`**: Core Qwen model loading and inference
  - `build_qwen_classifier()`: Load and initialize the model
  - `classify_sentences_batch()`: Batch classification with multi-label support
  - `parse_model_output()`: Robust JSON parsing with fallback strategies

- **`classify_text.py`**: High-level API for caption classification
  - `classify_caption_sentences_multilabel()`: Process 2D list of sentences, preserving structure
  - `generate_caption_labels()`: End-to-end pipeline (load data → classify → save)

- **`test_qwen_classifier.py`**: Comprehensive test suite

## Label Schema

```python
0 = instrumentation and timbre
1 = tempo rhythm beat meter
2 = harmony key chord progression
3 = genre style production
4 = emotion mood feeling
5 = scene imagery context
```

## Usage

### Basic Classification

```python
from preprocessing.curriculum import generate_caption_labels

# Generate labels for music captions
labels = generate_caption_labels(
    data_file="data/MusicBench_train.json",
    batch_size=8,
    device=0  # GPU device, or -1 for CPU
)

# Output structure:
# {
#     "main_caption": [[[0, 2], [1, 4]], [[3], [0, 1, 5]], ...],
#     "alt_caption": [[[4, 5], [2]], ...]
# }
```

### Direct Classifier Usage

```python
from preprocessing.curriculum.classifier import (
    build_qwen_classifier,
    classify_sentences_batch,
    LABELS
)

# Load model
classifier = build_qwen_classifier(device=0)

# Classify sentences
sentences = [
    "slow piano melody with sustained notes",
    "heavy drums with distorted electric guitar",
    "minor key progression creating a melancholic atmosphere"
]

results = classify_sentences_batch(sentences, classifier, batch_size=8)

# Results: [[0, 4], [0, 3], [2, 4]]
# (each sentence has multiple labels)
```

### Batch Processing with Structure Preservation

```python
from preprocessing.curriculum.classify_text import classify_caption_sentences_multilabel
from preprocessing.curriculum.classifier import build_qwen_classifier

# 2D list of split sentences (sentences grouped by caption)
split_captions = [
    ["slow piano melody", "sad emotional atmosphere"],
    ["heavy drums", "minor key progression"],
    ["orchestral strings", "baroque style"],
]

classifier = build_qwen_classifier(device=0)

# Classify while preserving 2D structure
results = classify_caption_sentences_multilabel(
    split_captions,
    classifier,
    batch_size=8
)

# Results preserve structure:
# [
#     [[0, 4], [4]],          # First caption: 2 sentences
#     [[0], [2, 4]],          # Second caption: 2 sentences
#     [[0], [3]]              # Third caption: 2 sentences
# ]
```

## Implementation Details

### Multi-Label Classification Prompt

The model receives the following system and user prompts:

**System:**
```
You are a music annotation expert. Your task is to classify music-related sentences into semantic categories.

Categories:
0 = instrumentation and timbre
1 = tempo rhythm meter
2 = harmony key chord progression
3 = genre style production
4 = emotion mood feeling
5 = scene imagery context

Rules:
- A sentence can belong to one or more categories.
- Return ONLY a JSON array of integers (e.g., [0, 2, 4]).
- Do not include any explanation or additional text.
- If the sentence is empty or not music-related, return an empty array [].
```

**User:**
```
Classify this music sentence into one or more categories.

Sentence: "{sentence}"

Return ONLY the JSON array:
```

### Output Parsing Strategy

The parser uses a multi-stage approach to robustly extract labels:

1. **JSON Pattern Match**: Try to find `[...]` pattern and parse as JSON
2. **Fallback Digit Extraction**: Extract individual digits [0-5] if JSON parsing fails
3. **Validation**: Ensure all extracted values are integers in range [0-5]
4. **Deduplication**: Remove duplicate labels and sort
5. **Empty Fallback**: Return empty list `[]` if all strategies fail

Example outputs handled:
- `[0, 2, 4]` → `[0, 2, 4]` ✓
- `0, 2, 4` → `[0, 2, 4]` ✓ (fallback digit extraction)
- `Labels: [0, 2, 4]` → `[0, 2, 4]` ✓ (pattern match)
- `0 2 4` → `[0, 2, 4]` ✓ (fallback digit extraction)
- `invalid` → `[]` ✓ (empty fallback)

### Hardware Requirements

- **GPU**: Recommended (Qwen2.5-7B uses ~16GB VRAM with float16)
- **CPU**: Supported but slow (not recommended for large datasets)

### Memory Optimization

- **float16 precision**: Reduces memory usage from ~28GB to ~16GB
- **Batch processing**: Process multiple sentences per forward pass
- **Device mapping**: Automatic device selection (GPU if available, else CPU)

## Testing

Run the comprehensive test suite:

```bash
cd preprocessing/curriculum
python test_qwen_classifier.py
```

Tests include:
1. Single sentence classification
2. Multi-label structure preservation
3. Empty sentence handling
4. Malformed output recovery

## Output Format

The classification results are saved as JSON with structure:

```json
{
  "main_caption": [
    [[0, 2], [1, 4], [3]],
    [[0], [2, 4]],
    [[4, 5], [1, 2]]
  ],
  "alt_caption": [
    [[2], [0, 1]],
    [[3, 4], [5]],
    [[0, 2, 4]]
  ]
}
```

Where:
- Outer list: captions
- Middle list: sentences within each caption
- Inner list: labels assigned to each sentence (can be multiple)

## Performance

- **Time per sentence**: ~0.5-1.0 seconds (GPU) / ~5-10 seconds (CPU)
- **Throughput**: ~8-10 sentences/sec (batch_size=8, single GPU)
- **Quality**: Multi-label support provides more nuanced classifications

## Dependencies

Key packages (see `requirements.txt`):
- `transformers>=4.57.0`
- `torch>=2.0.0`
- `tqdm`
- `datasets`

## Migration from Zero-Shot Classifier

**Previous output (single-label):**
```python
[
    [0, 1, 0],  # 3 sentences, each with 1 label
    [2, 3, 1]
]
```

**New output (multi-label):**
```python
[
    [[0, 2], [1, 4], [0, 3]],  # 3 sentences, each with multiple labels
    [[2], [3, 5], [1, 2]]
]
```

The output structure changes from `List[List[int]]` to `List[List[List[int]]]`. Update any downstream code accordingly.

## Advanced Usage

### Custom System Prompt

```python
from preprocessing.curriculum.classifier import build_qwen_classifier
from preprocessing.curriculum.classify_text import classify_caption_sentences_multilabel

classifier = build_qwen_classifier(device=0)

# Modify the SYSTEM_PROMPT in classifier.py for different instructions
```

### Custom Batch Size

```python
# Smaller batch for limited VRAM
results = classify_caption_sentences_multilabel(
    split_captions,
    classifier,
    batch_size=4  # or 2, or 1
)
```

### CPU-Only Mode

```python
# Disable GPU
classifier = build_qwen_classifier(device=-1)
```

## Troubleshooting

### Out of Memory Error

- Reduce `batch_size` to 4, 2, or 1
- Use CPU mode (slower but no VRAM limit)
- Reduce model precision (already using float16)

### Model Download Errors

Ensure HuggingFace credentials are configured:
```bash
huggingface-cli login
```

### Parsing Errors

Check `parse_model_output()` function in `classifier.py`. It handles various malformed outputs but can be extended with custom parsing logic if needed.

## Citation

Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
