import json
import re
import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

LABELS: List[str] = [
    "instrumentation timbre recording texture sound effects",
    "tempo rhythm meter beat bpm time signature",
    "key chords harmony progression tonality",
    "genre style mood emotion scene context",
]

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

SYSTEM_PROMPT = """You are a music annotation classifier.

Classify the sentence into ALL applicable labels.

Definitions:

0 = Instrumentation timbre recording texture sound effects
ONLY if the sentence describes instruments, sound texture, tone color, recording noise, acoustic qualities, or sound effects.

1 = Tempo rhythm meter beat bpm time signature
ONLY if the sentence describes BPM, rhythm, beat count, tempo, groove, meter, time signature.

2 = Key chords harmony progression tonality
ONLY if the sentence describes key, scale, tonality, chord names, chord progressions, harmony, harmonic structure.

3 = Genre style mood emotion scene context
ONLY if the sentence describes genre, stylistic tradition, production style, mixing style, era, mood, emotional affect, cinematic context, or scene.

Important:
- A sentence can have multiple labels.
- Do NOT force technical descriptions into genre.
- Do NOT confuse emotion with genre.
- Do NOT confuse scene with instrumentation.
- If no label fits, return [].

Return ONLY JSON."""

CLASSIFICATION_PROMPT = """Sentence: \"{sentence}\""""


def get_device(device: int = 0) -> str:
    """Get device string, falling back to CPU if CUDA unavailable."""
    if device >= 0 and torch.cuda.is_available():
        return f"cuda:{device}"
    return "cpu"


def build_qwen_classifier(model_name: Optional[str] = None, device: int = 0):
    """Load Qwen2.5-7B-Instruct model for multi-label classification."""
    device_str = get_device(device)
    model_path = model_name or DEFAULT_MODEL
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_str if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    
    return {"model": model, "tokenizer": tokenizer, "device": device_str}


INSTRUMENTATION_KEYWORDS = [
    "piano", "guitar", "drums", "bass", "violin", "synth", "flute", "vocal", "vocals",
    "reverb", "distortion", "ambient noise", "recording", "timbre", "texture",
]

TEMPO_KEYWORDS = [
    "bpm", "tempo", "beat", "beats per minute", "rhythm", "4/4", "3/4", "2/4", "6/8",
    "meter", "time signature", "allegro", "andante", "presto", "moderato", "largo",
]

HARMONY_KEYWORDS = [
    "chord", "chords", "progression", "key", "major", "minor", "maj7", "min7",
    "dim", "aug", "sus", "tonality", "scale",
]


def classify_by_rules(sentence: str) -> Optional[int]:
    """Return a rule-based label for the sentence if it matches a music keyword."""
    if not sentence or not sentence.strip():
        return None

    normalized = sentence.lower()

    for phrase in TEMPO_KEYWORDS:
        if phrase in normalized:
            return 1

    for phrase in HARMONY_KEYWORDS:
        if phrase in normalized:
            return 2

    for phrase in INSTRUMENTATION_KEYWORDS:
        if phrase in normalized:
            return 0

    return None


def parse_labels(output_text: str) -> List[int]:
    """
    Safely parse Qwen output and return a clean list of integer labels.

    Rejects malformed JSON, floats, non-list outputs, duplicates, and out-of-range labels.
    """
    output_text = output_text.strip()

    match = re.search(r'\[[^\]]*\]', output_text)
    if not match:
        return []

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    labels: List[int] = []
    for item in parsed:
        if not isinstance(item, int):
            return []
        if item < 0 or item >= len(LABELS):
            return []
        labels.append(item)

    return sorted(list(dict.fromkeys(labels)))


parse_model_output = parse_labels


def classify_sentences_batch(
    sentences: List[str],
    classifier_pipe: Dict,
    batch_size: int = 8,
) -> List[List[int]]:
    """
    Classify a batch of sentences using Qwen2.5-7B-Instruct.
    Returns multi-label classifications.
    """
    model = classifier_pipe["model"]
    tokenizer = classifier_pipe["tokenizer"]
    device = classifier_pipe["device"]
    
    results = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_results = []
        
        for sentence in batch:
            # Skip empty sentences
            if not sentence.strip():
                batch_results.append([])
                continue
            
            # Build conversation format for Qwen
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": CLASSIFICATION_PROMPT.format(sentence=sentence)}
            ]
            
            # Tokenize and generate
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=False,
                )
            
            # Decode output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the assistant's response (after the last occurrence of "Return ONLY the JSON array:")
            if "Return ONLY the JSON array:" in response:
                assistant_response = response.split("Return ONLY the JSON array:")[-1].strip()
            else:
                assistant_response = response
            
            # Parse labels
            labels = parse_labels(assistant_response)
            batch_results.append(labels)
        
        results.extend(batch_results)
    
    return results
