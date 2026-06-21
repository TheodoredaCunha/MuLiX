import json
import re
import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

LABELS: List[str] = [
    "instrumentation and timbre",
    "tempo rhythm beat meter",
    "key chords harmony progression",
    "genre style production",
    "emotion mood feeling",
    "scene imagery context",
]

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

SYSTEM_PROMPT = """You are a music annotation classifier.

Classify the sentence into ALL applicable labels.

Definitions:

0 = Instrumentation and timbre
ONLY if the sentence describes instruments, sound texture, tone color, recording noise, acoustic qualities, or sound effects.

1 = Tempo rhythm beat meter
ONLY if the sentence describes BPM, rhythm, beat count, tempo, groove, meter, time signature.

2 = Key chords harmony progression
ONLY if the sentence describes key, scale, tonality, chord names, chord progressions, harmony, harmonic structure.

3 = Genre style production
ONLY if the sentence describes genre, stylistic tradition, production style, mixing style, era.

4 = Emotion mood feeling
ONLY if the sentence describes feelings, emotional affect, mood, energy, tension.

5 = Scene imagery context
ONLY if the sentence describes situations, environments, visual scenes, places, use-cases, cinematic contexts.

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


def parse_model_output(output_text: str) -> List[int]:
    """
    Robustly parse model output to extract label integers.
    Handles various formats and malformed outputs.
    """
    output_text = output_text.strip()
    
    # Try to extract JSON array pattern
    json_match = re.search(r'\[[\d\s,]*\]', output_text)
    if json_match:
        try:
            labels = json.loads(json_match.group())
            # Validate: must be list of integers in range [0, len(LABELS))
            if isinstance(labels, list) and all(isinstance(x, int) and 0 <= x < len(LABELS) for x in labels):
                return sorted(list(set(labels)))  # Remove duplicates and sort
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: extract individual digits
    digits = re.findall(r'\b[0-5]\b', output_text)
    if digits:
        labels = sorted(list(set(int(d) for d in digits)))
        return labels
    
    # Empty output or invalid
    return []


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
            labels = parse_model_output(assistant_response)
            batch_results.append(labels)
        
        results.extend(batch_results)
    
    return results
