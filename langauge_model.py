from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel


TextInput = Union[str, Sequence[str]]
LabelInput = Optional[Sequence[Union[int, str]]]


class FLANT5TextEncoder(nn.Module):
	"""Encode curriculum-conditioned text with the FLAN-T5 encoder."""

	def __init__(
		self,
		model_name: str = "google/flan-t5-base",
		device: Optional[Union[str, torch.device]] = None,
		freeze: bool = True,
	):
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.encoder = T5EncoderModel.from_pretrained(model_name)
		self.embedding_dim = self.encoder.config.d_model

		if freeze:
			for parameter in self.encoder.parameters():
				parameter.requires_grad = False

		if device is not None:
			self.to(device)

	@staticmethod
	def curriculum_texts(
		texts: Sequence[str],
		curriculum_labels: LabelInput = None,
	) -> List[str]:
		if curriculum_labels is not None and len(texts) != len(curriculum_labels):
			raise ValueError("curriculum_labels must have one label per text")

		if curriculum_labels is None:
			return [f"music caption: {text}" for text in texts]

		return [
			f"curriculum class {label}; music caption: {text}"
			for text, label in zip(texts, curriculum_labels)
		]

	def forward(
		self,
		texts: TextInput,
		curriculum_labels: LabelInput = None,
		max_length: int = 128,
	) -> torch.Tensor:
		single_text = isinstance(texts, str)
		text_batch = [texts] if single_text else list(texts)
		prompted_texts = self.curriculum_texts(text_batch, curriculum_labels)

		device = next(self.encoder.parameters()).device
		tokens = self.tokenizer(
			prompted_texts,
			padding=True,
			truncation=True,
			max_length=max_length,
			return_tensors="pt",
		)
		tokens = {name: value.to(device) for name, value in tokens.items()}

		outputs = self.encoder(**tokens)
		hidden_states = outputs.last_hidden_state
		mask = tokens["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
		embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
		embeddings = F.normalize(embeddings, dim=-1)

		return embeddings[0] if single_text else embeddings


LanguageModel = FLANT5TextEncoder
