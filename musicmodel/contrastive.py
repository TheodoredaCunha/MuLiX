from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from language_model import FLANT5TextEncoder


class TextAudioContrastiveModel(nn.Module):
    """Contrast curriculum-conditioned text with concatenated audio features."""

    def __init__(
        self,
        audio_hidden_dim: int = 128,
        audio_projection_dim: int = 128,
        embedding_dim: int = 256,
        text_encoder: Optional[FLANT5TextEncoder] = None,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.text_encoder = text_encoder or FLANT5TextEncoder()
        self.temperature = temperature
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_hidden_dim + audio_projection_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_audio(self, h_audio: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        if h_audio.ndim != 2 or z_audio.ndim != 2:
            raise ValueError("h_audio and z_audio must have shape (batch, features)")
        if h_audio.shape[0] != z_audio.shape[0]:
            raise ValueError("h_audio and z_audio must have the same batch size")
        return F.normalize(
            self.audio_projection(torch.cat((h_audio, z_audio), dim=-1)),
            dim=-1,
        )

    def forward(
        self,
        texts: Union[str, Sequence[str]],
        h_audio: torch.Tensor,
        z_audio: torch.Tensor,
        curriculum_labels: Optional[Sequence[Union[int, str]]] = None,
    ):
        text_features = self.text_encoder(texts, curriculum_labels)
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)
        text_features = F.normalize(self.text_projection(text_features), dim=-1)
        audio_features = self.encode_audio(h_audio, z_audio)
        if text_features.shape[0] != audio_features.shape[0]:
            raise ValueError("texts, h_audio, and z_audio must have the same batch size")

        logits = audio_features @ text_features.transpose(0, 1)
        logits = logits / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.transpose(0, 1), targets)
        ) / 2
        return {
            "loss": loss,
            "logits_audio_to_text": logits,
            "logits_text_to_audio": logits.transpose(0, 1),
            "text_embedding": text_features,
            "audio_embedding": audio_features,
        }


Model = TextAudioContrastiveModel