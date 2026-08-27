import torch
import torchaudio

from utils.loading import load_model
from preprocessing.audio_preprocess import resample


class MusicConRec:
    def __init__(self, device):
        self.model = load_model().to(device)
        self.model.eval()

        self.encodec = self.model.encodec
        self.code_embedding = self.model.code_embedding

        self.audio_pool = self.model.audio_pool
        self.audio_proj = self.model.audio_proj

    @torch.no_grad()
    def encode_audio(self, path: str):
        """
        Args:
            path: path to audio file

        Returns:
            h_audio: (B, D)
            z_audio: (B, proj_dim)
        """

        # =====================================
        # LOAD AUDIO
        # =====================================
        audio, sr = torchaudio.load(path, backend="soundfile")

        # audio:
        # (C, T)

        # =====================================
        # RESAMPLE
        # =====================================
        if sr != 24000:
            audio, sr = resample(
                audio,
                sr=sr,
                target_sr=24000
            )

        # =====================================
        # ENCODE WITH ENCODEC
        # =====================================
        audio.unsqueeze_(0)  # imitate batch dimension
        encoder_outputs = self.encodec.encode(audio)

        audio_codes = encoder_outputs["audio_codes"].long()

        # =====================================
        # HANDLE DIFFERENT ENCODEC SHAPES
        # =====================================

        # Common HF shape:
        # (B, chunks, Q, T)

        if audio_codes.dim() == 4:
            if audio_codes.shape[1] == 1:
                audio_codes = audio_codes.squeeze(1)
            else:
                # merge chunks into time dimension
                B, chunks, Q, T = audio_codes.shape

                audio_codes = (
                    audio_codes.permute(0, 2, 1, 3)
                    .reshape(B, Q, chunks * T)
                )

        # Expected final shape:
        # (B, Q, T)

        # =====================================
        # TOKEN EMBEDDINGS
        # =====================================
        codes = audio_codes.permute(0, 2, 1)

        # (B, T, Q)

        codes = self.code_embedding(codes)

        # (B, T, Q, D)

        codes = codes.mean(dim=2)

        # (B, T, D)

        # =====================================
        # POOL + PROJECT
        # =====================================
        h_audio = self.audio_pool(codes)

        # (B, D)

        z_audio = self.audio_proj(h_audio)

        # (B, proj_dim)

        return h_audio, z_audio
    
    @torch.no_grad()
    def encode_waveform(self, audio, sr):
        """
        audio: (C, T)
        sr: sample rate

        Returns:
            h_audio: (1, D)
            z_audio: (1, proj_dim)
        """

        if sr != 24000:
            audio, sr = resample(audio, sr=sr, target_sr=24000)

        audio = audio.unsqueeze(0)

        encoder_outputs = self.encodec.encode(audio)
        audio_codes = encoder_outputs["audio_codes"].long()

        if audio_codes.dim() == 4:
            if audio_codes.shape[1] == 1:
                audio_codes = audio_codes.squeeze(1)
            else:
                B, chunks, Q, T = audio_codes.shape
                audio_codes = (
                    audio_codes.permute(0, 2, 1, 3)
                    .reshape(B, Q, chunks * T)
                )

        codes = audio_codes.permute(0, 2, 1)
        codes = self.code_embedding(codes)
        codes = codes.mean(dim=2)

        h_audio = self.audio_pool(codes)
        z_audio = self.audio_proj(h_audio)

        return h_audio, z_audio
    
    def __print__(self):
        return f"MusicConRec(model={self.model})"