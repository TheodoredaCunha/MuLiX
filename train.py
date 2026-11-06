import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from language_model import FLANT5TextEncoder
from musicmodel.contrastive import TextAudioContrastiveModel
from musicmodel.model import Model as AudioModel


class MusicBenchDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        labels_path: str,
        max_seconds: float = 10.0,
        audio_root: str = "dataset",
    ):
        self.root = Path(__file__).resolve().parent
        with (self.root / data_path).open("r", encoding="utf-8") as file:
            rows = json.load(file)
        with (self.root / labels_path).open("r", encoding="utf-8") as file:
            labels = json.load(file)

        if len(rows) != len(labels):
            raise ValueError(
                "Dataset rows and curriculum labels must have the same length "
                f"({len(rows)} rows, {len(labels)} label entries)"
            )

        self.rows = rows
        self.labels = labels
        self.max_samples = int(max_seconds * 24000)
        self.audio_root = Path(audio_root)
        if not self.audio_root.is_absolute():
            self.audio_root = self.root / self.audio_root
        self.difficulties = [self._difficulty_for_labels(lbls) for lbls in self.labels]

    @staticmethod
    def _difficulty_for_labels(row_labels: Sequence[int]) -> float:
        if not row_labels:
            return 0.0
        label_counts = Counter(row_labels)
        dominant_ratio = label_counts.most_common(1)[0][1] / len(row_labels)
        unique_labels = len(label_counts)
        return float((1.0 - dominant_ratio) + 0.25 * (unique_labels - 1))

    def curriculum_indices(self, epoch: int, total_epochs: int) -> List[int]:
        if total_epochs < 1:
            raise ValueError("total_epochs must be at least 1")
        if not 1 <= epoch <= total_epochs:
            raise ValueError("epoch must be between 1 and total_epochs")

        ordered_indices = sorted(range(len(self)), key=lambda idx: self.difficulties[idx])
        curriculum_fraction = min(1.0, max(0.2, epoch / total_epochs))
        cutoff = min(len(ordered_indices), max(2, math.ceil(curriculum_fraction * len(ordered_indices))))
        return ordered_indices[:cutoff]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        location = Path(row["location"])
        audio_candidates = (
            [location]
            if location.is_absolute()
            else [self.audio_root / location, self.audio_root / location.name]
        )
        audio_path = next((path for path in audio_candidates if path.is_file()), None)
        if audio_path is None:
            checked_paths = ", ".join(str(path) for path in audio_candidates)
            raise FileNotFoundError(
                f"Audio file for dataset row {index} was not found. "
                f"Checked: {checked_paths}. Set --audio-root to the audio dataset directory."
            )

        audio, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != 24000:
            audio = torchaudio.functional.resample(audio, sample_rate, 24000)
        audio = audio.mean(dim=0)
        audio = audio[: self.max_samples]
        if audio.numel() < self.max_samples:
            audio = torch.nn.functional.pad(audio, (0, self.max_samples - audio.numel()))

        row_labels = self.labels[index]
        curriculum_label = Counter(row_labels).most_common(1)[0][0] if row_labels else 0
        return audio, row["main_caption"], curriculum_label, self.difficulties[index]


def collate_batch(batch):
    audio, texts, labels, _ = zip(*batch)
    return torch.stack(audio), list(texts), list(labels)


def encode_audio_batch(audio_model: AudioModel, audio: torch.Tensor) -> Sequence[torch.Tensor]:
    h_audio = []
    z_audio = []
    with torch.no_grad():
        for waveform in audio:
            encoded = audio_model._encode_audio(
                waveform.unsqueeze(0),
                audio_model.encodec,
                audio_model.code_embedding,
                audio_model.audio_pool,
                audio_model.audio_proj,
            )
            z_audio.append(encoded[0].squeeze(0))
            h_audio.append(encoded[1].squeeze(0))
    return torch.stack(h_audio), torch.stack(z_audio)


def load_hyperparameters(config_path: str = "hyperparameters.json") -> Dict[str, Any]:
    config_file = Path(__file__).resolve().parent / config_path
    if not config_file.exists():
        return {}

    with config_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Hyperparameters config at {config_file} must contain a JSON object")
    return data


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for contrastive training")
    if args.max_seconds <= 0:
        raise ValueError("max_seconds must be greater than 0")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.log_every < 1:
        raise ValueError("log_every must be at least 1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = MusicBenchDataset(
        args.data,
        args.labels,
        args.max_seconds,
        args.audio_root,
    )
    if len(dataset) < 2:
        raise ValueError("At least two dataset rows are required for contrastive training")

    audio_model = AudioModel(train_encodec=False).to(device).eval()
    text_encoder = FLANT5TextEncoder(
        model_name=args.text_model,
        device=device,
        freeze=True,
    )
    model = TextAudioContrastiveModel(
        text_encoder=text_encoder,
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_indices = dataset.curriculum_indices(epoch, args.epochs)
        drop_last = (
            len(epoch_indices) > args.batch_size
            and len(epoch_indices) % args.batch_size == 1
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            sampler=torch.utils.data.SubsetRandomSampler(epoch_indices),
            collate_fn=collate_batch,
            drop_last=drop_last,
        )

        curriculum_fraction = len(epoch_indices) / max(len(dataset), 1)
        print(
            f"\n=== Curriculum stage: epoch {epoch}/{args.epochs} | "
            f"samples={len(epoch_indices)}/{len(dataset)} ({curriculum_fraction:.0%}) ==="
        )

        model.train()
        total_loss = 0.0
        progress = tqdm(
            enumerate(loader, start=1),
            total=len(loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for step, (audio, texts, labels) in progress:
            h_audio, z_audio = encode_audio_batch(audio_model, audio.to(device))
            outputs = model(texts, h_audio, z_audio, labels)
            optimizer.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            optimizer.step()
            total_loss += outputs["loss"].item()

            if step % args.log_every == 0:
                progress.set_postfix(loss=f"{outputs['loss'].item():.4f}")

        mean_loss = total_loss / max(len(loader), 1)
        print(f"epoch={epoch} curriculum_fraction={curriculum_fraction:.2f} mean_loss={mean_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "text_model": args.text_model,
            "embedding_dim": args.embedding_dim,
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    hyperparameters = load_hyperparameters()
    parser = argparse.ArgumentParser(description="Train the MuLiX text/audio contrastive model")
    parser.add_argument("--data", default=hyperparameters.get("data", "data/MusicBench_train.json"))
    parser.add_argument("--labels", default=hyperparameters.get("labels", "data/main_caption_classes.json"))
    parser.add_argument("--audio-root", default=hyperparameters.get("audio_root", "dataset"))
    parser.add_argument("--text-model", default=hyperparameters.get("text_model", "google/flan-t5-base"))
    parser.add_argument("--device", default=hyperparameters.get("device"))
    parser.add_argument("--epochs", type=int, default=hyperparameters.get("epochs", 1))
    parser.add_argument("--batch-size", type=int, default=hyperparameters.get("batch_size", 4))
    parser.add_argument("--workers", type=int, default=hyperparameters.get("workers", 0))
    parser.add_argument("--max-seconds", type=float, default=hyperparameters.get("max_seconds", 10.0))
    parser.add_argument("--learning-rate", type=float, default=hyperparameters.get("learning_rate", 1e-4))
    parser.add_argument("--embedding-dim", type=int, default=hyperparameters.get("embedding_dim", 256))
    parser.add_argument("--log-every", type=int, default=hyperparameters.get("log_every", 10))
    parser.add_argument("--output", default=hyperparameters.get("output", "checkpoints/mulix_text_audio.pt"))
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
