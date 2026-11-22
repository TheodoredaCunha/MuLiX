import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def ensure_validation_labels(
    data_path: str,
    labels_path: str,
    model_name: str,
    device: torch.device,
) -> str:
    project_root = Path(__file__).resolve().parent
    resolved_labels_path = Path(labels_path)
    if not resolved_labels_path.is_absolute():
        resolved_labels_path = project_root / resolved_labels_path
    if resolved_labels_path.is_file():
        return labels_path

    from preprocessing.curriculum.classify_text import generate_caption_labels

    alt_labels_path = resolved_labels_path.with_name("alt_caption_val_classes.json")
    print(
        f"Validation labels not found at {resolved_labels_path}; "
        f"generating them with {model_name}."
    )
    generate_caption_labels(
        data_path=data_path,
        model_name=model_name,
        device=str(device),
        save_paths={
            "main_caption": str(resolved_labels_path),
            "alt_caption": str(alt_labels_path),
        },
        make_reports=False,
    )
    return labels_path


def collate_batch(batch):
    audio, texts, labels, _ = zip(*batch)
    return torch.stack(audio), list(texts), list(labels)


def encode_audio_batch(
    audio_model: AudioModel,
    audio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_audio = []
    z_audio = []
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


def load_musicconrec_weights(audio_model: AudioModel, weights_path: Path) -> None:
    if not weights_path.is_file():
        raise FileNotFoundError(f"MusicConRec weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state-dict checkpoint at {weights_path}")

    state_dict = checkpoint
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(checkpoint.get(key), dict):
            state_dict = checkpoint[key]
            break

    if any(not isinstance(key, str) for key in state_dict):
        raise TypeError(f"Checkpoint keys must be strings: {weights_path}")
    for prefix in ("module.", "model."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix):]: value for key, value in state_dict.items()
            }

    audio_model.load_state_dict(state_dict, strict=True)


def _apply_freeze_ratio(
    layer_groups: Sequence[Sequence[torch.nn.Module]],
    freeze_ratio: float,
) -> int:
    if not math.isfinite(freeze_ratio) or not 0.0 <= freeze_ratio <= 1.0:
        raise ValueError("Freeze ratios must be between 0.0 and 1.0")

    frozen_groups = math.ceil(freeze_ratio * len(layer_groups))
    for index, group in enumerate(layer_groups):
        for module in group:
            for parameter in module.parameters():
                parameter.requires_grad = index >= frozen_groups
    return frozen_groups


def configure_musicconrec_freezing(
    audio_model: AudioModel,
    freeze_ratio: float,
) -> Tuple[int, int]:
    for parameter in audio_model.parameters():
        parameter.requires_grad = False

    # Encodec emits discrete codes here, so only the downstream audio layers
    # can receive gradients in this training path.
    layer_groups = [
        [audio_model.code_embedding],
        [audio_model.audio_pool.attn[0]],
        [audio_model.audio_pool.attn[2]],
        [audio_model.audio_proj.net[0], audio_model.audio_proj.net[1]],
        [audio_model.audio_proj.net[3]],
    ]
    return _apply_freeze_ratio(layer_groups, freeze_ratio), len(layer_groups)


def configure_flan_t5_freezing(
    text_encoder: FLANT5TextEncoder,
    freeze_ratio: float,
) -> Tuple[int, int]:
    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    t5_encoder = text_encoder.encoder.encoder
    layer_groups = (
        [[t5_encoder.embed_tokens]]
        + [[layer] for layer in t5_encoder.block]
        + [[t5_encoder.final_layer_norm]]
    )
    return _apply_freeze_ratio(layer_groups, freeze_ratio), len(layer_groups)


@torch.no_grad()
def validate(
    model: TextAudioContrastiveModel,
    audio_model: AudioModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    audio_model.eval()
    total_loss = 0.0
    total_retrieval_accuracy = 0.0
    total_examples = 0
    started_at = time.perf_counter()

    for audio, texts, labels in tqdm(
        loader,
        desc="Validation",
        unit="batch",
        leave=False,
    ):
        batch_size = audio.size(0)
        h_audio, z_audio = encode_audio_batch(audio_model, audio.to(device))
        outputs = model(texts, h_audio, z_audio, labels)
        batch_loss = outputs["loss"].item()
        targets = torch.arange(batch_size, device=device)
        audio_to_text_accuracy = (
            outputs["logits_audio_to_text"].argmax(dim=1) == targets
        ).float().mean().item()
        text_to_audio_accuracy = (
            outputs["logits_text_to_audio"].argmax(dim=1) == targets
        ).float().mean().item()
        retrieval_accuracy = (audio_to_text_accuracy + text_to_audio_accuracy) / 2

        total_loss += batch_loss * batch_size
        total_retrieval_accuracy += retrieval_accuracy * batch_size
        total_examples += batch_size

    duration = max(time.perf_counter() - started_at, 1e-8)
    if total_examples == 0:
        raise ValueError("Validation dataset produced no batches")
    return {
        "loss": total_loss / total_examples,
        "retrieval_accuracy": total_retrieval_accuracy / total_examples,
        "samples_per_second": total_examples / duration,
    }


def save_training_plot(metrics_path: Path, plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    with metrics_path.open("r", newline="", encoding="utf-8") as metrics_file:
        epoch_rows = [
            row
            for row in csv.DictReader(metrics_file)
            if row["event"] == "epoch"
        ]
    if not epoch_rows:
        raise ValueError(f"No epoch metrics found in {metrics_path}")

    epochs = [int(row["epoch"]) for row in epoch_rows]
    train_losses = [float(row["mean_loss"]) for row in epoch_rows]
    validation_losses = [float(row["validation_loss"]) for row in epoch_rows]
    train_accuracies = [float(row["retrieval_accuracy"]) for row in epoch_rows]
    validation_accuracies = [
        float(row["validation_retrieval_accuracy"]) for row in epoch_rows
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_losses, marker="o", label="Training")
    axes[0].plot(epochs, validation_losses, marker="o", label="Validation")
    axes[0].set_title("Contrastive loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracies, marker="o", label="Training")
    axes[1].plot(epochs, validation_accuracies, marker="o", label="Validation")
    axes[1].set_title("In-batch retrieval accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


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
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    for name in ("musicconrec_freeze_ratio", "flan_t5_freeze_ratio"):
        ratio = getattr(args, name)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name.replace('_', ' ')} must be between 0.0 and 1.0")
    output_path = Path(args.output)
    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else output_path.with_suffix(".metrics.csv")
    )
    if metrics_path.resolve() == output_path.resolve():
        raise ValueError("metrics output path must be different from the model output path")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    val_labels_path = ensure_validation_labels(
        args.val_data,
        args.val_labels,
        args.label_model,
        device,
    )
    dataset = MusicBenchDataset(
        args.data,
        args.labels,
        args.max_seconds,
        args.audio_root,
    )
    validation_dataset = MusicBenchDataset(
        args.val_data,
        val_labels_path,
        args.max_seconds,
        args.audio_root,
    )
    if len(dataset) < 2:
        raise ValueError("At least two dataset rows are required for contrastive training")
    if len(validation_dataset) < 2:
        raise ValueError("At least two validation rows are required for contrastive validation")

    project_root = Path(__file__).resolve().parent
    musicconrec_weights = Path(args.musicconrec_weights)
    if not musicconrec_weights.is_absolute():
        musicconrec_weights = project_root / musicconrec_weights

    audio_model = AudioModel(train_encodec=False)
    load_musicconrec_weights(audio_model, musicconrec_weights)
    musicconrec_frozen, musicconrec_layer_count = configure_musicconrec_freezing(
        audio_model,
        args.musicconrec_freeze_ratio,
    )
    audio_model = audio_model.to(device).eval()

    text_encoder = FLANT5TextEncoder(
        model_name=args.text_model,
        device=device,
        freeze=False,
    )
    flan_t5_frozen, flan_t5_layer_count = configure_flan_t5_freezing(
        text_encoder,
        args.flan_t5_freeze_ratio,
    )
    model = TextAudioContrastiveModel(
        text_encoder=text_encoder,
        embedding_dim=args.embedding_dim,
    ).to(device)
    print(
        f"MusicConRec frozen audio layers: {musicconrec_frozen}/{musicconrec_layer_count}; "
        f"FLAN-T5 frozen layers: {flan_t5_frozen}/{flan_t5_layer_count}"
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_parameters.extend(
        parameter
        for parameter in audio_model.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )
    validation_batch_size = min(args.batch_size, len(validation_dataset))
    if len(validation_dataset) % validation_batch_size == 1:
        validation_batch_size += 1
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = (
        "event",
        "epoch",
        "step",
        "global_step",
        "batch_size",
        "batch_loss",
        "mean_loss",
        "validation_loss",
        "retrieval_accuracy",
        "validation_retrieval_accuracy",
        "learning_rate",
        "samples_per_second",
        "validation_samples_per_second",
        "curriculum_fraction",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        csv.DictWriter(metrics_file, fieldnames=metric_fields).writeheader()

    def write_metrics(row: Dict[str, Any]) -> None:
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            csv.DictWriter(metrics_file, fieldnames=metric_fields).writerow(row)

    global_step = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint() -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "musicconrec_model": audio_model.state_dict(),
                "text_model": args.text_model,
                "embedding_dim": args.embedding_dim,
                "musicconrec_weights": str(musicconrec_weights),
                "musicconrec_freeze_ratio": args.musicconrec_freeze_ratio,
                "flan_t5_freeze_ratio": args.flan_t5_freeze_ratio,
                "metrics_output": str(metrics_path),
            },
            output_path,
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
        total_retrieval_accuracy = 0.0
        total_examples = 0
        epoch_started = time.perf_counter()
        last_log_time = epoch_started
        examples_since_log = 0
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
            batch_size = audio.size(0)
            batch_loss = outputs["loss"].item()
            targets = torch.arange(batch_size, device=device)
            audio_to_text_accuracy = (
                outputs["logits_audio_to_text"].argmax(dim=1) == targets
            ).float().mean().item()
            text_to_audio_accuracy = (
                outputs["logits_text_to_audio"].argmax(dim=1) == targets
            ).float().mean().item()
            retrieval_accuracy = (audio_to_text_accuracy + text_to_audio_accuracy) / 2

            total_loss += batch_loss * batch_size
            total_retrieval_accuracy += retrieval_accuracy * batch_size
            total_examples += batch_size
            examples_since_log += batch_size
            global_step += 1

            if step % args.log_every == 0 or step == len(loader):
                now = time.perf_counter()
                mean_loss = total_loss / total_examples
                mean_retrieval_accuracy = total_retrieval_accuracy / total_examples
                learning_rate = optimizer.param_groups[0]["lr"]
                samples_per_second = examples_since_log / max(now - last_log_time, 1e-8)
                progress.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    mean_loss=f"{mean_loss:.4f}",
                    retrieval_acc=f"{mean_retrieval_accuracy:.3f}",
                    lr=f"{learning_rate:.2e}",
                    samples_per_sec=f"{samples_per_second:.1f}",
                )
                write_metrics(
                    {
                        "event": "step",
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "batch_size": batch_size,
                        "batch_loss": batch_loss,
                        "mean_loss": mean_loss,
                        "validation_loss": "",
                        "retrieval_accuracy": mean_retrieval_accuracy,
                        "validation_retrieval_accuracy": "",
                        "learning_rate": learning_rate,
                        "samples_per_second": samples_per_second,
                        "validation_samples_per_second": "",
                        "curriculum_fraction": curriculum_fraction,
                    }
                )
                last_log_time = now
                examples_since_log = 0

        epoch_duration = time.perf_counter() - epoch_started
        mean_loss = total_loss / max(total_examples, 1)
        mean_retrieval_accuracy = total_retrieval_accuracy / max(total_examples, 1)
        learning_rate = optimizer.param_groups[0]["lr"]
        samples_per_second = total_examples / max(epoch_duration, 1e-8)
        validation_metrics = validate(model, audio_model, validation_loader, device)
        if not math.isfinite(validation_metrics["loss"]):
            raise ValueError(f"Non-finite validation loss at epoch {epoch}")
        print(
            f"epoch={epoch} curriculum_fraction={curriculum_fraction:.2f} "
            f"train_loss={mean_loss:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"train_retrieval_accuracy={mean_retrieval_accuracy:.3f} "
            f"val_retrieval_accuracy={validation_metrics['retrieval_accuracy']:.3f} "
            f"samples_per_sec={samples_per_second:.1f}"
        )
        write_metrics(
            {
                "event": "epoch",
                "epoch": epoch,
                "step": len(loader),
                "global_step": global_step,
                "batch_size": total_examples,
                "batch_loss": "",
                "mean_loss": mean_loss,
                "validation_loss": validation_metrics["loss"],
                "retrieval_accuracy": mean_retrieval_accuracy,
                "validation_retrieval_accuracy": validation_metrics["retrieval_accuracy"],
                "learning_rate": learning_rate,
                "samples_per_second": samples_per_second,
                "validation_samples_per_second": validation_metrics["samples_per_second"],
                "curriculum_fraction": curriculum_fraction,
            }
        )
        if args.early_stopping_patience:
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                epochs_without_improvement = 0
                save_checkpoint()
                print(f"Saved best checkpoint at epoch {epoch} (val_loss={best_validation_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch}: validation loss did not improve "
                        f"for {args.early_stopping_patience} epochs."
                    )
                    break

    if not args.early_stopping_patience:
        save_checkpoint()
    print(f"Saved checkpoint to {output_path}")
    print(f"Saved training metrics to {metrics_path}")
    plot_path = (
        Path(args.plot_output)
        if args.plot_output
        else output_path.with_suffix(".training.png")
    )
    save_training_plot(metrics_path, plot_path)
    print(f"Saved training graph to {plot_path}")


def build_parser() -> argparse.ArgumentParser:
    hyperparameters = load_hyperparameters()
    parser = argparse.ArgumentParser(description="Train the MuLiX text/audio contrastive model")
    parser.add_argument("--data", default=hyperparameters.get("data", "data/MusicBench_train.json"))
    parser.add_argument("--labels", default=hyperparameters.get("labels", "data/main_caption_classes.json"))
    parser.add_argument(
        "--val-data",
        default=hyperparameters.get("val_data", "data/MusicBench_val.json"),
    )
    parser.add_argument(
        "--val-labels",
        default=hyperparameters.get(
            "val_labels",
            "data/main_caption_val_classes.json",
        ),
    )
    parser.add_argument(
        "--label-model",
        default=hyperparameters.get("label_model", "BAAI/bge-base-en-v1.5"),
        help="SentenceTransformer used to generate missing validation curriculum labels",
    )
    parser.add_argument("--audio-root", default=hyperparameters.get("audio_root", "dataset"))
    parser.add_argument("--text-model", default=hyperparameters.get("text_model", "google/flan-t5-base"))
    parser.add_argument(
        "--musicconrec-weights",
        default=hyperparameters.get(
            "musicconrec_weights",
            "musicconrec_weights/best_model.pth",
        ),
        help="MusicConRec checkpoint used to initialize the audio model",
    )
    parser.add_argument(
        "--musicconrec-freeze-ratio",
        type=float,
        default=hyperparameters.get("musicconrec_freeze_ratio", 1.0),
        help=(
            "Fraction of ordered MusicConRec audio-path layers to freeze "
            "(0=train all, 1=freeze all; Encodec remains frozen)"
        ),
    )
    parser.add_argument(
        "--flan-t5-freeze-ratio",
        type=float,
        default=hyperparameters.get("flan_t5_freeze_ratio", 1.0),
        help=(
            "Fraction of ordered FLAN-T5 layers to freeze, from embeddings through "
            "encoder blocks (0=train all, 1=freeze all)"
        ),
    )
    parser.add_argument("--device", default=hyperparameters.get("device"))
    parser.add_argument("--epochs", type=int, default=hyperparameters.get("epochs", 1))
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=hyperparameters.get("early_stopping_patience", 5),
        help=(
            "Stop after N epochs without a validation-loss improvement and save "
            "the best checkpoint; 0 disables early stopping and saves the final epoch"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=hyperparameters.get("batch_size", 4))
    parser.add_argument("--workers", type=int, default=hyperparameters.get("workers", 0))
    parser.add_argument("--max-seconds", type=float, default=hyperparameters.get("max_seconds", 10.0))
    parser.add_argument("--learning-rate", type=float, default=hyperparameters.get("learning_rate", 1e-4))
    parser.add_argument("--embedding-dim", type=int, default=hyperparameters.get("embedding_dim", 256))
    parser.add_argument(
        "--log-every",
        type=int,
        default=hyperparameters.get("log_every", 10),
        help="Log batch metrics every N batches and always on each epoch's final batch",
    )
    parser.add_argument("--output", default=hyperparameters.get("output", "checkpoints/mulix_text_audio.pt"))
    parser.add_argument(
        "--metrics-output",
        default=hyperparameters.get("metrics_output"),
        help="CSV training log path (defaults to the model checkpoint path with .metrics.csv)",
    )
    parser.add_argument(
        "--plot-output",
        default=hyperparameters.get("plot_output"),
        help="Training graph path (defaults to the model checkpoint path with .training.png)",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
