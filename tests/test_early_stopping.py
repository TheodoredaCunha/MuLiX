import argparse
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import train


class Progress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        pass


class EarlyStoppingTests(unittest.TestCase):
    def run_training(self, losses, patience, epochs=5):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                epochs=epochs, batch_size=2, max_seconds=10.0, workers=0,
                log_every=1, early_stopping_patience=patience,
                musicconrec_freeze_ratio=1.0, text_encoder_freeze_ratio=1.0,
                output=str(Path(directory) / "model.pt"), metrics_output=None,
                plot_output=None, device="cpu", val_data="val.json",
                val_labels="labels.json", label_model="label-model", data="train.json",
                labels="labels.json", audio_root=directory, musicconrec_weights="audio.pt",
                text_model="text-model", embedding_dim=256, learning_rate=0.001,
            )
            dataset = MagicMock()
            dataset.__len__.return_value = 2
            dataset.curriculum_indices.return_value = [0, 1]
            audio = MagicMock()
            audio.size.return_value = 2
            loss = MagicMock()
            loss.item.return_value = 1.0
            accuracy = MagicMock()
            accuracy.float.return_value.mean.return_value.item.return_value = 0.5
            logits = MagicMock()
            logits.argmax.return_value.__eq__.return_value = accuracy
            model = MagicMock()
            model.to.return_value = model
            model.parameters.return_value = []
            model.return_value = {
                "loss": loss,
                "logits_audio_to_text": logits,
                "logits_text_to_audio": logits,
            }
            audio_model = MagicMock()
            audio_model.to.return_value = audio_model
            audio_model.eval.return_value = audio_model
            audio_model.parameters.return_value = []
            optimizer = MagicMock()
            optimizer.param_groups = [{"lr": 0.001}]
            torch = MagicMock()
            torch.optim.AdamW.return_value = optimizer
            metrics = [
                {"loss": value, "retrieval_accuracy": 0.5, "samples_per_second": 1.0}
                for value in losses
            ]
            with (
                patch.object(train, "torch", torch),
                patch.object(train, "ensure_validation_labels", return_value="labels.json"),
                patch.object(train, "MusicBenchDataset", return_value=dataset),
                patch.object(train, "AudioModel", return_value=audio_model),
                patch.object(train, "load_musicconrec_weights"),
                patch.object(train, "configure_musicconrec_freezing", return_value=(1, 1)),
                patch.object(train, "E5TextEncoder"),
                patch.object(train, "configure_text_encoder_freezing", return_value=(1, 1)),
                patch.object(train, "TextAudioContrastiveModel", return_value=model),
                patch.object(train, "DataLoader", return_value=[(audio, ["a", "b"], [0, 0])]),
                patch.object(train, "encode_audio_batch", return_value=(MagicMock(), MagicMock())),
                patch.object(train, "validate", side_effect=metrics) as validate,
                patch.object(train, "save_training_plot"),
                patch.object(train, "tqdm", Progress),
            ):
                model.state_dict.side_effect = lambda: {"epoch": validate.call_count}
                audio_model.state_dict.side_effect = lambda: {"epoch": validate.call_count}
                if any(not math.isfinite(value) for value in losses):
                    with self.assertRaisesRegex(ValueError, "Non-finite validation loss"):
                        train.train(args)
                else:
                    train.train(args)
                saves = [call.args[0] for call in torch.save.call_args_list]
                return validate.call_count, saves

    def test_stops_after_patience_and_saves_both_models_at_best_epochs(self):
        count, saves = self.run_training([3.0, 2.0, 2.0, 2.5, 1.0], patience=2)
        self.assertEqual(count, 4)
        self.assertEqual([save["model"]["epoch"] for save in saves], [1, 2])
        self.assertEqual([save["musicconrec_model"]["epoch"] for save in saves], [1, 2])

    def test_improvement_resets_patience(self):
        count, saves = self.run_training([3.0, 4.0, 2.0, 3.0, 4.0], patience=2)
        self.assertEqual(count, 5)
        self.assertEqual([save["model"]["epoch"] for save in saves], [1, 3])

    def test_disabled_saves_only_final_epoch(self):
        count, saves = self.run_training([1.0, 2.0, 3.0], patience=0, epochs=3)
        self.assertEqual(count, 3)
        self.assertEqual([save["model"]["epoch"] for save in saves], [3])

    def test_non_finite_loss_fails_without_overwriting_best(self):
        count, saves = self.run_training([1.0, float("nan")], patience=2)
        self.assertEqual(count, 2)
        self.assertEqual([save["model"]["epoch"] for save in saves], [1])


if __name__ == "__main__":
    unittest.main()
