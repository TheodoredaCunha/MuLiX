import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from language_model import E5TextEncoder
from musicmodel.contrastive import TextAudioContrastiveModel
from train import configure_text_encoder_freezing


class E5EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokens = {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }
        with (
            patch("langauge_model.AutoTokenizer.from_pretrained") as tokenizer_factory,
            patch("langauge_model.AutoModel.from_pretrained", return_value=self.bert) as model_factory,
        ):
            self.tokenizer = tokenizer_factory.return_value
            self.tokenizer.return_value = self.tokens
            self.encoder = E5TextEncoder(freeze=False)
            model_factory.assert_called_once_with("intfloat/e5-base-v2", add_pooling_layer=False)

    def test_prefix_masked_pooling_and_single_text(self):
        result = self.encoder(["first", "second"], [1, 2])
        self.assertEqual(
            self.tokenizer.call_args.args[0],
            [
                "query: curriculum class 1; music caption: first",
                "query: curriculum class 2; music caption: second",
            ],
        )
        hidden = self.bert(**self.tokens).last_hidden_state
        mask = self.tokens["attention_mask"].unsqueeze(-1)
        expected = F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
        torch.testing.assert_close(result, expected)

        self.tokenizer.return_value = {name: value[:1] for name, value in self.tokens.items()}
        self.assertEqual(self.encoder("first").shape, (8,))
        self.assertEqual(self.tokenizer.call_args.args[0], ["query: music caption: first"])
        with self.assertRaisesRegex(ValueError, "one label per text"):
            self.encoder(["first", "second"], [1])

    def test_freeze_ratio_and_contrastive_gradient(self):
        self.assertEqual(configure_text_encoder_freezing(self.encoder, 0.34), (2, 3))
        self.assertFalse(any(p.requires_grad for p in self.bert.embeddings.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.bert.encoder.layer[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in self.bert.encoder.layer[1].parameters()))

        model = TextAudioContrastiveModel(text_encoder=self.encoder)
        output = model(["first", "second"], torch.randn(2, 128), torch.randn(2, 128))
        self.assertEqual(output["text_embedding"].shape, (2, 256))
        self.assertEqual(output["logits_audio_to_text"].shape, (2, 2))
        output["loss"].backward()
        self.assertIsNotNone(self.bert.encoder.layer[1].attention.self.query.weight.grad)
        self.assertIsNone(self.bert.encoder.layer[0].attention.self.query.weight.grad)

        self.assertEqual(configure_text_encoder_freezing(self.encoder, 1.0), (3, 3))
        self.assertFalse(any(p.requires_grad for p in self.bert.parameters()))
        self.assertEqual(configure_text_encoder_freezing(self.encoder, 0.0), (0, 3))
        self.assertTrue(all(p.requires_grad for p in self.bert.parameters()))


if __name__ == "__main__":
    unittest.main()
