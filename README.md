# MuLiX training

The text encoder defaults to `intfloat/e5-base-v2`. It adds the `query:`
prefix to captions, including curriculum-conditioned captions, and uses masked
mean pooling. Use the same preprocessing at inference time.

Run training with `python main.py train`. Set `text_encoder_freeze_ratio` in
`hyperparameters.json`, or pass `--text-encoder-freeze-ratio`, to freeze a
fraction of E5's layer groups in order: embeddings, then Transformer blocks.
`0` trains all groups and `1` freezes all groups. The old
`flan_t5_freeze_ratio` config key and `--flan-t5-freeze-ratio` option remain
supported as aliases.

E5 and FLAN-T5 checkpoints have different text-encoder weights and are not
interchangeable. Train a new checkpoint after changing encoders.
