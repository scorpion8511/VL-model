MUSK performs vision--language pretraining for precision oncology. After a
masked image and language modeling stage, the second stage aligns paired images
and texts with a CLIP-style contrastive objective.

## Stage-two pretraining

Run the contrastive script with either WebDataset shards or a JSONL file of
`{"image": ..., "text": ...}` pairs:

```bash
accelerate launch -m musk.contrastive_pretrain \
    --pair-data /path/to/pairs/{0000..0100}.tar \
    --epochs 20 --output musk_stage2.pt
```

```bash
accelerate launch -m musk.contrastive_pretrain \
    --json-data pairs.jsonl \
    --epochs 20 --output musk_stage2.pt
```

The script minimizes a contrastive loss plus an auxiliary masked language
modeling (MLM) loss through a cross-attention decoder. Distributed training
requires `find_unused_parameters=True` because the auxiliary path does not use
all parameters every step.

## Stage-one encoder

`musk.pretrain` trains a shared encoder with masked modeling. Save only the
encoder weights with `--encoder-out` and load them into stage two using
`--encoder`:

```bash
accelerate launch -m musk.pretrain \
    --epochs 5 --encoder-out musk_pretrained_encoder.pt
```

```bash
accelerate launch -m musk.contrastive_pretrain \
    --encoder musk_pretrained_encoder.pt \
    --epochs 20 --output musk_stage2.pt
```

When a JSONL file is used, 10% of the pairs are reserved for validation and the
script reports average contrastive and MLM losses on this split each epoch. You
can also call `musk.json_dataset.get_json_loaders("data.jsonl")` to obtain the
same loaders programmatically.

Specify `--wandb-project <name>` to log metrics to Weights & Biases.
