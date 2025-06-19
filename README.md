# MUSK

This repository provides a minimal version of the MUSK vision--language model. A small multi‐modal adapter (MMA) can be attached to the base model for additional training.

## Multi‐Modal Adapter Training

Two example training scripts are provided:

- `musk/contrastive_pretrain.py` – first‑stage contrastive training with the adapter
- `musk/mma_stage2.py` – optional stage‑two finetuning

Both scripts create a `MultiModalAdapter` and insert it into the MUSK model before the training loop.

Run the first stage with `accelerate`:

```bash
accelerate launch musk/contrastive_pretrain.py --train-json train.json --wandb
```

This command will start a simple MMA-based training run using a JSON lines dataset of image/text pairs.

The second stage continues training from a Stage 1 checkpoint:

```bash
accelerate launch musk/mma_stage2.py --train-json train.json \
    --pretrained stage1.ckpt --wandb
```
