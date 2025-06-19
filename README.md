# MUSK

This repository provides a minimal version of the MUSK vision--language model. A small multi‐modal adapter (MMA) can be attached to the base model for additional training.

## Multi‐Modal Adapter Training

Two example training scripts are provided:

- `musk/contrastive_pretrain.py` – first‑stage contrastive training with the adapter
- `musk/mma_stage2.py` – optional stage‑two finetuning

Both scripts create a `MultiModalAdapter` and insert it into the MUSK model before the training loop.

Run the first stage with `accelerate`:

```bash
accelerate launch musk/contrastive_pretrain.py
```

This command will start a dummy MMA-based training run using randomly generated data. It serves as a template for real datasets and training configurations.
