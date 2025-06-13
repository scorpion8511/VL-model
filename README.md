
text captions. The script uses [HuggingFace Accelerate](https://github.com/huggingface/accelerate)
for device management, so run it with `accelerate launch` to enable multi‑GPU
training when available.
accelerate launch -m musk.pretrain \
accelerate launch -m musk.pretrain \

### Stage-two: Contrastive Pretraining

After stage-one masked modeling, MUSK aligns modalities with a contrastive
objective on paired image–text data. The repository provides
`musk.contrastive_pretrain` as a lightweight reference.

Using WebDataset shards of paired samples:

```shell
accelerate launch -m musk.contrastive_pretrain \
       --pair-data /path/to/pairs/{0000..0100}.tar \
       --epochs 20 --output musk_stage2.pt
```

Using a JSON lines file with `image` and `text` fields:

```shell
accelerate launch -m musk.contrastive_pretrain \
       --json-data pairs.jsonl \
       --epochs 20 --output musk_stage2.pt
```

The script minimizes a CLIP-style contrastive loss plus an auxiliary MLM loss
via a cross-attention decoder and reports both losses every epoch.
training when available. The scripts configure DDP with
`find_unused_parameters=True` to accommodate the two-step loss computation.
       --epochs 5 --output musk_pretrained.pt \
       --encoder-out musk_pretrained_encoder.pt
       --epochs 5 --output musk_pretrained.pt \
       --encoder-out musk_pretrained_encoder.pt
Optionally specify `--encoder-out` to save only the shared encoder weights for
use in stage-two contrastive pretraining.
       --encoder musk_pretrained_encoder.pt \
       --encoder musk_pretrained_encoder.pt \
accelerate launch --mixed_precision fp16 -m musk.contrastive_pretrain \
       --batch-size 16 --epochs 20 --output musk_stage2.pt
accelerate launch --mixed_precision fp16 -m musk.contrastive_pretrain \
       --batch-size 16 --epochs 20 --output musk_stage2.pt
