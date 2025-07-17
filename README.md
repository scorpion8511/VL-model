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

The script minimizes a CLIP-style contrastive loss plus an auxiliary MLM loss.
Enable caption generation or image reconstruction objectives with `--caption-loss`
and `--recon-loss`. Add `--domain-loss` to classify the domain name during training.

Stage-one pretraining example:

```shell
accelerate launch -m musk.pretrain \
       --json-data data.jsonl \
       --epochs 5 --output musk_pretrained.pt \
       --encoder-out musk_pretrained_encoder.pt
```

Initialize from a domain-specific checkpoint:

```shell
accelerate launch -m musk.pretrain \
       --json-data data.jsonl \
       --domain /path/endo.pth --epochs 5 --output musk_pretrained.pt
```

Built-in domain encoders are available for X-ray and MRI images:

```python
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
model = AutoModel.from_pretrained("microsoft/rad-dino")

from diffusers.models import AutoencoderKL
autoencoder = AutoencoderKL.from_pretrained("microsoft/mri-autoencoder-v0.1")
```

Use `--domain xray` or `--domain mri` to initialize from these defaults.

Use multiple domain teacher encoders for distillation with `--domains` and
enable Mixture-of-Experts layers using `--moe-freq` and `--num-experts`:

```shell
accelerate launch -m musk.pretrain \
       --json-data data.jsonl \
       --moe-freq 4 --num-experts 4 \
       --domains mri,xray=/path/xray.pth,patho=/path/patho.pth,endo=/path/endo.pth \
       --epochs 5 --output musk_pretrained.pt --domain-loss
```

Add `--domain-loss` to train a classifier predicting the domain label.

After stage one run contrastive pretraining:

```shell
accelerate launch --mixed_precision fp16 -m musk.contrastive_pretrain \
       --batch-size 16 --epochs 20 --output musk_stage2.pt \
       --encoder musk_pretrained_encoder.pt
```

When a JSON file is used, 10% of the samples are held out for validation and the
script reports average losses on this split each epoch. Obtain loaders
programmatically:

```python
from musk.json_dataset import get_json_loaders
train_img_loader, val_img_loader = get_json_loaders("data.jsonl", mode="image", batch_size=64, num_workers=4)
train_txt_loader, val_txt_loader = get_json_loaders("data.jsonl", mode="text", batch_size=64, num_workers=4, tokenizer=tokenizer)
```

Specify `--wandb-project <name>` to log metrics to Weights & Biases. The tokenizer
file is located at `musk/models/tokenizer.spm`.
