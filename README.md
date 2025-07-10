
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
When a JSON file is used, the loader automatically reserves 10% of the samples
for validation and reports average MIM and MLM losses on this split each epoch.

Use `musk.json_dataset.get_json_loaders` to obtain training and validation loaders from the same file:
from musk.json_dataset import get_json_loaders
train_img_loader, val_img_loader = get_json_loaders("data.jsonl", mode="image", batch_size=64, num_workers=4)
train_txt_loader, val_txt_loader = get_json_loaders("data.jsonl", mode="text", batch_size=64, num_workers=4, tokenizer=tokenizer)
When `--json-data` is specified, 10% of the pairs are held out for validation
and the script prints contrastive and MLM losses for both splits each epoch.
`musk/models/tokenizer.spm`. Pass `--wandb-project <name>` to log
training metrics to Weights & Biases.
Specify `--wandb-project <name>` to log these metrics to Weights & Biases.

## UMAP visualization

Use `musk.umap_json` to visualize embeddings stored in a JSON lines file. Each line
should contain an `embedding` array and may include a `domain` label.
With `--cluster-domains <k>` the script clusters the embeddings using
KMeans before plotting and reports the V-measure when true domain labels are
present. Use `--kmeans-model model.pth` to load or save trained centroids.

Example:

```shell
# train clusters and save centroids
python -m musk.umap_json data.jsonl --cluster-domains 3 --kmeans-model model.pth --output umap.png
```
