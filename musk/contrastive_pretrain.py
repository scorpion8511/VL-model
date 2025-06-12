"""Stage-two contrastive pretraining for MUSK.

This script aligns image and text modalities using a CLIP-style
contrastive loss with an auxiliary masked language modeling (MLM) loss
via a lightweight cross-attention decoder. It supports either
WebDataset shards or a JSON lines file containing paired image paths
and text captions. Use ``accelerate launch`` for multi-GPU training.

Example using WebDataset shards:
    accelerate launch -m musk.contrastive_pretrain \
        --pair-data /path/to/pairs/{0000..0100}.tar \
        --epochs 20 --output musk_stage2.pt

Example using a JSON lines file:
    accelerate launch -m musk.contrastive_pretrain \
        --json-data pairs.jsonl \
        --epochs 20 --output musk_stage2.pt
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import webdataset as wds
from timm.models import create_model
from transformers import XLMRobertaTokenizer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from .json_dataset import ImageTextJsonDataset
from .utils import xlm_tokenizer
from . import modeling  # register MUSK models


def get_pair_loader(urls: str, tokenizer: XLMRobertaTokenizer, batch_size: int, num_workers: int) -> DataLoader:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384, interpolation=3, antialias=True),
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor(),
    ])

    def preprocess(img, txt):
        tokens, pad = xlm_tokenizer(txt.strip(), tokenizer)
        return (
            transform(img.convert("RGB")),
            torch.tensor(tokens),
            torch.tensor(pad, dtype=torch.bool),
        )

    dataset = (
        wds.WebDataset(urls)
        .decode("pil")
        .to_tuple("jpg;png", "txt")
        .map(lambda img, txt: preprocess(img, txt))
    )
    return DataLoader(dataset.batched(batch_size), num_workers=num_workers)


def get_json_pair_loader(json_file: str, tokenizer: XLMRobertaTokenizer, batch_size: int, num_workers: int) -> DataLoader:
    dataset = ImageTextJsonDataset(json_file, mode="pair", tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def random_mask(shape, ratio, device):
    return torch.rand(shape, device=device) < ratio


class CrossAttentionDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)

    def forward(self, text: torch.Tensor, image: torch.Tensor, text_mask: torch.Tensor | None = None):
        return self.decoder(tgt=text, memory=image, tgt_key_padding_mask=text_mask)


def clip_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits_per_image = logit_scale * image_emb @ text_emb.t()
    labels = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_image.t(), labels)
    return (loss_i + loss_t) / 2


def get_args():
    p = argparse.ArgumentParser(description="Stage-two contrastive pretraining")
    p.add_argument("--pair-data", type=str, help="WebDataset pattern with paired image-text shards")
    p.add_argument("--json-data", type=str, help="JSON lines file with paired samples")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--output", type=str, default="musk_stage2.pt")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()
    # ``find_unused_parameters`` avoids reduction errors when some parameters
    # are used only in the auxiliary MLM path
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if not args.json_data and not args.pair_data:
        raise ValueError("Provide --json-data or --pair-data")

    tokenizer_path = Path(__file__).resolve().parent / "models" / "tokenizer.spm"
    tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    if args.json_data:
        pair_loader = get_json_pair_loader(args.json_data, tokenizer, args.batch_size, args.num_workers)
    else:
        pair_loader = get_pair_loader(args.pair_data, tokenizer, args.batch_size, args.num_workers)

    model = create_model("musk_large_patch16_384")
    embed_dim = model.beit3.args.encoder_embed_dim
    decoder = CrossAttentionDecoder(embed_dim)
    mlm_head = nn.Linear(embed_dim, len(tokenizer))

    optimizer = torch.optim.AdamW(
        itertools.chain(model.parameters(), decoder.parameters(), mlm_head.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(pair_loader))

    components = accelerator.prepare(model, decoder, mlm_head, optimizer, scheduler, pair_loader)
    model, decoder, mlm_head, optimizer, scheduler, pair_loader = components
    base_model = accelerator.unwrap_model(model)

    ce_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        loss_epoch = 0.0
        mlm_epoch = 0.0
        num_batches = 0
        for images, tokens, padding in pair_loader:
            optimizer.zero_grad()
            images = images.to(accelerator.device)
            tokens = tokens.to(accelerator.device)
            padding = padding.to(accelerator.device)

            # ----- Contrastive path -----
            with accelerator.no_sync(model):
                img_emb, txt_emb = model(
                    image=images,
                    text_description=tokens,
                    padding_mask=padding,
                    return_global=True,
                )
                logit_scale = base_model.logit_scale.exp()
                loss_c = clip_loss(img_emb, txt_emb, logit_scale)
                accelerator.backward(loss_c)

            # ----- Auxiliary MLM -----
            mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
            inp_tokens = tokens.clone()
            inp_tokens[mask_txt] = mask_token_id

            _, _, img_seq, txt_seq = model(
                image=images,
                text_description=inp_tokens,
                padding_mask=padding,
                with_head=False,
                out_norm=False,
                return_global=False,
                return_seq=True,
            )
            dec_out = decoder(txt_seq, img_seq, padding.bool())
            pred = mlm_head(dec_out[mask_txt])
            loss_mlm = ce_loss(pred, tokens[mask_txt])

            accelerator.backward(loss_mlm)
            optimizer.step()
            scheduler.step()

            loss_epoch += loss_c.item()
            mlm_epoch += loss_mlm.item()
            num_batches += 1

        denom = accelerator.reduce(torch.tensor(num_batches, device=accelerator.device), reduction="sum")
        contrast_total = accelerator.reduce(torch.tensor(loss_epoch, device=accelerator.device), reduction="sum")
        mlm_total = accelerator.reduce(torch.tensor(mlm_epoch, device=accelerator.device), reduction="sum")
        accelerator.print(
            f"Epoch {epoch + 1}: Contrastive={contrast_total / denom:.4f} MLM={mlm_total / denom:.4f}"
        )

    if accelerator.is_main_process:
        accelerator.print(f"Saving model to {args.output}")
        torch.save(base_model.state_dict(), args.output)


if __name__ == "__main__":
    main()
