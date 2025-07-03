"""Stage-two contrastive pretraining for MUSK.

This script aligns image and text modalities using a CLIP-style
contrastive loss with an auxiliary masked language modeling (MLM) loss
via a lightweight cross-attention decoder. It supports either
WebDataset shards or a JSON lines file containing paired image paths
and text captions. Use ``accelerate launch`` for multi-GPU training.

Example using WebDataset shards:
    accelerate launch --mixed_precision fp16 -m musk.contrastive_pretrain \
        --pair-data /path/to/pairs/{0000..0100}.tar \
        --batch-size 16 --epochs 20 --output musk_stage2.pt

Example using a JSON lines file:
    accelerate launch --mixed_precision fp16 -m musk.contrastive_pretrain \
        --json-data pairs.jsonl \
        --batch-size 16 --epochs 20 --output musk_stage2.pt

When ``--json-data`` is supplied, the loader reserves 10% of the pairs for
validation and both contrastive and MLM losses are reported on the validation
split each epoch.
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
import wandb

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


def get_json_pair_loaders(
    json_file: str,
    tokenizer: XLMRobertaTokenizer,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    dataset = ImageTextJsonDataset(json_file, mode="pair", tokenizer=tokenizer)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader


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
    # large batches can easily exceed GPU memory when training the
    # 384x384 MUSK model; use a conservative default
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--output", type=str, default="musk_stage2.pt")
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional Weights & Biases project for logging",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Path to encoder weights pretrained in stage one",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=0, help="Number of MoE experts")
    p.add_argument("--moe-freq", type=int, default=0, help="Insert MoE layer every N blocks")
    return p.parse_args()


def main():
    args = get_args()
    # ``find_unused_parameters`` avoids reduction errors when some parameters
    # are used only in the auxiliary MLM path
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    run = None
    if args.wandb_project and accelerator.is_main_process:
        run = wandb.init(project=args.wandb_project)

    if not args.json_data and not args.pair_data:
        raise ValueError("Provide --json-data or --pair-data")

    tokenizer_path = Path(__file__).resolve().parent / "models" / "tokenizer.spm"
    tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    if args.json_data:
        (
            pair_loader,
            val_loader,
        ) = get_json_pair_loaders(
            args.json_data,
            tokenizer,
            args.batch_size,
            args.num_workers,
        )
    else:
        pair_loader = get_pair_loader(args.pair_data, tokenizer, args.batch_size, args.num_workers)
        val_loader = None

    model = create_model(
        "musk_large_patch16_384",
        moe_expert_count=args.num_experts,
        moe_freq=args.moe_freq,
    )
    if args.encoder:
        state = torch.load(args.encoder, map_location="cpu")
        missing = model.beit3.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded encoder weights from {args.encoder}")
        if missing.missing_keys:
            accelerator.print(f"Missing keys in encoder load: {missing.missing_keys}")
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
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    base_model = accelerator.unwrap_model(model)

    ce_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        loss_epoch = 0.0
        mlm_epoch = 0.0
        num_batches = 0
        for batch in pair_loader:
            if len(batch) == 4:
                images, tokens, padding, domain = batch
            else:
                images, tokens, padding = batch
                domain = None
            optimizer.zero_grad()
            images = images.to(accelerator.device)
            tokens = tokens.to(accelerator.device)
            padding = padding.to(accelerator.device)

            # ----- Contrastive path -----
            with accelerator.no_sync(model):
                img_emb, txt_emb, l_aux_c = model(
                    image=images,
                    text_description=tokens,
                    padding_mask=padding,
                    return_global=True,
                    domain=domain,
                    return_l_aux=True,
                )
                logit_scale = base_model.logit_scale.exp()
                loss_c = clip_loss(img_emb, txt_emb, logit_scale)
                if l_aux_c is not None:
                    loss_c = loss_c + l_aux_c
                accelerator.backward(loss_c)

            # ----- Auxiliary MLM -----
            mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
            inp_tokens = tokens.clone()
            inp_tokens[mask_txt] = mask_token_id

            _, _, img_seq, txt_seq, l_aux_mlm = model(
                image=images,
                text_description=inp_tokens,
                padding_mask=padding,
                with_head=False,
                out_norm=False,
                return_global=False,
                return_seq=True,
                domain=domain,
                return_l_aux=True,
            )
            dec_out = decoder(txt_seq, img_seq, padding.bool())
            pred = mlm_head(dec_out[mask_txt])
            loss_mlm = ce_loss(pred, tokens[mask_txt])
            if l_aux_mlm is not None:
                loss_mlm = loss_mlm + l_aux_mlm

            accelerator.backward(loss_mlm)
            optimizer.step()
            scheduler.step()

            loss_epoch += loss_c.item()
            mlm_epoch += loss_mlm.item()
            num_batches += 1

        denom = accelerator.reduce(torch.tensor(num_batches, device=accelerator.device), reduction="sum")
        contrast_total = accelerator.reduce(torch.tensor(loss_epoch, device=accelerator.device), reduction="sum")
        mlm_total = accelerator.reduce(torch.tensor(mlm_epoch, device=accelerator.device), reduction="sum")

        c_avg = (contrast_total / denom).item()
        mlm_avg = (mlm_total / denom).item()

        if val_loader is not None:
            val_c = 0.0
            val_mlm = 0.0
            val_batches = 0
            for batch in val_loader:
                if len(batch) == 4:
                    images, tokens, padding, domain = batch
                else:
                    images, tokens, padding = batch
                    domain = None
                images = images.to(accelerator.device)
                tokens = tokens.to(accelerator.device)
                padding = padding.to(accelerator.device)

                with torch.no_grad():
                    img_emb, txt_emb, l_aux_c = model(
                        image=images,
                        text_description=tokens,
                        padding_mask=padding,
                        return_global=True,
                        domain=domain,
                        return_l_aux=True,
                    )
                logit_scale = base_model.logit_scale.exp()
                loss_c = clip_loss(img_emb, txt_emb, logit_scale)
                if l_aux_c is not None:
                    loss_c = loss_c + l_aux_c

                mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
                inp_tokens = tokens.clone()
                inp_tokens[mask_txt] = mask_token_id
                with torch.no_grad():
                    _, _, img_seq, txt_seq, l_aux_txt = model(
                        image=images,
                        text_description=inp_tokens,
                        padding_mask=padding,
                        with_head=False,
                        out_norm=False,
                        return_global=False,
                        return_seq=True,
                        domain=domain,
                        return_l_aux=True,
                    )
                    dec_out = decoder(txt_seq, img_seq, padding.bool())
                    pred = mlm_head(dec_out[mask_txt])
                    loss_mlm = ce_loss(pred, tokens[mask_txt])
                    if l_aux_txt is not None:
                        loss_mlm = loss_mlm + l_aux_txt

                val_c += loss_c.item()
                val_mlm += loss_mlm.item()
                val_batches += 1

            val_denom = accelerator.reduce(torch.tensor(val_batches, device=accelerator.device), reduction="sum")
            val_c_total = accelerator.reduce(torch.tensor(val_c, device=accelerator.device), reduction="sum")
            val_mlm_total = accelerator.reduce(torch.tensor(val_mlm, device=accelerator.device), reduction="sum")
            c_val_avg = (val_c_total / val_denom).item()
            mlm_val_avg = (val_mlm_total / val_denom).item()
            accelerator.print(
                f"Epoch {epoch + 1}: Contrastive={c_avg:.4f} MLM={mlm_avg:.4f} Val_Contrastive={c_val_avg:.4f} Val_MLM={mlm_val_avg:.4f}"
            )
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_contrastive": c_avg,
                        "train_mlm": mlm_avg,
                        "val_contrastive": c_val_avg,
                        "val_mlm": mlm_val_avg,
                    }
                )
        else:
            accelerator.print(
                f"Epoch {epoch + 1}: Contrastive={c_avg:.4f} MLM={mlm_avg:.4f}"
            )
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_contrastive": c_avg,
                        "train_mlm": mlm_avg,
                    }
                )

    if accelerator.is_main_process:
        accelerator.print(f"Saving model to {args.output}")
        torch.save(base_model.state_dict(), args.output)
        if run:
            run.finish()


if __name__ == "__main__":
    main()
