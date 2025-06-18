"""Stage-two MMA pretraining for MUSK.

This script trains MUSK using contrastive loss with an auxiliary
cross-attention decoder for masked language modeling (MMA).
It mirrors ``musk.contrastive_pretrain`` but is provided as an
independent example. Use ``accelerate launch`` to run on multiple GPUs.
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


def get_json_loaders(json_file: str, tokenizer: XLMRobertaTokenizer, batch_size: int, num_workers: int, val_split: float = 0.1) -> tuple[DataLoader, DataLoader]:
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
    logits = logit_scale * image_emb @ text_emb.t()
    targets = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2


def parse_args():
    p = argparse.ArgumentParser(description="Stage-two MMA pretraining")
    p.add_argument("--pair-data", type=str, help="WebDataset pattern of image-text pairs")
    p.add_argument("--json-data", type=str, help="JSON lines file with paired samples")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--output", type=str, default="musk_stage2_mma.pt")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])

    if not args.json_data and not args.pair_data:
        raise ValueError("Provide --json-data or --pair-data")

    tokenizer_path = Path(__file__).resolve().parent / "models" / "tokenizer.spm"
    tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    if args.json_data:
        train_loader, val_loader = get_json_loaders(
            args.json_data, tokenizer, args.batch_size, args.num_workers
        )
    else:
        train_loader = get_pair_loader(args.pair_data, tokenizer, args.batch_size, args.num_workers)
        val_loader = None

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    components = accelerator.prepare(model, decoder, mlm_head, optimizer, scheduler, train_loader)
    model, decoder, mlm_head, optimizer, scheduler, train_loader = components
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    base_model = accelerator.unwrap_model(model)

    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        lc, lm = 0.0, 0.0
        n = 0
        for images, tokens, padding in train_loader:
            optimizer.zero_grad()
            images = images.to(accelerator.device)
            tokens = tokens.to(accelerator.device)
            padding = padding.to(accelerator.device)

            with accelerator.no_sync(model):
                img_emb, txt_emb = model(
                    image=images,
                    text_description=tokens,
                    padding_mask=padding,
                    return_global=True,
                )
                loss_c = clip_loss(img_emb, txt_emb, base_model.logit_scale.exp())
                accelerator.backward(loss_c)

            mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
            inp = tokens.clone()
            inp[mask_txt] = mask_token_id

            _, _, img_seq, txt_seq = model(
                image=images,
                text_description=inp,
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

            lc += loss_c.item()
            lm += loss_mlm.item()
            n += 1

        lc_total = accelerator.reduce(torch.tensor(lc, device=accelerator.device), reduction="sum")
        lm_total = accelerator.reduce(torch.tensor(lm, device=accelerator.device), reduction="sum")
        n_total = accelerator.reduce(torch.tensor(n, device=accelerator.device), reduction="sum")
        accelerator.print(f"Epoch {epoch + 1}: Contrastive={(lc_total/n_total).item():.4f} MLM={(lm_total/n_total).item():.4f}")

        if val_loader is not None:
            vc, vm = 0.0, 0.0
            vn = 0
            model.eval()
            for images, tokens, padding in val_loader:
                images = images.to(accelerator.device)
                tokens = tokens.to(accelerator.device)
                padding = padding.to(accelerator.device)
                with torch.no_grad():
                    img_emb, txt_emb = model(
                        image=images,
                        text_description=tokens,
                        padding_mask=padding,
                        return_global=True,
                    )
                loss_c = clip_loss(img_emb, txt_emb, base_model.logit_scale.exp())

                mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
                inp = tokens.clone()
                inp[mask_txt] = mask_token_id
                with torch.no_grad():
                    _, _, img_seq, txt_seq = model(
                        image=images,
                        text_description=inp,
                        padding_mask=padding,
                        with_head=False,
                        out_norm=False,
                        return_global=False,
                        return_seq=True,
                    )
                    dec_out = decoder(txt_seq, img_seq, padding.bool())
                    pred = mlm_head(dec_out[mask_txt])
                    loss_mlm = ce_loss(pred, tokens[mask_txt])

                vc += loss_c.item()
                vm += loss_mlm.item()
                vn += 1

            vc_t = accelerator.reduce(torch.tensor(vc, device=accelerator.device), reduction="sum")
            vm_t = accelerator.reduce(torch.tensor(vm, device=accelerator.device), reduction="sum")
            vn_t = accelerator.reduce(torch.tensor(vn, device=accelerator.device), reduction="sum")
            accelerator.print(
                f"Epoch {epoch + 1}: Val_Contrastive={(vc_t/vn_t).item():.4f} Val_MLM={(vm_t/vn_t).item():.4f}"
            )

    if accelerator.is_main_process:
        accelerator.print(f"Saving model to {args.output}")
        torch.save(base_model.state_dict(), args.output)


if __name__ == "__main__":
    main()
