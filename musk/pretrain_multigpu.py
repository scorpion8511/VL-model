"""Stage-one pretraining script with multi-GPU support.

This script trains the MUSK model using masked image modeling (MIM) and
masked language modeling (MLM) losses on unpaired datasets. It mirrors the
behaviour of ``musk.pretrain`` but is provided as a self-contained example that
leverages HuggingFace Accelerate for distributed training.

Example:
    accelerate launch -m musk.pretrain_multigpu \
        --json-data /path/to/data.jsonl \
        --epochs 5 --output musk_pretrained.pt
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import XLMRobertaTokenizer
import webdataset as wds
from timm.models import create_model

from . import modeling  # register custom model
from .json_dataset import get_json_loaders, get_json_loader
from .utils import xlm_tokenizer


def random_mask(shape, ratio, device):
    return (torch.rand(shape, device=device) < ratio)


def get_image_loader(urls, batch_size, num_workers):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384, interpolation=3, antialias=True),
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor(),
    ])

    def preprocess(img):
        return transform(img.convert("RGB"))

    dataset = (
        wds.WebDataset(urls)
        .decode("pil")
        .to_tuple("jpg;png")
        .map(preprocess)
    )
    return DataLoader(dataset.batched(batch_size), num_workers=num_workers)


def get_text_loader(urls, tokenizer, batch_size, num_workers):
    def preprocess(txt):
        tokens, pad = xlm_tokenizer(txt.strip(), tokenizer)
        return torch.tensor(tokens), torch.tensor(pad, dtype=torch.bool)

    dataset = wds.WebDataset(urls).to_tuple("txt").map(lambda x: preprocess(x[0]))
    return DataLoader(dataset.batched(batch_size), num_workers=num_workers)


def parse_args():
    p = argparse.ArgumentParser(description="MUSK pretraining (multi GPU)")
    p.add_argument("--json-data", type=str, help="Path to JSON lines dataset")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mask-ratio", type=float, default=0.15)
    p.add_argument("--output", type=str, default="musk.pt")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])

    tokenizer_path = Path(__file__).resolve().parent / "models" / "tokenizer.spm"
    tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    image_loader, val_image_loader = get_json_loaders(
        args.json_data,
        mode="image",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer=None,
    )
    text_loader, val_text_loader = get_json_loaders(
        args.json_data,
        mode="text",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
    )

    model = create_model("musk_large_patch16_384")
    embed_dim = model.beit3.args.encoder_embed_dim
    patch_size = model.beit3.args.patch_size
    img_decoder = torch.nn.Linear(embed_dim, 3 * patch_size * patch_size)
    txt_decoder = torch.nn.Linear(embed_dim, len(tokenizer))

    optimizer = torch.optim.AdamW(
        itertools.chain(model.parameters(), img_decoder.parameters(), txt_decoder.parameters()),
        lr=args.lr,
    )

    components = accelerator.prepare(model, img_decoder, txt_decoder, optimizer, image_loader, text_loader)
    model, img_decoder, txt_decoder, optimizer, image_loader, text_loader = components
    val_image_loader = accelerator.prepare(val_image_loader)
    val_text_loader = accelerator.prepare(val_text_loader)

    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        mim_total = 0.0
        mlm_total = 0.0
        num_batches = 0
        for images, (tokens, padding) in zip(image_loader, text_loader):
            optimizer.zero_grad()
            B, _, H, W = images.shape
            num_patches = (H // patch_size) * (W // patch_size)
            mask_img = random_mask((B, num_patches), args.mask_ratio, images.device)
            _, _, img_seq, _ = model(
                image=images,
                vision_mask=mask_img,
                with_head=False,
                out_norm=False,
                return_global=False,
                return_seq=True,
            )
            patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
            target = patches[mask_img]
            pred = img_decoder(img_seq[mask_img])
            loss_img = mse_loss(pred, target)

            tokens = tokens.to(images.device)
            padding = padding.to(images.device)
            mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
            inp_tokens = tokens.clone()
            inp_tokens[mask_txt] = mask_token_id
            _, _, _, txt_seq = model(
                text_description=inp_tokens,
                padding_mask=padding,
                with_head=False,
                out_norm=False,
                return_global=False,
                return_seq=True,
            )
            pred_txt = txt_decoder(txt_seq[mask_txt])
            loss_txt = ce_loss(pred_txt, tokens[mask_txt])

            loss = loss_img + loss_txt
            accelerator.backward(loss)
            optimizer.step()

            mim_total += loss_img.item()
            mlm_total += loss_txt.item()
            num_batches += 1

        mim_avg = accelerator.reduce(torch.tensor(mim_total, device=accelerator.device), reduction="sum")
        mlm_avg = accelerator.reduce(torch.tensor(mlm_total, device=accelerator.device), reduction="sum")
        denom = accelerator.reduce(torch.tensor(num_batches, device=accelerator.device), reduction="sum")
        mim_avg = (mim_avg / denom).item()
        mlm_avg = (mlm_avg / denom).item()

        accelerator.print(f"Epoch {epoch + 1}: MIM={mim_avg:.4f} MLM={mlm_avg:.4f}")

        # validation
        val_mim = 0.0
        val_mlm = 0.0
        val_batches = 0
        model.eval()
        for images, (tokens, padding) in zip(val_image_loader, val_text_loader):
            with torch.no_grad():
                B, _, H, W = images.shape
                num_patches = (H // patch_size) * (W // patch_size)
                mask_img = random_mask((B, num_patches), args.mask_ratio, images.device)
                _, _, img_seq, _ = model(
                    image=images,
                    vision_mask=mask_img,
                    with_head=False,
                    out_norm=False,
                    return_global=False,
                    return_seq=True,
                )
                patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
                target = patches[mask_img]
                pred = img_decoder(img_seq[mask_img])
                loss_img = mse_loss(pred, target)

                tokens = tokens.to(images.device)
                padding = padding.to(images.device)
                mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
                inp_tokens = tokens.clone()
                inp_tokens[mask_txt] = mask_token_id
                _, _, _, txt_seq = model(
                    text_description=inp_tokens,
                    padding_mask=padding,
                    with_head=False,
                    out_norm=False,
                    return_global=False,
                    return_seq=True,
                )
                pred_txt = txt_decoder(txt_seq[mask_txt])
                loss_txt = ce_loss(pred_txt, tokens[mask_txt])

                val_mim += loss_img.item()
                val_mlm += loss_txt.item()
                val_batches += 1

        vm = accelerator.reduce(torch.tensor(val_mim, device=accelerator.device), reduction="sum")
        vl = accelerator.reduce(torch.tensor(val_mlm, device=accelerator.device), reduction="sum")
        vd = accelerator.reduce(torch.tensor(val_batches, device=accelerator.device), reduction="sum")
        vm = (vm / vd).item()
        vl = (vl / vd).item()
        accelerator.print(
            f"Epoch {epoch + 1}: Val_MIM={vm:.4f} Val_MLM={vl:.4f}")

    if accelerator.is_main_process:
        base_model = accelerator.unwrap_model(model)
        accelerator.print(f"Saving model to {args.output}")
        torch.save(base_model.state_dict(), args.output)


if __name__ == "__main__":
    main()
