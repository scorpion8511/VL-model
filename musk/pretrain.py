"""Pretraining script for MUSK using unified masked modeling.

This example demonstrates stage-one training on **unpaired** image and text
collections. Images and texts are loaded independently and the model is
optimized with masked image modeling (MIM) and masked language modeling (MLM)
losses.

Example using WebDataset shards:
    python -m musk.pretrain \
        --image-data /path/to/images/{0000..0100}.tar \
        --text-data /path/to/texts/{0000..0100}.tar \
        --epochs 5 --output musk_pretrained.pt

Example using a local JSON lines file:
    python -m musk.pretrain \
        --json-data /path/to/data.jsonl \
        --epochs 5 --output musk_pretrained.pt
"""

import argparse
import itertools
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import webdataset as wds
from .json_dataset import get_json_loader
from timm.models import create_model
from accelerate import Accelerator
from transformers import XLMRobertaTokenizer
from .utils import xlm_tokenizer
from . import modeling  # ensure custom models are registered


def get_image_loader(urls, batch_size, num_workers):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384, interpolation=3, antialias=True),
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor(),
    ])

    def preprocess(img):
        return transform(img.convert("RGB"))

    dataset = wds.WebDataset(urls).decode("pil").to_tuple("jpg;png").map(preprocess)
    return DataLoader(dataset.batched(batch_size), num_workers=num_workers)


def get_text_loader(urls, tokenizer, batch_size, num_workers):
    def preprocess(txt):
        tokens, pad = xlm_tokenizer(txt.strip(), tokenizer)
        return torch.tensor(tokens), torch.tensor(pad, dtype=torch.bool)

    dataset = wds.WebDataset(urls).to_tuple("txt").map(lambda x: preprocess(x[0]))
    return DataLoader(dataset.batched(batch_size), num_workers=num_workers)


def random_mask(shape, ratio, device):
    return (torch.rand(shape, device=device) < ratio)


def get_args():
    parser = argparse.ArgumentParser(description="MUSK masked-modeling pretraining")
    parser.add_argument("--image-data", type=str, help="WebDataset pattern for image shards")
    parser.add_argument("--text-data", type=str, help="WebDataset pattern for text shards")
    parser.add_argument("--json-data", type=str, help="JSON lines file with 'image' and 'text' fields")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="musk.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = get_args()
    accelerator = Accelerator()

    if not args.json_data and not (args.image_data and args.text_data):
        raise ValueError("Provide --json-data or both --image-data and --text-data")

    tokenizer = XLMRobertaTokenizer(str((__file__).replace("pretrain.py", "models/tokenizer.spm")))
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    if args.json_data:
        image_loader = get_json_loader(
            args.json_data,
            mode="image",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tokenizer=None,
        )
        text_loader = get_json_loader(
            args.json_data,
            mode="text",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tokenizer=tokenizer,
        )
    else:
        image_loader = get_image_loader(args.image_data, args.batch_size, args.num_workers)
        text_loader = get_text_loader(args.text_data, tokenizer, args.batch_size, args.num_workers)

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

    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        for images, (tokens, padding) in zip(image_loader, text_loader):
            optimizer.zero_grad()

            # ----- Masked Image Modeling -----
            B, _, H, W = images.shape
            num_patches = (H // patch_size) * (W // patch_size)
            mask_img = random_mask((B, num_patches), args.mask_ratio, images.device)
            out = model.beit3(visual_tokens=images, vision_masked_position=mask_img)
            img_seq = out["encoder_out"][:, 1:]
            patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
            target = patches[mask_img]
            pred = img_decoder(img_seq[mask_img])
            loss_img = mse_loss(pred, target)

            # ----- Masked Language Modeling -----
            tokens = tokens.to(images.device)
            padding = padding.to(images.device)
            mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
            inp_tokens = tokens.clone()
            inp_tokens[mask_txt] = mask_token_id
            out = model.beit3(textual_tokens=inp_tokens, text_padding_position=padding)
            txt_seq = out["encoder_out"]
            pred = txt_decoder(txt_seq[mask_txt])
            loss_txt = ce_loss(pred, tokens[mask_txt])

            loss = (loss_img + loss_txt) / 2
            accelerator.backward(loss)
            optimizer.step()

        accelerator.print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")

    if accelerator.is_main_process:
        accelerator.print(f"Saving model to {args.output}")
        torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
