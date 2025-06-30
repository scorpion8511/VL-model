"""Pretraining script for MUSK using unified masked modeling.

This example demonstrates stage-one training on **unpaired** image and text
collections. Images and texts are loaded independently and the model is
optimized with masked image modeling (MIM) and masked language modeling (MLM)
losses.

Example using WebDataset shards:
    accelerate launch -m musk.pretrain \
        --image-data /path/to/images/{0000..0100}.tar \
        --text-data /path/to/texts/{0000..0100}.tar \
        --epochs 5 --output musk_pretrained.pt

Example using a local JSON lines file:
    accelerate launch -m musk.pretrain \
        --json-data /path/to/data.jsonl \
        --epochs 5 --output musk_pretrained.pt

When a JSON file is provided, 10% of the samples are automatically used for
validation and the script reports MIM and MLM losses on the validation split.
"""

import argparse
import itertools
from pathlib import Path
import wandb

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import webdataset as wds
from .json_dataset import get_json_loader, get_json_loaders
from timm.models import create_model
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import XLMRobertaTokenizer
from .utils import xlm_tokenizer
from . import modeling  # ensure custom models are registered


class MLPAdapter(torch.nn.Module):
    """Simple bottleneck adapter."""

    def __init__(self, d_model: int, mid_dim: int = 256, scale: float = 1.0) -> None:
        super().__init__()
        self.down = torch.nn.Linear(d_model, mid_dim)
        self.relu = torch.nn.ReLU()
        self.up = torch.nn.Linear(mid_dim, d_model)
        self.scale = scale

    def forward(self, x: torch.Tensor, domains: torch.Tensor | None = None) -> torch.Tensor:
        y = self.down(x)
        y = self.relu(y)
        y = self.up(y)
        return x + self.scale * y


class MOEAdapter(torch.nn.Module):
    """Mixture-of-experts adapter with optional domain routing."""

    def __init__(
        self,
        d_model: int,
        mid_dim: int = 256,
        num_experts: int = 2,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(d_model, mid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(mid_dim, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.scale = scale

    def forward(self, x: torch.Tensor, domains: torch.Tensor | None = None) -> torch.Tensor:
        if domains is None:
            y = sum(expert(x) for expert in self.experts) / len(self.experts)
            return x + self.scale * y

        out = x.clone()
        domains = domains.view(-1)
        for idx, expert in enumerate(self.experts):
            mask = domains == idx
            if mask.any():
                if x.dim() == 2:
                    out[mask] = out[mask] + self.scale * expert(x[mask])
                else:
                    out[mask] = out[mask] + self.scale * expert(x[mask])
        return out


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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional Weights & Biases project for logging",
    )
    parser.add_argument(
        "--encoder-out",
        type=str,
        default=None,
        help="Optional path to save only the encoder weights for stage-two training",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--use-adapter",
        action="store_true",
        help="Apply bottleneck adapters to image and text features",
    )
    parser.add_argument(
        "--use-moe",
        action="store_true",
        help="Use mixture-of-experts adapters",
    )
    parser.add_argument("--adapter-dim", type=int, default=256, help="Adapter hidden dimension")
    parser.add_argument("--num-experts", type=int, default=2, help="Number of experts for MOE")
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Residual scaling applied to adapter output",
    )
    return parser.parse_args()


def main():
    args = get_args()
    # ``find_unused_parameters`` allows DDP to handle parameters that are only
    # involved in one of the two masked modeling losses
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    run = None
    if args.wandb_project and accelerator.is_main_process:
        run = wandb.init(project=args.wandb_project)

    if not args.json_data and not (args.image_data and args.text_data):
        raise ValueError("Provide --json-data or both --image-data and --text-data")

    tokenizer_path = Path(__file__).resolve().parent / "models" / "tokenizer.spm"
    tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")

    if args.json_data:
        (
            image_loader,
            val_image_loader,
        ) = get_json_loaders(
            args.json_data,
            mode="image",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tokenizer=None,
        )
        (
            text_loader,
            val_text_loader,
        ) = get_json_loaders(
            args.json_data,
            mode="text",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tokenizer=tokenizer,
        )
    else:
        image_loader = get_image_loader(args.image_data, args.batch_size, args.num_workers)
        text_loader = get_text_loader(args.text_data, tokenizer, args.batch_size, args.num_workers)
        val_image_loader = None
        val_text_loader = None

    model = create_model("musk_large_patch16_384")
    unwrapped = accelerator.unwrap_model(model)
    embed_dim = unwrapped.beit3.args.encoder_embed_dim
    patch_size = unwrapped.beit3.args.patch_size
    img_decoder = torch.nn.Linear(embed_dim, 3 * patch_size * patch_size)
    txt_decoder = torch.nn.Linear(embed_dim, len(tokenizer))

    if args.use_moe:
        img_adapter = MOEAdapter(
            embed_dim, args.adapter_dim, args.num_experts, args.adapter_scale
        )
        txt_adapter = MOEAdapter(
            embed_dim, args.adapter_dim, args.num_experts, args.adapter_scale
        )
    elif args.use_adapter:
        img_adapter = MLPAdapter(embed_dim, args.adapter_dim, args.adapter_scale)
        txt_adapter = MLPAdapter(embed_dim, args.adapter_dim, args.adapter_scale)
    else:
        img_adapter = torch.nn.Identity()
        txt_adapter = torch.nn.Identity()

    optimizer = torch.optim.AdamW(
        itertools.chain(
            model.parameters(),
            img_decoder.parameters(),
            txt_decoder.parameters(),
            img_adapter.parameters(),
            txt_adapter.parameters(),
        ),
        lr=args.lr,
    )

    components = accelerator.prepare(
        model,
        img_decoder,
        txt_decoder,
        img_adapter,
        txt_adapter,
        optimizer,
        image_loader,
        text_loader,
    )
    (
        model,
        img_decoder,
        txt_decoder,
        img_adapter,
        txt_adapter,
        optimizer,
        image_loader,
        text_loader,
    ) = components
    if val_image_loader is not None:
        val_image_loader = accelerator.prepare(val_image_loader)
        val_text_loader = accelerator.prepare(val_text_loader)

    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    model.train()
    if hasattr(img_adapter, "train"):
        img_adapter.train()
    if hasattr(txt_adapter, "train"):
        txt_adapter.train()
    for epoch in range(args.epochs):
        mim_loss_epoch = 0.0
        mlm_loss_epoch = 0.0
        num_batches = 0
        for images, (tokens, padding) in zip(image_loader, text_loader):
            optimizer.zero_grad()

            # ----- Masked Image Modeling -----
            B, _, H, W = images.shape
            num_patches = (H // patch_size) * (W // patch_size)
            mask_img = random_mask((B, num_patches), args.mask_ratio, images.device)
            with accelerator.no_sync(model):
                _, _, img_seq, _ = model(
                    image=images,
                    vision_mask=mask_img,
                    with_head=False,
                    out_norm=False,
                    return_global=False,
                    return_seq=True,
                )
                img_seq = img_adapter(img_seq)
                patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
                target = patches[mask_img]
                pred = img_decoder(img_seq[mask_img])
                loss_img = mse_loss(pred, target)
                accelerator.backward(loss_img)

            # ----- Masked Language Modeling -----
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
            txt_seq = txt_adapter(txt_seq)
            pred = txt_decoder(txt_seq[mask_txt])
            loss_txt = ce_loss(pred, tokens[mask_txt])

            accelerator.backward(loss_txt)
            optimizer.step()

            mim_loss_epoch += loss_img.item()
            mlm_loss_epoch += loss_txt.item()
            num_batches += 1

        denom = accelerator.reduce(torch.tensor(num_batches, device=accelerator.device), reduction="sum")
        mim_total = accelerator.reduce(torch.tensor(mim_loss_epoch, device=accelerator.device), reduction="sum")
        mlm_total = accelerator.reduce(torch.tensor(mlm_loss_epoch, device=accelerator.device), reduction="sum")

        mim_avg = (mim_total / denom).item()
        mlm_avg = (mlm_total / denom).item()

        if val_image_loader is not None:
            model.eval()
            if hasattr(img_adapter, "eval"):
                img_adapter.eval()
            if hasattr(txt_adapter, "eval"):
                txt_adapter.eval()

            val_mim = 0.0
            val_mlm = 0.0
            val_batches = 0
            for images, (tokens, padding) in zip(val_image_loader, val_text_loader):
                B, _, H, W = images.shape
                num_patches = (H // patch_size) * (W // patch_size)
                mask_img = random_mask((B, num_patches), args.mask_ratio, images.device)
                with torch.no_grad():
                    _, _, img_seq, _ = model(
                        image=images,
                        vision_mask=mask_img,
                        with_head=False,
                        out_norm=False,
                        return_global=False,
                        return_seq=True,
                    )
                    img_seq = img_adapter(img_seq)
                patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
                target = patches[mask_img]
                pred = img_decoder(img_seq[mask_img])
                loss_img = mse_loss(pred, target)

                tokens = tokens.to(images.device)
                padding = padding.to(images.device)
                mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
                inp_tokens = tokens.clone()
                inp_tokens[mask_txt] = mask_token_id
                with torch.no_grad():
                    _, _, _, txt_seq = model(
                        text_description=inp_tokens,
                        padding_mask=padding,
                        with_head=False,
                        out_norm=False,
                        return_global=False,
                        return_seq=True,
                    )
                    txt_seq = txt_adapter(txt_seq)
                pred_txt = txt_decoder(txt_seq[mask_txt])
                loss_txt = ce_loss(pred_txt, tokens[mask_txt])

                val_mim += loss_img.item()
                val_mlm += loss_txt.item()
                val_batches += 1

            val_denom = accelerator.reduce(torch.tensor(val_batches, device=accelerator.device), reduction="sum")
            val_mim_total = accelerator.reduce(torch.tensor(val_mim, device=accelerator.device), reduction="sum")
            val_mlm_total = accelerator.reduce(torch.tensor(val_mlm, device=accelerator.device), reduction="sum")
            mim_val_avg = (val_mim_total / val_denom).item()
            mlm_val_avg = (val_mlm_total / val_denom).item()
            accelerator.print(
                f"Epoch {epoch + 1}: MIM={mim_avg:.4f} MLM={mlm_avg:.4f} Val_MIM={mim_val_avg:.4f} Val_MLM={mlm_val_avg:.4f}"
            )
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_mim": mim_avg,
                        "train_mlm": mlm_avg,
                        "val_mim": mim_val_avg,
                        "val_mlm": mlm_val_avg,
                    }
                )
            model.train()
            if hasattr(img_adapter, "train"):
                img_adapter.train()
            if hasattr(txt_adapter, "train"):
                txt_adapter.train()
        else:
            accelerator.print(f"Epoch {epoch + 1}: MIM={mim_avg:.4f} MLM={mlm_avg:.4f}")
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_mim": mim_avg,
                        "train_mlm": mlm_avg,
                    }
                )

    if accelerator.is_main_process:
        base_model = accelerator.unwrap_model(model)
        accelerator.print(f"Saving model to {args.output}")
        torch.save(base_model.state_dict(), args.output)
        if args.encoder_out:
            accelerator.print(f"Saving encoder weights to {args.encoder_out}")
            torch.save(base_model.beit3.state_dict(), args.encoder_out)
        if run:
            run.finish()


if __name__ == "__main__":
    main()
