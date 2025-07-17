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
from .domain_encoders import get_domain_encoder, load_domain_encoders, parse_domain_list
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
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Optional path to pretrained encoder weights to initialize stage one",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Name of domain-specific encoder to initialize the model",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Comma-separated list of additional domain encoders for distillation",
    )
    parser.add_argument(
        "--domain-loss",
        action="store_true",
        help="Enable domain classification loss using dataset domain labels",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=1.0,
        help="Weight for distillation loss from domain encoders",
    )
    parser.add_argument(
        "--moe-freq",
        type=int,
        default=0,
        help="Insert a mixture-of-experts layer every N transformer blocks",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=0,
        help="Number of experts to use in MoE layers",
    )
    parser.add_argument("--num-workers", type=int, default=4)
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

    domain_manager = None
    domain_list = []
    if args.domains:
        if not args.json_data:
            raise ValueError("--domains requires --json-data dataset with domain field")
        domain_list = parse_domain_list(args.domains)

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
            return_domain=bool(domain_list) or args.domain_loss,
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
            return_domain=False,
        )
    else:
        image_loader = get_image_loader(args.image_data, args.batch_size, args.num_workers)
        text_loader = get_text_loader(args.text_data, tokenizer, args.batch_size, args.num_workers)
        val_image_loader = None
        val_text_loader = None

    model = create_model(
        "musk_large_patch16_384",
        moe_freq=args.moe_freq,
        moe_expert_count=args.num_experts,
    )
    if args.domain:
        try:
            _, domain_model = get_domain_encoder(args.domain)
            state = domain_model.state_dict()
            missing = model.beit3.load_state_dict(state, strict=False)
            accelerator.print(f"Initialized encoder from domain '{args.domain}'")
            if missing.missing_keys:
                accelerator.print(
                    f"Missing keys when loading domain encoder: {missing.missing_keys}"
                )
        except Exception as e:
            accelerator.print(f"Failed to load domain encoder '{args.domain}': {e}")
    if args.encoder:
        state = torch.load(args.encoder, map_location="cpu")
        missing = model.beit3.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded encoder weights from {args.encoder}")
        if missing.missing_keys:
            accelerator.print(f"Missing keys in encoder load: {missing.missing_keys}")
    unwrapped = accelerator.unwrap_model(model)
    embed_dim = unwrapped.beit3.args.encoder_embed_dim
    patch_size = unwrapped.beit3.args.patch_size
    if domain_list:
        domain_manager = load_domain_encoders(domain_list, out_dim=embed_dim)
        domain_manager = domain_manager.to(accelerator.device)
        domain_manager.eval()
        accelerator.print(f"Loaded domain encoders: {', '.join(domain_list)}")
    domain_head = None
    domain_names = []
    if args.domain_loss:
        ds = image_loader.dataset
        if isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        domain_names = getattr(ds, "domains", [])
        if not domain_names:
            raise ValueError("--domain-loss requires domain labels in dataset")
        domain_head = torch.nn.Linear(embed_dim, len(domain_names))
    img_decoder = torch.nn.Linear(embed_dim, 3 * patch_size * patch_size)
    txt_decoder = torch.nn.Linear(embed_dim, len(tokenizer))

    params = [model.parameters(), img_decoder.parameters(), txt_decoder.parameters()]
    if domain_head is not None:
        params.append(domain_head.parameters())
    optimizer = torch.optim.AdamW(itertools.chain(*params), lr=args.lr)

    modules = [model, img_decoder, txt_decoder]
    if domain_head is not None:
        modules.append(domain_head)
    modules += [optimizer, image_loader, text_loader]
    prepared = accelerator.prepare(*modules)
    ptr = 0
    model = prepared[ptr]; ptr += 1
    img_decoder = prepared[ptr]; ptr += 1
    txt_decoder = prepared[ptr]; ptr += 1
    if domain_head is not None:
        domain_head = prepared[ptr]; ptr += 1
    optimizer = prepared[ptr]; ptr += 1
    image_loader = prepared[ptr]; ptr += 1
    text_loader = prepared[ptr]
    if val_image_loader is not None:
        val_image_loader = accelerator.prepare(val_image_loader)
        val_text_loader = accelerator.prepare(val_text_loader)

    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        mim_loss_epoch = 0.0
        mlm_loss_epoch = 0.0
        dom_loss_epoch = 0.0
        num_batches = 0
        for img_data, (tokens, padding) in zip(image_loader, text_loader):
            if domain_manager is not None:
                images, domains = img_data
            else:
                images = img_data
            optimizer.zero_grad()

            # ----- Masked Image Modeling -----
            B, _, H, W = images.shape
            num_patches = (H // patch_size) * (W // patch_size)
            mask_img = random_mask((B, num_patches), args.mask_ratio, images.device)
            with accelerator.no_sync(model):
                img_cls, _, img_seq, _ = model(
                    image=images,
                    vision_mask=mask_img,
                    with_head=False,
                    out_norm=False,
                    return_global=domain_manager is not None,
                    return_seq=True,
                )
                patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
                target = patches[mask_img]
                pred = img_decoder(img_seq[mask_img])
                loss_img = mse_loss(pred, target)
                if domain_manager is not None:
                    with torch.no_grad():
                        teach = domain_manager(images, list(domains)).to(images.device)
                    loss_img = loss_img + args.distill_weight * mse_loss(img_cls, teach)
                accelerator.backward(loss_img)

                if args.domain_loss and domain_head is not None:
                    dom_idx = torch.tensor([domain_names.index(d) if d in domain_names else -1 for d in domains], device=images.device)
                    valid = dom_idx >= 0
                    if valid.any():
                        dom_pred = domain_head(img_cls[valid])
                        loss_dom = ce_loss(dom_pred, dom_idx[valid])
                        accelerator.backward(loss_dom)
                        dom_loss_epoch += loss_dom.item()

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
        if args.domain_loss:
            dom_total = accelerator.reduce(torch.tensor(dom_loss_epoch, device=accelerator.device), reduction="sum")

        mim_avg = (mim_total / denom).item()
        mlm_avg = (mlm_total / denom).item()
        if args.domain_loss:
            dom_avg = (dom_total / denom).item()

        if val_image_loader is not None:
            val_mim = 0.0
            val_mlm = 0.0
            val_dom = 0.0
            val_batches = 0
            for img_data, (tokens, padding) in zip(val_image_loader, val_text_loader):
                if domain_manager is not None:
                    images, domains = img_data
                else:
                    images = img_data
                B, _, H, W = images.shape
                num_patches = (H // patch_size) * (W // patch_size)
                mask_img = random_mask((B, num_patches), args.mask_ratio, images.device)
                with torch.no_grad():
                    img_cls, _, img_seq, _ = model(
                        image=images,
                        vision_mask=mask_img,
                        with_head=False,
                        out_norm=False,
                        return_global=domain_manager is not None,
                        return_seq=True,
                    )
                patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
                target = patches[mask_img]
                pred = img_decoder(img_seq[mask_img])
                loss_img = mse_loss(pred, target)
                if domain_manager is not None:
                    with torch.no_grad():
                        teach = domain_manager(images, list(domains)).to(images.device)
                    loss_img = loss_img + args.distill_weight * mse_loss(img_cls, teach)

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
                pred_txt = txt_decoder(txt_seq[mask_txt])
                loss_txt = ce_loss(pred_txt, tokens[mask_txt])

                if args.domain_loss and domain_head is not None:
                    dom_idx = torch.tensor([domain_names.index(d) if d in domain_names else -1 for d in domains], device=images.device)
                    valid = dom_idx >= 0
                    if valid.any():
                        with torch.no_grad():
                            pred_dom = domain_head(img_cls[valid])
                            loss_dom = ce_loss(pred_dom, dom_idx[valid])
                        val_dom += loss_dom.item()

                val_mim += loss_img.item()
                val_mlm += loss_txt.item()
                val_batches += 1

            val_denom = accelerator.reduce(torch.tensor(val_batches, device=accelerator.device), reduction="sum")
            val_mim_total = accelerator.reduce(torch.tensor(val_mim, device=accelerator.device), reduction="sum")
            val_mlm_total = accelerator.reduce(torch.tensor(val_mlm, device=accelerator.device), reduction="sum")
            mim_val_avg = (val_mim_total / val_denom).item()
            mlm_val_avg = (val_mlm_total / val_denom).item()
            if args.domain_loss:
                val_dom_total = accelerator.reduce(torch.tensor(val_dom, device=accelerator.device), reduction="sum")
                dom_val_avg = (val_dom_total / val_denom).item()
            msg = f"Epoch {epoch + 1}: MIM={mim_avg:.4f} MLM={mlm_avg:.4f} Val_MIM={mim_val_avg:.4f} Val_MLM={mlm_val_avg:.4f}"
            if args.domain_loss:
                msg += f" Val_DOM={dom_val_avg:.4f}"
            accelerator.print(msg)
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_mim": mim_avg,
                        "train_mlm": mlm_avg,
                        **({"train_dom": dom_avg} if args.domain_loss else {}),
                        "val_mim": mim_val_avg,
                        "val_mlm": mlm_val_avg,
                        **({"val_dom": dom_val_avg} if args.domain_loss else {})
                    }
                )
        else:
            msg = f"Epoch {epoch + 1}: MIM={mim_avg:.4f} MLM={mlm_avg:.4f}"
            if args.domain_loss:
                avg_dom = dom_loss_epoch / denom.item()
                msg += f" DOM={avg_dom:.4f}"
            accelerator.print(msg)
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_mim": mim_avg,
                        "train_mlm": mlm_avg,
                        **({"train_dom": dom_loss_epoch / denom.item()} if args.domain_loss else {}),
                    }
                )

    if accelerator.is_main_process:
        base_model = accelerator.unwrap_model(model)
        accelerator.print(f"Saving model to {args.output}")
        torch.save(base_model.state_dict(), args.output)
        if args.encoder_out:
            accelerator.print(f"Saving encoder weights to {args.encoder_out}")
            torch.save(base_model.beit3.state_dict(), args.encoder_out)
        if args.domain_loss and domain_head is not None:
            torch.save(domain_head.state_dict(), args.output + ".domain_head.pth")
        if run:
            run.finish()


if __name__ == "__main__":
    main()
