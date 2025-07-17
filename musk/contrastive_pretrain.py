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

from .domain_encoders import get_domain_encoder, load_domain_encoders, parse_domain_list

from .json_dataset import ImageTextJsonDataset
from .utils import xlm_tokenizer
from . import modeling  # register MUSK models
from .decoders import CrossAttentionDecoder, CaptionDecoder, PatchDecoder


def get_pair_loader(
    urls: str,
    tokenizer: XLMRobertaTokenizer,
    batch_size: int,
    num_workers: int,
    return_patches: bool = False,
    patch_size: int = 16,
    return_domain: bool = False,
) -> DataLoader:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384, interpolation=3, antialias=True),
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor(),
    ])

    def preprocess(img, txt):
        tokens, pad = xlm_tokenizer(txt.strip(), tokenizer)
        img_t = transform(img.convert("RGB"))
        out = (
            img_t,
            torch.tensor(tokens),
            torch.tensor(pad, dtype=torch.bool),
        )
        if return_patches:
            patches = F.unfold(img_t.unsqueeze(0), kernel_size=patch_size, stride=patch_size).squeeze(0).T
            out = out + (patches,)
        if return_domain:
            out = out + (None,)
        return out

    dataset = (
        wds.WebDataset(urls)
        .decode("pil")
        .to_tuple("jpg;png", "txt")
        .map(lambda img, txt: preprocess(img, txt))
    )
    return DataLoader(dataset.batched(batch_size), num_workers=num_workers)


def get_json_pair_loader(
    json_file: str,
    tokenizer: XLMRobertaTokenizer,
    batch_size: int,
    num_workers: int,
    return_patches: bool = False,
    patch_size: int = 16,
    return_domain: bool = False,
) -> DataLoader:
    dataset = ImageTextJsonDataset(
        json_file,
        mode="pair",
        tokenizer=tokenizer,
        return_patches=return_patches,
        patch_size=patch_size,
        return_domain=return_domain,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_json_pair_loaders(
    json_file: str,
    tokenizer: XLMRobertaTokenizer,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    return_patches: bool = False,
    patch_size: int = 16,
    return_domain: bool = False,
) -> tuple[DataLoader, DataLoader]:
    dataset = ImageTextJsonDataset(
        json_file,
        mode="pair",
        tokenizer=tokenizer,
        return_patches=return_patches,
        patch_size=patch_size,
        return_domain=return_domain,
    )
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader


def random_mask(shape, ratio, device):
    return torch.rand(shape, device=device) < ratio



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
    p.add_argument("--caption-loss", action="store_true", help="Enable caption generation loss")
    p.add_argument("--recon-loss", action="store_true", help="Enable image reconstruction loss")
    p.add_argument("--domain-loss", action="store_true", help="Enable domain classification loss")
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
    p.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=None,
        help="Comma-separated list of domain encoders to use",
    )
    p.add_argument(
        "--moe-freq",
        type=int,
        default=0,
        help="Insert a mixture-of-experts layer every N transformer blocks",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        default=0,
        help="Number of experts to use in MoE layers",
    )
    p.add_argument("--num-workers", type=int, default=4)
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

    model = create_model(
        "musk_large_patch16_384",
        moe_freq=args.moe_freq,
        moe_expert_count=args.num_experts,
    )
    embed_dim = model.beit3.args.encoder_embed_dim

    domain_manager = None
    if args.domains:
        domain_list = parse_domain_list(args.domains)
        if domain_list:
            domain_manager = load_domain_encoders(domain_list, out_dim=embed_dim)
            domain_manager = domain_manager.to(accelerator.device)
            domain_manager.eval()
            accelerator.print(f"Loaded domain encoders: {', '.join(domain_list)}")
    patch_size = model.beit3.args.patch_size
    img_size = model.beit3.args.img_size

    if args.json_data:
        (
            pair_loader,
            val_loader,
        ) = get_json_pair_loaders(
            args.json_data,
            tokenizer,
            args.batch_size,
            args.num_workers,
            return_patches=args.recon_loss,
            patch_size=patch_size,
            return_domain=domain_manager is not None or args.domain_loss,
        )
    else:
        pair_loader = get_pair_loader(
            args.pair_data,
            tokenizer,
            args.batch_size,
            args.num_workers,
            return_patches=args.recon_loss,
            patch_size=patch_size,
            return_domain=domain_manager is not None or args.domain_loss,
        )
        val_loader = None

    base_ds = pair_loader.dataset
    if isinstance(base_ds, torch.utils.data.Subset):
        base_ds = base_ds.dataset
    domain_names = getattr(base_ds, "domains", [])
    domain_to_idx = getattr(base_ds, "domain_to_idx", {})
    if args.domain_loss and not domain_names:
        raise ValueError("--domain-loss requires domain labels in the dataset")
    if args.encoder:
        state = torch.load(args.encoder, map_location="cpu")
        missing = model.beit3.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded encoder weights from {args.encoder}")
        if missing.missing_keys:
            accelerator.print(f"Missing keys in encoder load: {missing.missing_keys}")
    decoder = CrossAttentionDecoder(embed_dim)
    mlm_head = nn.Linear(embed_dim, len(tokenizer))
    caption_dec = CaptionDecoder(embed_dim, len(tokenizer)) if args.caption_loss else None
    patch_dec = (
        PatchDecoder(
            embed_dim,
            3 * patch_size * patch_size,
            (img_size // patch_size) ** 2,
        )
        if args.recon_loss
        else None
    )
    if args.domain_loss:
        n_dom = len(domain_manager.names) if domain_manager is not None else len(domain_names)
        domain_head = nn.Linear(embed_dim, n_dom)
    else:
        domain_head = None

    params = [model.parameters(), decoder.parameters(), mlm_head.parameters()]
    if caption_dec is not None:
        params.append(caption_dec.parameters())
    if patch_dec is not None:
        params.append(patch_dec.parameters())
    if domain_head is not None:
        params.append(domain_head.parameters())
    optimizer = torch.optim.AdamW(
        itertools.chain(*params),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(pair_loader))

    modules = [model, decoder, mlm_head]
    if caption_dec is not None:
        modules.append(caption_dec)
    if patch_dec is not None:
        modules.append(patch_dec)
    if domain_head is not None:
        modules.append(domain_head)
    modules.extend([optimizer, scheduler, pair_loader])
    prepared = accelerator.prepare(*modules)
    ptr = 0
    model = prepared[ptr]; ptr += 1
    decoder = prepared[ptr]; ptr += 1
    mlm_head = prepared[ptr]; ptr += 1
    if args.caption_loss:
        caption_dec = prepared[ptr]; ptr += 1
    if args.recon_loss:
        patch_dec = prepared[ptr]; ptr += 1
    if args.domain_loss and domain_head is not None:
        domain_head = prepared[ptr]; ptr += 1
    optimizer = prepared[ptr]; ptr += 1
    scheduler = prepared[ptr]; ptr += 1
    pair_loader = prepared[ptr]
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    base_model = accelerator.unwrap_model(model)

    ce_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        loss_epoch = 0.0
        mlm_epoch = 0.0
        cap_epoch = 0.0
        rec_epoch = 0.0
        dom_epoch = 0.0
        num_batches = 0
        for batch in pair_loader:
            if domain_manager is not None:
                if args.recon_loss:
                    images, tokens, padding, patches, domains = batch
                else:
                    images, tokens, padding, domains = batch
            else:
                if args.recon_loss:
                    images, tokens, padding, patches = batch
                else:
                    images, tokens, padding = batch
            optimizer.zero_grad()
            images = images.to(accelerator.device)
            tokens = tokens.to(accelerator.device)
            padding = padding.to(accelerator.device)
            if args.recon_loss:
                patches = patches.to(accelerator.device)
            if domain_manager is not None:
                domains = list(domains)
                img_emb_dom = domain_manager(images, domains)
                img_emb_dom = img_emb_dom.to(accelerator.device)

            # ----- Contrastive path -----
            with accelerator.no_sync(model):
                img_emb_base, txt_emb = model(
                    image=images,
                    text_description=tokens,
                    padding_mask=padding,
                    return_global=True,
                )
                img_emb = img_emb_dom if domain_manager is not None else img_emb_base
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

            loss_domain = torch.tensor(0.0, device=accelerator.device)
            if args.domain_loss and domain_head is not None:
                if domain_manager is not None:
                    domain_idx = domain_manager.indices(domains).to(accelerator.device)
                else:
                    domain_idx = torch.tensor([domain_to_idx[d] for d in domains], device=accelerator.device)
                dom_logits = domain_head(img_emb_base.detach())
                loss_domain = ce_loss(dom_logits, domain_idx)
                accelerator.backward(loss_domain)

            loss_cap = torch.tensor(0.0, device=accelerator.device)
            if args.caption_loss:
                inp = tokens[:, :-1]
                targ = tokens[:, 1:]
                pad_in = padding[:, :-1]
                pad_targ = padding[:, 1:]
                _, _, _, seq_inp = model(
                    text_description=inp,
                    padding_mask=pad_in,
                    with_head=False,
                    out_norm=False,
                    return_global=False,
                    return_seq=True,
                )
                cap_logits = caption_dec(seq_inp, img_seq, pad_in.bool())
                loss_cap = ce_loss(cap_logits[~pad_targ], targ[~pad_targ])
                accelerator.backward(loss_cap)

            loss_rec = torch.tensor(0.0, device=accelerator.device)
            if args.recon_loss:
                _, _, _, full_txt = model(
                    text_description=tokens,
                    padding_mask=padding,
                    with_head=False,
                    out_norm=False,
                    return_global=False,
                    return_seq=True,
                )
                rec_pred = patch_dec(full_txt)
                loss_rec = F.mse_loss(rec_pred, patches)
                accelerator.backward(loss_rec)

            accelerator.backward(loss_mlm)
            optimizer.step()
            scheduler.step()

            loss_epoch += loss_c.item()
            mlm_epoch += loss_mlm.item()
            cap_epoch += loss_cap.item()
            rec_epoch += loss_rec.item()
            dom_epoch += loss_domain.item()
            num_batches += 1

        denom = accelerator.reduce(torch.tensor(num_batches, device=accelerator.device), reduction="sum")
        contrast_total = accelerator.reduce(torch.tensor(loss_epoch, device=accelerator.device), reduction="sum")
        mlm_total = accelerator.reduce(torch.tensor(mlm_epoch, device=accelerator.device), reduction="sum")
        cap_total = accelerator.reduce(torch.tensor(cap_epoch, device=accelerator.device), reduction="sum") if args.caption_loss else torch.tensor(0.0, device=accelerator.device)
        rec_total = accelerator.reduce(torch.tensor(rec_epoch, device=accelerator.device), reduction="sum") if args.recon_loss else torch.tensor(0.0, device=accelerator.device)
        dom_total = accelerator.reduce(torch.tensor(dom_epoch, device=accelerator.device), reduction="sum") if args.domain_loss else torch.tensor(0.0, device=accelerator.device)

        c_avg = (contrast_total / denom).item()
        mlm_avg = (mlm_total / denom).item()
        cap_avg = (cap_total / denom).item() if args.caption_loss else 0.0
        rec_avg = (rec_total / denom).item() if args.recon_loss else 0.0
        dom_avg = (dom_total / denom).item() if args.domain_loss else 0.0

        if val_loader is not None:
            val_c = 0.0
            val_mlm = 0.0
            val_cap = 0.0
            val_rec = 0.0
            val_dom = 0.0
            val_batches = 0
            for batch in val_loader:
                if domain_manager is not None:
                    if args.recon_loss:
                        images, tokens, padding, patches, domains = batch
                    else:
                        images, tokens, padding, domains = batch
                else:
                    if args.recon_loss:
                        images, tokens, padding, patches = batch
                    else:
                        images, tokens, padding = batch
                images = images.to(accelerator.device)
                tokens = tokens.to(accelerator.device)
                padding = padding.to(accelerator.device)
                if args.recon_loss:
                    patches = patches.to(accelerator.device)
                if domain_manager is not None:
                    domains = list(domains)
                    img_emb_dom = domain_manager(images, domains).to(accelerator.device)
                    
                with torch.no_grad():
                    img_emb_base, txt_emb = model(
                        image=images,
                        text_description=tokens,
                        padding_mask=padding,
                        return_global=True,
                    )
                img_emb = img_emb_dom if domain_manager is not None else img_emb_base
                logit_scale = base_model.logit_scale.exp()
                loss_c = clip_loss(img_emb, txt_emb, logit_scale)

                mask_txt = random_mask(tokens.shape, args.mask_ratio, tokens.device) & (~padding)
                inp_tokens = tokens.clone()
                inp_tokens[mask_txt] = mask_token_id
                with torch.no_grad():
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

                    if args.caption_loss:
                        inp = tokens[:, :-1]
                        targ = tokens[:, 1:]
                        pad_in = padding[:, :-1]
                        pad_targ = padding[:, 1:]
                        _, _, _, seq_inp = model(
                            text_description=inp,
                            padding_mask=pad_in,
                            with_head=False,
                            out_norm=False,
                            return_global=False,
                            return_seq=True,
                        )
                        cap_logits = caption_dec(seq_inp, img_seq, pad_in.bool())
                        loss_cap_val = ce_loss(cap_logits[~pad_targ], targ[~pad_targ])
                    else:
                        loss_cap_val = torch.tensor(0.0, device=accelerator.device)

                    if args.recon_loss:
                        _, _, _, full_txt = model(
                            text_description=tokens,
                            padding_mask=padding,
                            with_head=False,
                            out_norm=False,
                            return_global=False,
                            return_seq=True,
                        )
                        rec_pred = patch_dec(full_txt)
                    loss_rec_val = F.mse_loss(rec_pred, patches)
                else:
                    loss_rec_val = torch.tensor(0.0, device=accelerator.device)

                loss_dom_val = torch.tensor(0.0, device=accelerator.device)
                if args.domain_loss and domain_head is not None:
                    if domain_manager is not None:
                        domain_idx = domain_manager.indices(domains).to(accelerator.device)
                    else:
                        domain_idx = torch.tensor([domain_to_idx[d] for d in domains], device=accelerator.device)
                    dom_logits_val = domain_head(img_emb_base)
                    loss_dom_val = ce_loss(dom_logits_val, domain_idx)

                val_c += loss_c.item()
                val_mlm += loss_mlm.item()
                val_cap += loss_cap_val.item()
                val_rec += loss_rec_val.item()
                val_dom += loss_dom_val.item()
                val_batches += 1

            val_denom = accelerator.reduce(torch.tensor(val_batches, device=accelerator.device), reduction="sum")
            val_c_total = accelerator.reduce(torch.tensor(val_c, device=accelerator.device), reduction="sum")
            val_mlm_total = accelerator.reduce(torch.tensor(val_mlm, device=accelerator.device), reduction="sum")
            val_cap_total = accelerator.reduce(torch.tensor(val_cap, device=accelerator.device), reduction="sum") if args.caption_loss else torch.tensor(0.0, device=accelerator.device)
            val_rec_total = accelerator.reduce(torch.tensor(val_rec, device=accelerator.device), reduction="sum") if args.recon_loss else torch.tensor(0.0, device=accelerator.device)
            val_dom_total = accelerator.reduce(torch.tensor(val_dom, device=accelerator.device), reduction="sum") if args.domain_loss else torch.tensor(0.0, device=accelerator.device)
            c_val_avg = (val_c_total / val_denom).item()
            mlm_val_avg = (val_mlm_total / val_denom).item()
            cap_val_avg = (val_cap_total / val_denom).item() if args.caption_loss else 0.0
            rec_val_avg = (val_rec_total / val_denom).item() if args.recon_loss else 0.0
            dom_val_avg = (val_dom_total / val_denom).item() if args.domain_loss else 0.0
            extras = ""
            if args.caption_loss:
                extras += f" Cap={cap_avg:.4f}"
            if args.recon_loss:
                extras += f" Rec={rec_avg:.4f}"
            if args.domain_loss:
                extras += f" Dom={dom_avg:.4f}"
            accelerator.print(
                f"Epoch {epoch + 1}: Contrastive={c_avg:.4f} MLM={mlm_avg:.4f}{extras} Val_Contrastive={c_val_avg:.4f} Val_MLM={mlm_val_avg:.4f}"
            )
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_contrastive": c_avg,
                        "train_mlm": mlm_avg,
                        **({"train_caption": cap_avg} if args.caption_loss else {}),
                        **({"train_recon": rec_avg} if args.recon_loss else {}),
                        **({"train_domain": dom_avg} if args.domain_loss else {}),
                        "val_contrastive": c_val_avg,
                        "val_mlm": mlm_val_avg,
                        **({"val_caption": cap_val_avg} if args.caption_loss else {}),
                        **({"val_recon": rec_val_avg} if args.recon_loss else {}),
                        **({"val_domain": dom_val_avg} if args.domain_loss else {}),
                    }
                )
        else:
            extras = ""
            if args.caption_loss:
                extras += f" Cap={cap_avg:.4f}"
            if args.recon_loss:
                extras += f" Rec={rec_avg:.4f}"
            if args.domain_loss:
                extras += f" Dom={dom_avg:.4f}"
            accelerator.print(
                f"Epoch {epoch + 1}: Contrastive={c_avg:.4f} MLM={mlm_avg:.4f}{extras}"
            )
            if run:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train_contrastive": c_avg,
                        "train_mlm": mlm_avg,
                        **({"train_caption": cap_avg} if args.caption_loss else {}),
                        **({"train_recon": rec_avg} if args.recon_loss else {}),
                        **({"train_domain": dom_avg} if args.domain_loss else {}),
                    }
                )

    if accelerator.is_main_process:
        accelerator.print(f"Saving model to {args.output}")
        save_dict = {"model": base_model.state_dict()}
        if args.domain_loss and domain_head is not None:
            save_dict["domain_head"] = domain_head.state_dict()
        torch.save(save_dict, args.output)
        if run:
            run.finish()


if __name__ == "__main__":
    main()
