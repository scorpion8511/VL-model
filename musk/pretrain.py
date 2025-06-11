import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.models import create_model

from transformers import AutoTokenizer

from . import modeling  # required for timm model registration  # noqa: F401


def get_parser():
    parser = argparse.ArgumentParser(description="MUSK pretrain demo")
    parser.add_argument("--json-data", type=str, required=True,
                        help="Path to JSONL file with 'image' and 'text' fields")
    parser.add_argument("--model", type=str, default="musk_large_patch16_384")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    return parser


class JsonlDataset(Dataset):
    def __init__(self, path, transform, tokenizer):
        self.items = []
        with open(path, "r") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = Path(item["image"])
        image = self.transform(Image.open(img_path).convert("RGB"))
        tokens = self.tokenizer(
            item["text"], return_tensors="pt", padding="max_length",
            max_length=64, truncation=True
        )
        ids = tokens.input_ids.squeeze(0)
        padding_mask = tokens.attention_mask.eq(0)
        return image, ids, padding_mask


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor()
    ])

    dataset = JsonlDataset(args.json_data, transform, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = create_model(args.model).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(args.epochs):
        for imgs, tokens, mask in dataloader:
            imgs = imgs.to(device)
            tokens = tokens.to(device)
            mask = mask.to(device)
            vision, text = model(image=imgs, text_description=tokens, padding_mask=mask)
            logits = vision @ text.t() * model.logit_scale.exp()
            target = torch.arange(len(imgs), device=device)
            loss = (nn.functional.cross_entropy(logits, target) +
                    nn.functional.cross_entropy(logits.t(), target)) / 2
            loss.backward()
            optim.step()
            optim.zero_grad()
            print(f"loss: {loss.item():.4f}")

    print(f"Model {args.model} trained for {args.epochs} epoch(s)")
    return model


if __name__ == "__main__":
    main()
