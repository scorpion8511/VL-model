"""Dataset utilities for loading image and text samples from a JSON file.

Each line in the JSON file should be an object with two keys:
```
{"image": "/path/to/image.jpg", "text": "free form caption"}
```

Use ``mode="image"`` to iterate over images only, ``mode="text"`` for text
only, or ``mode="pair"`` to return ``(image, text)`` tuples.
"""

import json
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision

from transformers import PreTrainedTokenizer
from .utils import xlm_tokenizer


def _load_json_lines(path: str) -> List[dict]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class ImageTextJsonDataset(Dataset):
    """Dataset reading samples from a JSON lines file."""

    def __init__(
        self,
        json_file: str,
        mode: str = "image",
        transform: torchvision.transforms.Compose | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        return_patches: bool = False,
        patch_size: int = 16,
        return_domain: bool = False,
        return_path: bool = False,
    ) -> None:
        assert mode in {"image", "text", "pair"}
        self.items = _load_json_lines(json_file)
        self.mode = mode
        self.transform = transform or torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(384, interpolation=3, antialias=True),
                torchvision.transforms.CenterCrop((384, 384)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.tokenizer = tokenizer
        self.return_patches = return_patches
        self.patch_size = patch_size
        self.return_domain = return_domain
        self.return_path = return_path
        self.domains = sorted({item.get("domain") for item in self.items if item.get("domain") is not None})
        self.domain_to_idx = {d: i for i, d in enumerate(self.domains)}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.items)

    def _load_image(self, path: str) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        if self.return_patches:
            patches = F.unfold(img_t.unsqueeze(0), kernel_size=self.patch_size, stride=self.patch_size).squeeze(0).T
            return img_t, patches
        return img_t

    def _load_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.tokenizer is not None, "Tokenizer required for text mode"
        tokens, pad = xlm_tokenizer(text.strip(), self.tokenizer)
        return torch.tensor(tokens), torch.tensor(pad, dtype=torch.bool)

    def __getitem__(self, idx: int):  # type: ignore[override]
        item = self.items[idx]
        image_path = item.get("image")
        caption = item.get("text")
        domain = item.get("domain")

        if self.mode == "image":
            out = self._load_image(image_path)
            extras = []
            if self.return_domain:
                extras.append(domain)
            if self.return_path:
                extras.append(image_path)
            return (out, *extras) if extras else out
        if self.mode == "text":
            out = self._load_text(caption)
            if self.return_domain or self.return_path:
                extras = []
                if self.return_domain:
                    extras.append(domain)
                if self.return_path:
                    extras.append(image_path)
                return out + tuple(extras)
            return out

        img = self._load_image(image_path)
        if self.return_patches:
            img, patches = img  # type: ignore
            out = (img, patches) + self._load_text(caption)
        else:
            out = (img,) + self._load_text(caption)
        extras = []
        if self.return_domain:
            extras.append(domain)
        if self.return_path:
            extras.append(image_path)
        return out + tuple(extras)


def get_json_loader(
    json_file: str,
    mode: str,
    batch_size: int,
    num_workers: int,
    tokenizer: PreTrainedTokenizer | None = None,
    return_patches: bool = False,
    patch_size: int = 16,
    return_domain: bool = False,
    return_path: bool = False,
) -> DataLoader:
    dataset = ImageTextJsonDataset(
        json_file,
        mode=mode,
        tokenizer=tokenizer,
        return_patches=return_patches,
        patch_size=patch_size,
        return_domain=return_domain,
        return_path=return_path,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_json_loaders(
    json_file: str,
    mode: str,
    batch_size: int,
    num_workers: int,
    tokenizer: PreTrainedTokenizer | None = None,
    val_split: float = 0.1,
    return_patches: bool = False,
    patch_size: int = 16,
    return_domain: bool = False,
    return_path: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Return training and validation loaders split from a JSON lines dataset."""
    dataset = ImageTextJsonDataset(
        json_file,
        mode=mode,
        tokenizer=tokenizer,
        return_patches=return_patches,
        patch_size=patch_size,
        return_domain=return_domain,
        return_path=return_path,
    )
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader
