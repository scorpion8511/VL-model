"""Dataset utilities for loading image and text samples from a JSON file.

Each line in the JSON file should be an object with at least ``image`` and
``text`` fields:
```
{"image": "/path/to/image.jpg", "text": "free form caption"}
```

Use ``mode="image"`` to iterate over images only, ``mode="text"`` for text
only, or ``mode="pair"`` to return ``(image, text)`` tuples. When a ``domain``
field is present and ``mode="pair"``, it is returned as an integer label after
the text tensors.
"""

import json
from typing import List, Tuple

import torch
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

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.items)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

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
            return self._load_image(image_path)
        if self.mode == "text":
            return self._load_text(caption)

        pair = (self._load_image(image_path),) + self._load_text(caption)
        if domain is not None:
            pair = pair + (torch.tensor(int(domain)),)
        return pair


def get_json_loader(
    json_file: str,
    mode: str,
    batch_size: int,
    num_workers: int,
    tokenizer: PreTrainedTokenizer | None = None,
) -> DataLoader:
    dataset = ImageTextJsonDataset(json_file, mode=mode, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_json_loaders(
    json_file: str,
    mode: str,
    batch_size: int,
    num_workers: int,
    tokenizer: PreTrainedTokenizer | None = None,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """Return training and validation loaders split from a JSON lines dataset."""
    dataset = ImageTextJsonDataset(json_file, mode=mode, tokenizer=tokenizer)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader
