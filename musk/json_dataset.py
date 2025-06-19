import json
from typing import List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size

    def __call__(self, text: str) -> List[int]:
        tokens = text.strip().split()
        return [abs(hash(tok)) % self.vocab_size for tok in tokens]


def default_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class JsonDataset(Dataset):
    """Dataset reading image-caption entries from a JSON lines file.

    Each line should contain an ``image`` path and either a single ``text``
    string or a list of ``captions``.  When multiple captions are present the
    image will be repeated once for each caption.
    """

    def __init__(self, json_path: str, transform=None, tokenizer=None, max_length: int = 32):
        raw_items = [json.loads(line) for line in open(json_path, "r")]
        self.items = []
        for obj in raw_items:
            captions = []
            if "text" in obj:
                if isinstance(obj["text"], list):
                    captions = obj["text"]
                else:
                    captions = [obj["text"]]
            elif "captions" in obj:
                captions = obj["captions"]
            else:
                raise KeyError("JSON entry must contain 'text' or 'captions'")
            for cap in captions:
                self.items.append({"image": obj["image"], "text": cap})
        self.transform = transform or default_transform()
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        img = Image.open(entry["image"]).convert("RGB")
        img = self.transform(img)
        token_ids = self.tokenizer(entry["text"])
        token_ids = token_ids[: self.max_length]
        padding_mask = torch.zeros(self.max_length, dtype=torch.long)
        if len(token_ids) < self.max_length:
            padding_mask[len(token_ids):] = 1
            token_ids += [0] * (self.max_length - len(token_ids))
        return img, torch.tensor(token_ids, dtype=torch.long), padding_mask

