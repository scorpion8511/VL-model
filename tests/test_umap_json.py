import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

from musk.json_dataset import ImageTextJsonDataset
from musk.umap_json import collect_embeddings


class DummyModel(torch.nn.Module):
    def forward(self, image=None, **kwargs):
        batch = image.size(0)
        return torch.ones(batch, 4), None


def make_dummy_json(tmp_path: Path, n: int = 3) -> Path:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    entries = []
    for i in range(n):
        arr = torch.randint(0, 255, (8, 8, 3), dtype=torch.uint8).numpy()
        img_path = img_dir / f"{i}.png"
        Image.fromarray(arr).save(img_path)
        entries.append({"image": str(img_path)})
    json_file = tmp_path / "data.jsonl"
    with open(json_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return json_file


def test_collect_embeddings_shape(tmp_path: Path) -> None:
    json_file = make_dummy_json(tmp_path)
    dataset = ImageTextJsonDataset(str(json_file), mode="image")
    loader = DataLoader(dataset, batch_size=2)
    feats, labels = collect_embeddings(DummyModel(), loader, torch.device("cpu"))
    assert feats.shape == (len(dataset), 4)
    assert labels is None
