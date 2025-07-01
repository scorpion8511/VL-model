import json
import tempfile
from pathlib import Path
from PIL import Image
import torch

from musk.json_dataset import ImageTextJsonDataset


def test_domain_label_image_mode():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "a" / "img.jpg"
        img_path.parent.mkdir()
        Image.new("RGB", (384, 384)).save(img_path)
        data = [{"image": str(img_path), "text": "t", "domain": 1}]
        json_file = Path(tmp) / "data.jsonl"
        with open(json_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        ds = ImageTextJsonDataset(str(json_file), mode="image")
        img, domain = ds[0]
        assert isinstance(img, torch.Tensor)
        assert domain == 1
