import json
from PIL import Image
import torch
from musk.json_dataset import ImageTextJsonDataset


def test_domain_mapping(tmp_path):
    # create tiny image
    img = Image.new('RGB', (4,4), color='white')
    img_path = tmp_path/'a.jpg'
    img.save(img_path)
    data = [
        {"image": str(img_path), "text": "t1", "domain": "x"},
        {"image": str(img_path), "text": "t2", "domain": "y"},
    ]
    json_path = tmp_path/'data.jsonl'
    with open(json_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item)+'\n')
    ds = ImageTextJsonDataset(str(json_path), return_domain=True)
    assert ds.domains == ['x', 'y']
    img, domain = ds[0]
    assert domain == 'x'
