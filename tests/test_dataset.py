import json
import sys
from pathlib import Path

import pytest
import types

# provide minimal stubs when torch or torchvision are missing
if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.tensor = lambda *a, **k: a[0]
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object, DataLoader=object))
    torch.nn = types.SimpleNamespace(functional=types.ModuleType('functional'))
    class _Tensor: pass
    torch.Tensor = _Tensor
    sys.modules['torch'] = torch
    sys.modules['torch.utils'] = torch.utils
    sys.modules['torch.utils.data'] = torch.utils.data
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.nn.functional'] = torch.nn.functional

if 'torchvision' not in sys.modules:
    tv = types.ModuleType('torchvision')
    class _T:
        def __init__(self, *a, **k): pass
        def __call__(self, x): return x
    class _Compose:
        def __init__(self, *a, **k): pass
        def __call__(self, x): return x
    tv.transforms = types.SimpleNamespace(
        Compose=_Compose,
        Resize=_T,
        CenterCrop=_T,
        ToTensor=_T,
    )
    sys.modules['torchvision'] = tv

if 'PIL' not in sys.modules:
    pil = types.ModuleType('PIL')
    class Image:
        @staticmethod
        def open(path):
            class Img:
                def convert(self, mode):
                    return self
            return Img()
    pil.Image = Image
    sys.modules['PIL'] = pil
    sys.modules['PIL.Image'] = pil.Image

if 'transformers' not in sys.modules:
    tr = types.ModuleType('transformers')
    class PreTrainedTokenizer:
        pass
    tr.PreTrainedTokenizer = PreTrainedTokenizer
    sys.modules['transformers'] = tr

if 'huggingface_hub' not in sys.modules:
    hh = types.ModuleType('huggingface_hub')
    hh.snapshot_download = lambda *a, **k: None
    hh.hf_hub_download = lambda *a, **k: None
    sys.modules['huggingface_hub'] = hh

if 'safetensors.torch' not in sys.modules:
    st = types.ModuleType('safetensors.torch')
    st.load_file = lambda *a, **k: {}
    sys.modules['safetensors.torch'] = st
    base = types.ModuleType('safetensors')
    base.torch = st
    sys.modules['safetensors'] = base

if 'einops' not in sys.modules:
    einops = types.ModuleType('einops')
    einops.rearrange = lambda x, *a, **k: x
    sys.modules['einops'] = einops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
from musk.json_dataset import ImageTextJsonDataset


def test_string_domain_mapping(tmp_path):
    items = [
        {"image": "img1.png", "text": "a", "domain": "foo"},
        {"image": "img2.png", "text": "b", "domain": "bar"},
        {"image": "img3.png", "text": "c", "domain": "foo"},
    ]
    json_file = tmp_path / "data.jsonl"
    with open(json_file, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    ds = ImageTextJsonDataset(str(json_file))
    domains = [ds._infer_domain(item) for item in ds.items]
    assert domains == [0, 1, 0]
    assert ds.domain_map == {"foo": 0, "bar": 1}
