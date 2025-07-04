import os
import torch
import tempfile
from musk.domain_encoders import load_local_encoder
from safetensors.torch import save_file


def test_load_local_encoder_torch(tmp_path):
    model = torch.nn.Linear(3, 2)
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)
    _, loaded = load_local_encoder(str(ckpt))
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.allclose(a, b)


def test_load_local_encoder_safetensor(tmp_path):
    model = torch.nn.Linear(4, 1)
    ckpt = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(ckpt))
    _, loaded = load_local_encoder(str(ckpt))
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.allclose(a, b)
