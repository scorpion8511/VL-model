import torch
import torch.nn as nn
from musk.domain_encoders import DomainEncoderManager, IdentityGate

class DummyEncoder(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # simple feature: mean over spatial dims scaled by factor
        x = x.mean(dim=(2, 3)) * self.factor
        return x

def test_domain_encoder_manager_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.randn(4, 3, 8, 8).to(device)
    domains = torch.tensor([0, 1, 0, 1])  # remains on CPU

    enc1 = DummyEncoder(1.0)
    enc2 = DummyEncoder(2.0)
    manager = DomainEncoderManager([enc1, enc2], gate=IdentityGate())

    out = manager(images, domains)
    assert out.device == device
    assert out.shape[0] == images.shape[0]
