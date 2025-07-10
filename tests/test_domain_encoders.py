import torch
from types import SimpleNamespace
from musk.domain_encoders import load_mri_encoder, DomainEncoderManager


class DummyAutoencoder:
    def __init__(self, in_channels=2):
        self.config = SimpleNamespace(in_channels=in_channels)

    class _LatentDist:
        def __init__(self, x):
            self.x = x
        def sample(self):
            return self.x

    def encode(self, x):
        return SimpleNamespace(latent_dist=self._LatentDist(x))


def test_mri_wrapper_accepts_three_channel_input():
    ae = DummyAutoencoder(in_channels=2)
    manager = DomainEncoderManager({"mri": lambda: load_mri_encoder(ae)})
    img = torch.randn(1, 3, 64, 64)
    manager.encode("mri", img)
