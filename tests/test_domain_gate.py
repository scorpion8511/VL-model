import torch
import torch.nn as nn
from musk.domain_encoders import DomainGate, DomainEncoderManager, DOMAIN_ENCODERS


def test_domain_gate_indices():
    gate = DomainGate(["a", "b", "c"])
    idx = gate(["a", "c", "b", "a"])
    assert idx.tolist() == [0, 2, 1, 0]


def test_domain_encoder_manager_routing():
    original = DOMAIN_ENCODERS.copy()

    class Dummy(nn.Module):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def forward(self, x):
            return (torch.full((x.size(0), 1), self.val),)

    DOMAIN_ENCODERS.clear()
    DOMAIN_ENCODERS["a"] = lambda: (None, Dummy(1))
    DOMAIN_ENCODERS["b"] = lambda: (None, Dummy(2))
    mgr = DomainEncoderManager(["a", "b"])
    imgs = torch.zeros(3, 3, 2, 2)
    feats = mgr(imgs, ["a", "b", "a"])
    assert feats.shape == (3, 1)
    assert feats[0, 0] == 1
    assert feats[1, 0] == 2
    assert feats[2, 0] == 1

    idx = mgr.indices(["b", "a"])
    assert idx.tolist() == [1, 0]
    DOMAIN_ENCODERS.clear()
    DOMAIN_ENCODERS.update(original)


def test_domain_encoder_manager_three_domains():
    original = DOMAIN_ENCODERS.copy()

    class Dummy(nn.Module):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def forward(self, x):
            return (torch.full((x.size(0), 1), self.val),)

    DOMAIN_ENCODERS.clear()
    DOMAIN_ENCODERS["xray"] = lambda: (None, Dummy(1))
    DOMAIN_ENCODERS["mri"] = lambda: (None, Dummy(2))
    DOMAIN_ENCODERS["patho"] = lambda: (None, Dummy(3))
    mgr = DomainEncoderManager(["xray", "mri", "patho"])
    imgs = torch.zeros(3, 3, 2, 2)
    feats = mgr(imgs, ["mri", "xray", "patho"])
    assert feats[0, 0] == 2
    assert feats[1, 0] == 1
    assert feats[2, 0] == 3
    DOMAIN_ENCODERS.clear()
    DOMAIN_ENCODERS.update(original)


def test_domain_encoder_manager_four_domains():
    original = DOMAIN_ENCODERS.copy()

    class Dummy(nn.Module):
        def __init__(self, val: int):
            super().__init__()
            self.val = val

        def forward(self, x):
            return (torch.full((x.size(0), 1), self.val),)

    DOMAIN_ENCODERS.clear()
    DOMAIN_ENCODERS["xray"] = lambda: (None, Dummy(1))
    DOMAIN_ENCODERS["mri"] = lambda: (None, Dummy(2))
    DOMAIN_ENCODERS["patho"] = lambda: (None, Dummy(3))
    DOMAIN_ENCODERS["endo"] = lambda: (None, Dummy(4))
    mgr = DomainEncoderManager(["xray", "mri", "patho", "endo"])
    imgs = torch.zeros(4, 3, 2, 2)
    feats = mgr(imgs, ["endo", "mri", "xray", "patho"])
    assert feats.shape == (4, 1)
    assert feats[0, 0] == 4
    assert feats[1, 0] == 2
    assert feats[2, 0] == 1
    assert feats[3, 0] == 3
    DOMAIN_ENCODERS.clear()
    DOMAIN_ENCODERS.update(original)
