# --------------------------------------------------------
# MUSK: A Vision-Language Foundation Model for Precision Oncology
# Published in Nature, 2025
# GitHub Repository: https://github.com/lilab-stanford/MUSK
# Copyright (c) 2025 Stanford University
# Licensed under the CC-BY-NC-ND 4.0 License (https://creativecommons.org/licenses/by-nc-nd/4.0/)
# Please see LICENSE for additional details.
# --------------------------------------------------------
"""Domain-specific encoder utilities."""

from __future__ import annotations

from typing import Callable, Dict
from types import SimpleNamespace

import torch


class DomainEncoderManager:
    """Simple manager for domain-specific encoders."""

    def __init__(self, loaders: Dict[str, Callable[[], Callable[[torch.Tensor], tuple]]]):
        self.loaders = loaders
        self.encoders: Dict[str, Callable[[torch.Tensor], tuple]] = {}

    def get_encoder(self, domain: str) -> Callable[[torch.Tensor], tuple]:
        if domain not in self.encoders:
            self.encoders[domain] = self.loaders[domain]()
        return self.encoders[domain]

    def encode(self, domain: str, images: torch.Tensor) -> tuple:
        return self.get_encoder(domain)(images)

    __call__ = encode


def load_mri_encoder(autoencoder) -> Callable[[torch.Tensor], tuple]:
    """Return a wrapper around ``AutoencoderKL`` for MRI images."""

    in_channels = autoencoder.config.in_channels

    def encode(images: torch.Tensor) -> tuple:
        if images.shape[1] != in_channels:
            if images.shape[1] > in_channels:
                images = images[:, :in_channels]
            else:
                repeat = (in_channels + images.shape[1] - 1) // images.shape[1]
                images = images.repeat(1, repeat, 1, 1)[:, :in_channels]
        latents = autoencoder.encode(images).latent_dist.sample()
        pooled = latents.mean(dim=(-2, -1))
        return (pooled,)

    return encode
