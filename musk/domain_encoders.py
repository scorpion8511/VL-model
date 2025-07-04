from typing import Tuple, Callable, Dict, Iterable, List
import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel


def load_xray_encoder() -> Tuple[AutoImageProcessor, AutoModel]:
    """Load the pretrained encoder for the x-ray domain using Rad-DINO."""
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    return processor, model


DOMAIN_ENCODERS: Dict[str, Callable[[], Tuple[AutoImageProcessor, AutoModel]]] = {
    "xray": load_xray_encoder,
}


def get_domain_encoder(name: str) -> Tuple[AutoImageProcessor, AutoModel]:
    """Return the encoder corresponding to ``name``."""
    if name not in DOMAIN_ENCODERS:
        raise ValueError(f"Unknown domain '{name}'")
    return DOMAIN_ENCODERS[name]()


class DomainEncoderManager(nn.Module):
    """Wrapper holding multiple domain-specific encoders."""

    def __init__(self, names: Iterable[str]):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.processors: Dict[str, AutoImageProcessor] = {}
        for n in names:
            proc, model = get_domain_encoder(n)
            self.processors[n] = proc
            self.encoders[n] = model

    def forward(self, images: torch.Tensor, domains: List[str]) -> torch.Tensor:
        """Encode a batch of images using domain-specific encoders."""
        device = images.device
        outs: list[torch.Tensor] = [torch.empty(0)] * len(domains)
        for name in set(domains):
            idx = [i for i, d in enumerate(domains) if d == name]
            if not idx:
                continue
            imgs = images[idx]
            proc = self.processors[name]
            enc = self.encoders[name]
            proc_out = proc(images=list(imgs), return_tensors="pt")
            inp = proc_out[proc.model_input_names[0]].to(device)
            feats = enc(inp).last_hidden_state[:, 0]
            for i, f in zip(idx, feats):
                outs[i] = f
        return torch.stack(outs)


def load_domain_encoders(names: Iterable[str]) -> DomainEncoderManager:
    """Convenience function to create :class:`DomainEncoderManager`."""
    return DomainEncoderManager(list(names))
