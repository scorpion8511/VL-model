from typing import Tuple, Callable, Dict, Iterable, List
from pathlib import Path
import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel
from timm.models import create_model


def load_xray_encoder() -> Tuple[AutoImageProcessor, AutoModel]:
    """Load the pretrained encoder for the x-ray domain using Rad-DINO."""
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    return processor, model


def load_local_encoder(path: str) -> Tuple[None, nn.Module]:
    """Load a MUSK encoder from a local ``.pth`` checkpoint."""
    model = create_model("musk_large_patch16_384")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return None, model


# Default file paths for locally stored encoders. These paths can be adjusted
# if the models are saved elsewhere.
DOMAIN_ENCODER_FILES: Dict[str, str] = {
    "xray": "/home/jovyan/work/MUSK/Encoder_factory_dewan/xray.pth",
    "patho": "/home/jovyan/work/MUSK/Encoder_factory_dewan/patho.pth",
    "endo": "/home/jovyan/work/MUSK/Encoder_factory_dewan/endo.pth",
}

DOMAIN_ENCODERS: Dict[str, Callable[[], Tuple[AutoImageProcessor | None, nn.Module]]] = {
    "xray": load_xray_encoder,
}


class DomainGate(nn.Module):
    """Simple mapping from domain names to expert indices."""

    def __init__(self, domains: Iterable[str]):
        super().__init__()
        self.domains = list(domains)
        self.mapping = {n: i for i, n in enumerate(self.domains)}

    def forward(self, names: List[str]) -> torch.Tensor:
        idx = [self.mapping[n] for n in names]
        return torch.tensor(idx, dtype=torch.long)


def get_domain_encoder(name: str) -> Tuple[AutoImageProcessor | None, nn.Module]:
    """Return the encoder corresponding to ``name``.

    ``name`` can be either a known domain key or a path to a checkpoint. If the
    name matches a key in ``DOMAIN_ENCODERS`` the corresponding loader is used.
    Otherwise ``name`` is treated as a local ``.pth`` file and loaded via
    :func:`load_local_encoder`.
    """
    if name in DOMAIN_ENCODERS:
        return DOMAIN_ENCODERS[name]()
    if name in DOMAIN_ENCODER_FILES:
        return load_local_encoder(DOMAIN_ENCODER_FILES[name])
    if Path(name).is_file():
        return load_local_encoder(name)
    raise ValueError(f"Unknown domain '{name}'")


class DomainEncoderManager(nn.Module):
    """Wrapper holding multiple domain-specific encoders with gating."""

    def __init__(self, names: Iterable[str]):
        super().__init__()
        self.names = list(names)
        self.encoders = nn.ModuleDict()
        self.processors: Dict[str, AutoImageProcessor | None] = {}
        for n in self.names:
            proc, model = get_domain_encoder(n)
            self.processors[n] = proc
            self.encoders[n] = model
        self.gate = DomainGate(self.names)

    def forward(self, images: torch.Tensor, domains: List[str]) -> torch.Tensor:
        """Encode a batch of images using domain-specific encoders."""
        device = images.device
        idxs = self.gate(domains)
        outs: list[torch.Tensor] = [torch.empty(0, device=device)] * len(domains)
        for expert_idx in idxs.unique():
            mask = idxs == expert_idx
            if not mask.any():
                continue
            name = self.names[int(expert_idx)]
            imgs = images[mask]
            proc = self.processors[name]
            enc = self.encoders[name]
            if proc is not None:
                proc_out = proc(images=list(imgs), return_tensors="pt")
                inp = proc_out[proc.model_input_names[0]].to(device)
            else:
                inp = imgs.to(device)
            out = enc(inp)
            feats = (
                out.last_hidden_state[:, 0]
                if hasattr(out, "last_hidden_state")
                else out[0]
            )
            for i, f in zip(mask.nonzero(as_tuple=True)[0].tolist(), feats):
                outs[i] = f
        return torch.stack(outs)


def load_domain_encoders(names: Iterable[str]) -> DomainEncoderManager:
    """Convenience function to create :class:`DomainEncoderManager`."""
    return DomainEncoderManager(list(names))
