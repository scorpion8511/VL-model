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
    # use ``weights_only=False`` since MUSK checkpoints may contain full pickled objects
    # and recent PyTorch defaults to ``weights_only=True``
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    return None, model


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
    if Path(name).is_file():
        return load_local_encoder(name)
    raise ValueError(f"Unknown domain '{name}'")


class DomainEncoderManager(nn.Module):
    """Wrapper holding multiple domain-specific encoders with gating."""

    def __init__(self, specs: Iterable[str]):
        """Create domain encoders from specification strings.

        Each entry in ``specs`` may be either ``"name"`` to load a built-in
        encoder or ``"name=path"`` to load a checkpoint from ``path`` while using
        ``name`` for routing.
        """
        super().__init__()
        self.names: list[str] = []
        self.encoders = nn.ModuleDict()
        self.processors: Dict[str, AutoImageProcessor | None] = {}

        for spec in specs:
            if "=" in spec:
                name, src = spec.split("=", 1)
            else:
                name, src = spec, spec
            self.names.append(name)
            proc, model = get_domain_encoder(src)
            self.processors[name] = proc
            self.encoders[name] = model

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


def load_domain_encoders(specs: Iterable[str]) -> DomainEncoderManager:
    """Create :class:`DomainEncoderManager` from specification strings."""
    return DomainEncoderManager(list(specs))
