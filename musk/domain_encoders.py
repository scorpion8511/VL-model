from typing import Tuple, Callable, Dict, Iterable, List, Sequence
from pathlib import Path
import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel
from diffusers.models import AutoencoderKL
from timm.models import create_model


def load_xray_encoder() -> Tuple[AutoImageProcessor, AutoModel]:
    """Load the pretrained encoder for the x-ray domain using Rad-DINO."""
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    return processor, model


def load_mri_encoder() -> Tuple[None, AutoencoderKL]:
    """Load the pretrained encoder for MRI images."""
    model = AutoencoderKL.from_pretrained("microsoft/mri-autoencoder-v0.1")
    return None, model


def load_local_encoder(path: str) -> Tuple[None, nn.Module]:
    """Load a MUSK encoder from a local ``.pth`` checkpoint.

    ``torch.load`` is wrapped in a ``try`` block so we can surface a clearer
    error when the file is not a valid PyTorch checkpoint. This often manifests
    as ``UnpicklingError: invalid load key`` when the path points to an invalid
    or corrupted file.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(path)

    model = create_model("musk_large_patch16_384")
    try:
        # ``weights_only=False`` handles older MUSK checkpoints that store
        # objects other than tensors.
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        # fall back to safetensors format
        try:
            from safetensors.torch import load_file as load_safetensors

            state = load_safetensors(path)
        except Exception:
            raise RuntimeError(f"Failed to load checkpoint '{path}': {e}") from e

    if hasattr(state, "state_dict"):
        state = state.state_dict()

    model.load_state_dict(state, strict=False)
    return None, model


DOMAIN_ENCODERS: Dict[str, Callable[[], Tuple[AutoImageProcessor | None, nn.Module]]] = {
    "xray": load_xray_encoder,
    "mri": load_mri_encoder,
}


def parse_domain_list(arg: Sequence[str] | str | None) -> List[str]:
    """Parse a ``--domains`` argument into a list of specification strings.

    The CLI may provide the value either as a single comma-separated string or
    as a sequence of tokens when users include spaces. This helper handles both
    cases and strips extraneous whitespace.
    """
    if not arg:
        return []
    if isinstance(arg, (list, tuple)):
        arg = " ".join(arg)
    return [p.strip() for p in str(arg).split(',') if p.strip()]


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

    def indices(self, domains: List[str]) -> torch.Tensor:
        """Return the expert index for each domain in ``domains``."""
        return self.gate(domains)

    def forward(self, images: torch.Tensor, domains: List[str]) -> torch.Tensor:
        """Encode a batch of images using domain-specific encoders."""
        device = images.device
        idxs = self.gate(domains)
        outs: list[torch.Tensor | None] = [None for _ in domains]
        for expert_idx in torch.unique(idxs, sorted=True):
            mask = idxs == expert_idx
            if not mask.any():
                continue
            name = self.names[int(expert_idx)]
            imgs = images[mask]
            proc = self.processors[name]
            enc = self.encoders[name]
            if name == "mri" and imgs.size(1) != 2:
                # the pretrained MRI autoencoder expects 2-channel inputs
                # convert RGB or single-channel images to two channels by
                # averaging then repeating
                gray = imgs.mean(dim=1, keepdim=True)
                imgs = gray.repeat(1, 2, 1, 1)
            if proc is not None:
                proc_out = proc(images=list(imgs), return_tensors="pt")
                inp = proc_out[proc.model_input_names[0]].to(device)
            else:
                inp = imgs.to(device)
            out = enc(inp)
            if hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state[:, 0]
            elif isinstance(out, (tuple, list)):
                out = out[0]
                feats = out
            else:
                feats = out
            if feats.dim() > 2:
                feats = feats.mean(dim=(2, 3))
            for i, f in zip(mask.nonzero(as_tuple=True)[0].tolist(), feats):
                outs[i] = f
        return torch.stack([o for o in outs])


def load_domain_encoders(specs: Iterable[str]) -> DomainEncoderManager:
    """Create :class:`DomainEncoderManager` from specification strings."""
    return DomainEncoderManager(list(specs))
