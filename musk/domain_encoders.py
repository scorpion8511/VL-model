from typing import Tuple, Callable, Dict

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
