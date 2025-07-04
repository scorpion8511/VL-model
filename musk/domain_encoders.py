"""Utilities for loading domain-specific image encoders from local directories."""

from __future__ import annotations

from transformers import AutoImageProcessor, AutoModel


# Map each domain to the directory containing its pretrained weights
DOMAIN_ENCODER_PATHS: dict[str, str] = {
    "xray": "models/xray",
    "endoscopy": "models/endoscopy",
    "pathology": "models/pathology",
}


def load_local_encoder(path: str) -> tuple[AutoImageProcessor, AutoModel]:
    """Load an image processor and model from ``path``.

    Parameters
    ----------
    path:
        Path to a directory with Hugging Face ``config.json`` and weight files.
    """
    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    return processor, model


def load_xray_encoder() -> tuple[AutoImageProcessor, AutoModel]:
    """Return the X-ray encoder loaded from :data:`DOMAIN_ENCODER_PATHS`."""
    return load_local_encoder(DOMAIN_ENCODER_PATHS["xray"])


def load_endoscopy_encoder() -> tuple[AutoImageProcessor, AutoModel]:
    """Return the endoscopy encoder loaded from :data:`DOMAIN_ENCODER_PATHS`."""
    return load_local_encoder(DOMAIN_ENCODER_PATHS["endoscopy"])


def load_pathology_encoder() -> tuple[AutoImageProcessor, AutoModel]:
    """Return the pathology encoder loaded from :data:`DOMAIN_ENCODER_PATHS`."""
    return load_local_encoder(DOMAIN_ENCODER_PATHS["pathology"])


# Mapping from domain name to loader function
DOMAIN_ENCODERS: dict[str, callable[[], tuple[AutoImageProcessor, AutoModel]]] = {
    "xray": load_xray_encoder,
    "endoscopy": load_endoscopy_encoder,
    "pathology": load_pathology_encoder,
}
