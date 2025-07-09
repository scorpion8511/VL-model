"""Visualize image embeddings from a JSON dataset using UMAP.

This utility loads a MUSK checkpoint, extracts image embeddings from a JSON
lines dataset and plots a 2‑D UMAP projection. When the JSON objects include a
``domain`` key, domain labels are used for clustering and colored in the plot.

Example:
    python -m musk.umap_json \
        --json-data images.jsonl \
        --checkpoint musk_stage2.pt \
        --arch musk_large_patch16_384 \
        --output umap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader
from timm.models import create_model

# Importing modeling registers MUSK architectures with timm
from . import modeling  # noqa: F401

from .json_dataset import ImageTextJsonDataset
from .utils import load_model_and_may_interpolate


def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_domain: bool = False,
) -> tuple[torch.Tensor, list[str] | None]:
    """Return CLS embeddings for all images in ``loader``."""

    embeds: List[torch.Tensor] = []
    domains: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if return_domain:
                images, dom = batch
                domains.extend([str(d) for d in dom])
            else:
                images = batch
            images = images.to(device)
            feats = model(image=images, return_global=True, with_head=False)[0]
            embeds.append(feats.cpu())

    feats = torch.cat(embeds, dim=0)
    if return_domain:
        return feats, domains
    return feats, None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="UMAP visualization from JSON dataset")
    parser.add_argument("--json-data", type=str, required=True, help="JSON lines file with image paths")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--arch", type=str, default="musk_large_patch16_384", help="Model architecture name")
    parser.add_argument("--output", type=str, default="umap.png", help="Output image path")
    parser.add_argument("--return-domain", action="store_true", help="Use domain labels from the JSON file")
    args = parser.parse_args(argv)

    dataset = ImageTextJsonDataset(args.json_data, mode="image", return_domain=args.return_domain)
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.arch)
    load_model_and_may_interpolate(args.checkpoint, model, "model|module", "")
    model.to(device)

    feats, labels = collect_embeddings(model, loader, device, args.return_domain)

    # Run clustering and UMAP only when sklearn/umap are available.
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score
    import umap
    import matplotlib.pyplot as plt
    import numpy as np

    if labels:
        n_clusters = len(set(labels))
        km = KMeans(n_clusters=n_clusters, n_init="auto").fit(feats.numpy())
        v_score = v_measure_score(labels, km.labels_)
        print(f"V-measure: {v_score:.3f}")

    proj = umap.UMAP(random_state=42).fit_transform(feats.numpy())
    plt.figure(figsize=(6, 6))
    if labels:
        labels_arr = np.array(labels)
        for lab in sorted(set(labels)):
            mask = labels_arr == lab
            plt.scatter(proj[mask, 0], proj[mask, 1], s=6, label=str(lab))
        plt.legend()
    else:
        plt.scatter(proj[:, 0], proj[:, 1], s=6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
