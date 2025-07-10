# --------------------------------------------------------
# MUSK: A Vision-Language Foundation Model for Precision Oncology
# Published in Nature, 2025
# GitHub Repository: https://github.com/lilab-stanford/MUSK
# --------------------------------------------------------
"""UMAP visualization utilities for JSON embeddings."""

from __future__ import annotations

import argparse
import json
import math
import random
import os
from typing import List, Optional, Tuple
from pathlib import Path

import pickle


def _load_json_lines(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def collect_embeddings(json_file: str, model_path: Optional[str] = None) -> Tuple[List[List[float]], Optional[List[str]]]:
    """Return embeddings and optional domain labels from a JSON lines file.

    If ``model_path`` is provided or the JSON items do not contain an ``embedding``
    field, embeddings are computed using the specified MUSK model. Items may
    contain either an ``image`` or ``text`` field.
    """

    items = _load_json_lines(json_file)
    embeddings: List[List[float]] = []
    domains: List[str] = []

    # Use embeddings from JSON when available and no model is specified
    if model_path is None and all("embedding" in it for it in items):
        for it in items:
            embeddings.append([float(x) for x in it["embedding"]])
            if "domain" in it:
                domains.append(it["domain"])
        labels = domains if domains else None
        return embeddings, labels

    if model_path is None:
        raise ValueError("Embeddings missing and no model checkpoint provided")

    import torch
    from PIL import Image
    import torchvision.transforms as T
    from transformers import XLMRobertaTokenizer
    from timm.models import create_model
    from .utils import xlm_tokenizer, load_model_and_may_interpolate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("musk_large_patch16_384")
    load_model_and_may_interpolate(model_path, model, "model|module", "")
    model.to(device)
    model.eval()

    tokenizer_path = Path(__file__).resolve().parent / "models" / "tokenizer.spm"
    tokenizer = None
    transform = T.Compose([
        T.Resize(384, interpolation=3, antialias=True),
        T.CenterCrop((384, 384)),
        T.ToTensor(),
    ])

    for it in items:
        if "image" in it:
            img = Image.open(it["image"]).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat, _ = model(image=inp)
            vec = feat[0].cpu().tolist()
        elif "text" in it:
            if tokenizer is None:
                tokenizer = XLMRobertaTokenizer(str(tokenizer_path))
            tokens, pad = xlm_tokenizer(it["text"], tokenizer)
            tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            pad = torch.tensor(pad, dtype=torch.bool).unsqueeze(0).to(device)
            with torch.no_grad():
                _, feat = model(text_description=tokens, padding_mask=pad)
            vec = feat[0].cpu().tolist()
        elif "embedding" in it:
            vec = [float(x) for x in it["embedding"]]
        else:
            raise ValueError("Each JSON item must include 'image', 'text', or 'embedding'")

        embeddings.append([float(x) for x in vec])
        if "domain" in it:
            domains.append(it["domain"])

    labels = domains if domains else None
    return embeddings, labels


def _dist_sq(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))

def kmeans_cluster(
    x: List[List[float]],
    k: int,
    n_iter: int = 20,
    seed: int = 0,
    return_centroids: bool = False,
) -> List[int] | Tuple[List[int], List[List[float]]]:
    """Simple k-means clustering implemented without external dependencies."""
    random.seed(seed)
    centroids = [x[i][:] for i in random.sample(range(len(x)), k)]
    for _ in range(n_iter):
        labels = [min(range(k), key=lambda j: _dist_sq(pt, centroids[j])) for pt in x]
        new_centroids = [centroids[j][:] for j in range(k)]
        for j in range(k):
            pts = [p for p, lab in zip(x, labels) if lab == j]
            if pts:
                dim = len(pts[0])
                new_centroids[j] = [sum(p[d] for p in pts) / len(pts) for d in range(dim)]
        if all(all(abs(a - b) < 1e-6 for a, b in zip(c_old, c_new)) for c_old, c_new in zip(centroids, new_centroids)):
            centroids = new_centroids
            break
        centroids = new_centroids
    return (labels, centroids) if return_centroids else labels


def assign_kmeans(x: List[List[float]], centroids: List[List[float]]) -> List[int]:
    """Assign points to pre-trained centroids."""
    k = len(centroids)
    return [min(range(k), key=lambda j: _dist_sq(pt, centroids[j])) for pt in x]


def save_kmeans(centroids: List[List[float]], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"centroids": centroids}, f)


def load_kmeans(path: str) -> List[List[float]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        centroids = data.get("centroids", data.get("centers"))
        if centroids is None:
            centroids = next(iter(data.values()))
    else:
        centroids = data
    return [[float(v) for v in c] for c in centroids]


def _entropy(labels: List) -> float:
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    total = len(labels)
    return -sum((c / total) * math.log(c / total + 1e-10) for c in counts.values())


def v_measure(labels_true: List, labels_pred: List[int]) -> float:
    """Compute V-measure between two clusterings."""
    n = len(labels_true)
    h_c = _entropy(labels_true)
    h_k = _entropy(labels_pred)

    h_c_k = 0.0
    for k_val in set(labels_pred):
        idxs = [i for i, l in enumerate(labels_pred) if l == k_val]
        sub = [labels_true[i] for i in idxs]
        h_c_k += len(idxs) / n * _entropy(sub)

    h_k_c = 0.0
    for c_val in set(labels_true):
        idxs = [i for i, l in enumerate(labels_true) if l == c_val]
        sub = [labels_pred[i] for i in idxs]
        h_k_c += len(idxs) / n * _entropy(sub)

    homogeneity = 1.0 if h_c == 0 else 1 - h_c_k / h_c
    completeness = 1.0 if h_k == 0 else 1 - h_k_c / h_k
    denom = homogeneity + completeness
    return 0.0 if denom == 0 else 2 * homogeneity * completeness / denom


def plot_umap(embeddings: List[List[float]], labels: Optional[list] = None, out_file: str = "umap.png") -> None:
    """Generate a UMAP scatter plot."""
    import umap
    import matplotlib.pyplot as plt
    import numpy as np

    reducer = umap.UMAP()
    pts = reducer.fit_transform(np.array(embeddings))
    plt.figure()
    if labels is None:
        plt.scatter(pts[:, 0], pts[:, 1], s=5)
    else:
        labels = np.asarray(labels)
        for l in np.unique(labels):
            idx = labels == l
            plt.scatter(pts[idx, 0], pts[idx, 1], s=5, label=str(l))
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_file)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UMAP visualization for JSON embeddings")
    p.add_argument("json_file", type=str, nargs="?", help="JSON lines file with embeddings")
    p.add_argument("--json-data", type=str, help="JSON lines file (alias of positional arg)")
    p.add_argument("--output", type=str, default="umap.png", help="Output image path")
    p.add_argument("--cluster-domains", type=int, metavar="k", default=None, help="Cluster embeddings into k groups")
    p.add_argument("--kmeans-model", type=str, default=None, help="Path to trained k-means .pth file")
    p.add_argument("--embedding-model", type=str, default=None, help="Trained MUSK .pth to compute embeddings")
    return p.parse_args()


def main() -> None:
    args = get_args()
    json_path = args.json_data or args.json_file
    if json_path is None:
        raise SystemExit("JSON file must be specified")

    embeddings, domains = collect_embeddings(json_path, args.embedding_model)
    colour_labels = None
    if args.kmeans_model is not None and os.path.exists(args.kmeans_model):
        centroids = load_kmeans(args.kmeans_model)
        colour_labels = assign_kmeans(embeddings, centroids)
        if domains is not None:
            score = v_measure(domains, colour_labels)
            print(f"V-measure: {score:.3f}")
    elif args.cluster_domains is not None:
        result = kmeans_cluster(embeddings, args.cluster_domains, return_centroids=True)
        colour_labels, centroids = result
        if args.kmeans_model:
            save_kmeans(centroids, args.kmeans_model)
        if domains is not None:
            score = v_measure(domains, colour_labels)
            print(f"V-measure: {score:.3f}")
    elif domains is not None:
        colour_labels = domains
    try:
        plot_umap(embeddings, colour_labels, args.output)
    except ImportError as e:
        raise RuntimeError("Plotting requires 'umap-learn' and 'matplotlib'") from e


if __name__ == "__main__":
    main()
