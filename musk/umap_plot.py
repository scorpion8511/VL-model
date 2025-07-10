import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score

try:
    from umap import UMAP
except Exception:
    from umap.umap_ import UMAP

from timm.models import create_model


class PTFeatDataset(Dataset):
    """Dataset for slide features stored as ``.pt`` files."""

    def __init__(self, slide_names, labels, feat_dir):
        self.slide_names = slide_names
        self.labels = labels
        self.feat_dir = feat_dir

    def __len__(self):
        return len(self.slide_names)

    def __getitem__(self, index):
        slide = self.slide_names[index]
        label = self.labels[index]
        feat_path = os.path.join(self.feat_dir, slide + ".pt")
        if not os.path.exists(feat_path):
            feat_path = os.path.join(self.feat_dir, os.path.splitext(slide)[0] + ".pt")
        data = torch.load(feat_path)
        if isinstance(data, dict):
            feat = data.get("feat") if "feat" in data else data.get("features", data)
        else:
            feat = data
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        return {"input": feat, "label": label, "slide": slide}


def load_model(checkpoint, arch=None):
    """Load a MUSK model checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    conf = ckpt.get("config", {}) or {}
    if arch is not None:
        conf["arch"] = arch
    arch_name = conf.get("arch", "musk_large_patch16_384")
    model = create_model(arch_name)
    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, device


def collect_embeddings(model, loader, device):
    """Collect embeddings by hooking the classifier layer."""
    features = []
    labels = []
    names = []
    gathered = []
    if hasattr(model, "Slide_classifier"):
        fc = model.Slide_classifier.fc
    else:
        fc = model.slide_classifier.fc if hasattr(model, "slide_classifier") else None
    if fc is None:
        raise AttributeError("Model must have a `slide_classifier.fc` layer")

    def hook(mod, inp, out):
        gathered.append(inp[0].detach().cpu())

    handle = fc.register_forward_hook(hook)
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, dtype=torch.float32)
            _ = model(x)
            names.append(batch["slide"])
            lab = batch["label"]
            if torch.is_tensor(lab):
                lab = lab.item()
            labels.append(lab)
            features.append(gathered.pop(0))
    handle.remove()
    feats = torch.cat(features).numpy()
    labels = np.array(labels).reshape(-1)
    names = [n for n in names]
    return names, labels, feats


def main():
    parser = argparse.ArgumentParser(description="UMAP of model embeddings")
    parser.add_argument("--csv_path", default="patient_label.csv", help="CSV with pathology_id and label")
    parser.add_argument("--feat_dir", required=True, help="directory with slide .pt files")
    parser.add_argument("--checkpoint", required=True, help="trained model checkpoint")
    parser.add_argument("--arch", default=None, help="model architecture if not in checkpoint")
    parser.add_argument("--output", default="umap_model.png", help="output image path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    slide_names = df["pathology_id"].tolist()
    labels = df["label"].tolist()

    dataset = PTFeatDataset(slide_names, labels, args.feat_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model, device = load_model(args.checkpoint, args.arch)

    names, labels, feats = collect_embeddings(model, loader, device)

    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(feats)
    v_score = v_measure_score(labels, kmeans.labels_)
    print(f"V-measure score: {v_score:.4f}")

    reducer = UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(feats)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    handles, _ = scatter.legend_elements()
    classes = sorted(np.unique(labels))
    plt.legend(handles, classes, title="Label")
    plt.title("UMAP of model slide embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
