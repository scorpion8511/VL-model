import json
from pathlib import Path

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from musk.umap_json import (
    collect_embeddings,
    kmeans_cluster,
    assign_kmeans,
    save_kmeans,
    load_kmeans,
    apply_domain_map,
    map_clusters_by_majority,
)


def make_dummy_json(tmp_path: Path, n: int = 4) -> Path:
    path = tmp_path / "data.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            item = {
                "embedding": [float(i), float(i + 1)],
                "domain": "a" if i % 2 == 0 else "b",
            }
            f.write(json.dumps(item) + "\n")
    return path


def test_cluster_labels(tmp_path: Path):
    json_path = make_dummy_json(tmp_path, n=6)
    emb, _ = collect_embeddings(json_path)
    labels, centroids = kmeans_cluster(emb, 2, return_centroids=True)
    assert len(labels) == 6
    model_path = tmp_path / "model.pth"
    save_kmeans(centroids, model_path)
    loaded = load_kmeans(model_path)
    assigned = assign_kmeans(emb, loaded)
    assert assigned == labels


def test_apply_domain_map():
    labels = [0, 1, "2", "x"]
    mapping = {"0": "foo", "1": "bar", "2": "baz"}
    mapped = apply_domain_map(labels, mapping)
    assert mapped == ["foo", "bar", "baz", "x"]


def test_majority_mapping(tmp_path: Path):
    json_path = make_dummy_json(tmp_path, n=4)
    emb, domains = collect_embeddings(json_path)
    labels, _ = kmeans_cluster(emb, 2, seed=0, return_centroids=True)
    mapped = map_clusters_by_majority(domains, labels)
    assert len(mapped) == 4
    assert set(mapped).issubset({"a", "b"})
