import json
from pathlib import Path

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from musk.umap_json import collect_embeddings, kmeans_cluster


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
    labels = kmeans_cluster(emb, 2)
    assert len(labels) == 6
