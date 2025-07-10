import torch
from torch.utils.data import DataLoader
from musk.umap_plot import PTFeatDataset, collect_embeddings

class DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slide_classifier = DummyClassifier()
    def forward(self, x):
        return self.slide_classifier(x)

def test_collect_embeddings(tmp_path):
    data1 = torch.randn(4)
    data2 = torch.randn(4)
    torch.save(data1, tmp_path/'a.pt')
    torch.save(data2, tmp_path/'b.pt')
    dataset = PTFeatDataset(['a', 'b'], [0, 1], str(tmp_path))
    loader = DataLoader(dataset, batch_size=1)
    model = DummyNet()
    names, labels, feats = collect_embeddings(model, loader, torch.device('cpu'))
    assert names == ['a', 'b']
    assert labels.tolist() == [0, 1]
    assert feats.shape == (2, 4)
