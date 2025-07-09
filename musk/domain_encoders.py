import torch
import torch.nn as nn

class IdentityGate(nn.Module):
    """Simple gate that returns the domain indices as-is."""

    def forward(self, domains: torch.Tensor) -> torch.Tensor:
        return domains


class DomainEncoderManager(nn.Module):
    """Manage multiple encoders based on domain indices."""

    def __init__(self, encoders, gate=None):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.gate = gate or IdentityGate()

    def forward(self, images: torch.Tensor, domains: torch.Tensor):
        device = images.device
        idxs = self.gate(domains).to(device)
        batch_size = images.size(0)
        feat_dim = self.encoders[0](images[:1]).shape[1]
        outputs = torch.zeros(batch_size, feat_dim, device=device, dtype=images.dtype)
        for i, encoder in enumerate(self.encoders):
            mask = idxs == i
            if mask.any():
                outputs[mask] = encoder(images[mask])
        return outputs
