import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalAdapter(nn.Module):
    """Simple multi-modal adapter used for MUSK training.

    The adapter receives image and text embeddings and produces fused
    representations that are used to compute the contrastive loss.
    The design follows the implementation from
    `VLM-MultiModalAdapter` but is simplified for this repository.
    """

    def __init__(self, embed_dim: int, adapter_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.image_proj = nn.Linear(embed_dim, adapter_dim)
        self.text_proj = nn.Linear(embed_dim, adapter_dim)
        self.fuse = nn.Sequential(
            nn.Linear(adapter_dim * 2, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, embed_dim),
        )

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        if image_feat is None or text_feat is None:
            raise ValueError("Both image and text features must be provided")
        img = self.image_proj(image_feat)
        txt = self.text_proj(text_feat)
        fused = torch.cat([img, txt], dim=-1)
        return self.fuse(fused)

    def compute_loss(self, image_feat: torch.Tensor, text_feat: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        fused = self.forward(image_feat, text_feat)
        logits = logit_scale.exp() * fused @ fused.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2
