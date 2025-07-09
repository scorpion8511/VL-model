import torch
import torch.nn as nn

class CrossAttentionDecoder(nn.Module):
    """Single-layer transformer decoder used for cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None):
        return self.decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_mask)


class CaptionDecoder(nn.Module):
    """Decode text tokens from image features using cross-attention."""

    def __init__(self, embed_dim: int, vocab_size: int, num_heads: int = 8):
        super().__init__()
        self.cross = CrossAttentionDecoder(embed_dim, num_heads)
        self.to_vocab = nn.Linear(embed_dim, vocab_size)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor, text_mask: torch.Tensor | None = None):
        x = self.cross(text_emb, image_emb, text_mask)
        return self.to_vocab(x)


class PatchDecoder(nn.Module):
    """Decode image patches from text embeddings."""

    def __init__(self, embed_dim: int, patch_dim: int, num_patches: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_patches, embed_dim))
        self.cross = CrossAttentionDecoder(embed_dim, num_heads)
        self.to_patch = nn.Linear(embed_dim, patch_dim)

    def forward(self, text_emb: torch.Tensor):
        b = text_emb.size(0)
        query = self.query.unsqueeze(0).expand(b, -1, -1)
        x = self.cross(query, text_emb)
        return self.to_patch(x)
