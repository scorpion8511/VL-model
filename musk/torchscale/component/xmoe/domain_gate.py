import math
from typing import Optional, Tuple, Dict

import torch
from torch import Tensor

from .routing import one_hot
from .moe_layer import fused_cumsum_sub_one


class DomainGate(torch.nn.Module):
    """Gate routing tokens to a fixed expert based on domain labels."""

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts

    def forward(self, input: Tensor, mask: Optional[Tensor] = None, domain_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        assert domain_ids is not None, "Domain IDs required"
        num_tokens = input.size(0)
        device = input.device
        domain_ids = domain_ids.to(device).view(-1, 1)
        mask1 = one_hot(domain_ids, self.num_experts)
        if mask is not None:
            mask1 = mask1 * (~mask).unsqueeze(-1).to(mask1.dtype)
        locations1 = fused_cumsum_sub_one(mask1)
        capacity = math.ceil(num_tokens / self.num_experts)
        mask1 = mask1 * (locations1 < capacity)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        gates1_s = torch.ones(num_tokens, device=device)
        gates1 = gates1_s.unsqueeze(-1) * mask1
        locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
        combine1_sec = torch.bmm(gates1.unsqueeze(-1), locations1_sc.to(gates1.dtype).unsqueeze(1))
        dispatch_mask = combine1_sec.bool()
        l_aux = torch.tensor(0.0, device=device)
        metadata: Dict = {}
        return l_aux, combine1_sec, dispatch_mask, metadata
