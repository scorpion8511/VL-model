import torch
from musk.torchscale.component.xmoe.domain_gate import DomainGate


def test_domain_gate_routes():
    gate = DomainGate(num_experts=2)
    x = torch.randn(4, 8)
    domains = torch.tensor([0, 0, 1, 1])
    l_aux, combine, dispatch, _ = gate(x, domain_ids=domains)
    # dispatch shape S,E,C; tokens routed to correct expert
    expert0 = dispatch[:, 0, :].any(dim=1)
    expert1 = dispatch[:, 1, :].any(dim=1)
    assert expert0[:2].all() and not expert0[2:].any()
    assert expert1[2:].all() and not expert1[:2].any()
