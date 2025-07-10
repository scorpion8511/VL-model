import torch
from musk.decoders import CaptionDecoder, PatchDecoder


def test_caption_decoder_shape():
    B, T, I, D, V = 2, 5, 3, 4, 7
    text = torch.randn(B, T, D)
    img = torch.randn(B, I, D)
    dec = CaptionDecoder(D, V)
    out = dec(text, img)
    assert out.shape == (B, T, V)


def test_patch_decoder_shape():
    B, T, D, P, PD = 2, 4, 6, 3, 8
    text = torch.randn(B, T, D)
    dec = PatchDecoder(D, PD, P)
    out = dec(text)
    assert out.shape == (B, P, PD)


def test_losses_decrease():
    torch.manual_seed(0)
    B, T, I, D, V, P, PD = 4, 6, 5, 8, 10, 3, 12
    cap_dec = CaptionDecoder(D, V)
    rec_dec = PatchDecoder(D, PD, P)
    opt = torch.optim.Adam(list(cap_dec.parameters()) + list(rec_dec.parameters()), lr=0.01)
    losses = []
    for _ in range(20):
        text = torch.randn(B, T, D)
        img = torch.randn(B, I, D)
        tokens = torch.randint(0, V, (B, T))
        patches = torch.randn(B, P, PD)
        logits = cap_dec(text, img)
        loss_cap = torch.nn.functional.cross_entropy(logits.view(-1, V), tokens.view(-1))
        rec_pred = rec_dec(text)
        loss_rec = torch.nn.functional.mse_loss(rec_pred, patches)
        loss = loss_cap + loss_rec
        losses.append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert losses[-1] < losses[0]
