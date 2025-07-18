import argparse
import torch

from .json_dataset import get_json_loader
from timm.models import create_model

# ensure MUSK models are registered
from . import modeling


def load_model_and_head(model_ckpt: str, head_ckpt: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model("musk_large_patch16_384")
    state = torch.load(model_ckpt, map_location='cpu')
    model.load_state_dict(state, strict=False)
    embed_dim = model.beit3.args.encoder_embed_dim

    hd = torch.load(head_ckpt, map_location='cpu')
    domains = hd.get('domains')
    head = torch.nn.Linear(embed_dim, len(domains))
    state = hd['state_dict']
    if 'module.weight' in state:
        state = {k.replace('module.', ''): v for k, v in state.items()}
    head.load_state_dict(state)

    model.to(device).eval()
    head.to(device).eval()
    return model, head, domains, device


def main():
    p = argparse.ArgumentParser(description="Predict image domains")
    p.add_argument('--images', required=True, help='JSON lines dataset with image paths and domain labels')
    p.add_argument('--model', required=True, help='Stage-one model checkpoint')
    p.add_argument('--head', required=True, help='Domain head checkpoint')
    args = p.parse_args()

    loader = get_json_loader(args.images, mode='image', batch_size=1, num_workers=2, tokenizer=None, return_domain=True)
    model, head, domains, device = load_model_and_head(args.model, args.head)

    correct = 0
    total = 0
    with torch.no_grad():
        for img, dom in loader:
            img = img.to(device)
            # model returns only a tuple of (vision_cls, language_cls) when
            # return_seq=False and only images are provided
            feat, _ = model(
                image=img,
                with_head=False,
                out_norm=False,
                return_global=True,
                return_seq=False,
            )
            pred = head(feat).argmax(dim=-1).item()
            if dom[0] == domains[pred]:
                correct += 1
            total += 1
    acc = 100 * correct / total if total else 0
    print(f'Accuracy: {acc:.2f}% ({correct}/{total})')


if __name__ == '__main__':
    main()
