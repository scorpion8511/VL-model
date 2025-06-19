import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import torch.nn.functional as F
from accelerate import Accelerator

from musk.modeling import musk_large_patch16_384
from musk.mmadapter import MultiModalAdapter


def get_dummy_dataloader(batch_size: int = 8):
    images = torch.randn(64, 3, 224, 224)
    texts = torch.randint(0, 1000, (64, 32))
    masks = torch.zeros_like(texts)
    dataset = TensorDataset(images, texts, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    accelerator = Accelerator()
    model = musk_large_patch16_384()
    adapter = MultiModalAdapter(embed_dim=1024)
    model.mma = adapter

    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataloader = get_dummy_dataloader()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    for step, batch in enumerate(dataloader):
        images, texts, masks = batch
        optimizer.zero_grad()
        with accelerator.autocast():
            vision_feat, text_feat = model(image=images, text_description=texts, padding_mask=masks)
            loss = model.mma.compute_loss(vision_feat, text_feat, model.logit_scale)
        accelerator.backward(loss)
        optimizer.step()
        if step % 10 == 0:
            accelerator.print(f"Step {step}: loss {loss.item():.4f}")

        if step >= 20:
            break


if __name__ == "__main__":
    main()
