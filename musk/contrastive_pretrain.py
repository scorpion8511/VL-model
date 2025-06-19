import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import wandb

from musk.modeling import musk_large_patch16_384
from musk.mmadapter import MultiModalAdapter
from musk.json_dataset import JsonDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, help="Path to training JSON file")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    if args.wandb and accelerator.is_main_process:
        wandb.init(project="musk-mma", name="stage1")

    model = musk_large_patch16_384()
    adapter = MultiModalAdapter(embed_dim=1024)
    model.mma = adapter

    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataset = JsonDataset(args.train_json)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            images, texts, masks = batch
            optimizer.zero_grad()
            with accelerator.autocast():
                vision_feat, text_feat = model(image=images, text_description=texts, padding_mask=masks)
                loss = model.mma.compute_loss(vision_feat, text_feat, model.logit_scale)
            accelerator.backward(loss)
            optimizer.step()
            if accelerator.is_main_process and step % 10 == 0:
                accelerator.print(f"Epoch {epoch} Step {step}: loss {loss.item():.4f}")
                if args.wandb:
                    wandb.log({"loss": loss.item()}, step=global_step)
            global_step += 1


if __name__ == "__main__":
    main()
