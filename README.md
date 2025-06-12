
text captions. The script uses [HuggingFace Accelerate](https://github.com/huggingface/accelerate)
for device management, so run it with `accelerate launch` to enable multi‑GPU
training when available.
accelerate launch -m musk.pretrain \
accelerate launch -m musk.pretrain \
