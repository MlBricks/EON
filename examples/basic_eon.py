import torch
from mlbricks import eon

model = eon(
    dim=384,
    width=[1024, 1536],
    depth=2,
    mixer=["esa", "bolt"],
    ffn=["saffn", "ffnbrick"],
    mixer_config=[
        {"head": 6, "compass": "auto"},
        {"head": 6, "latent_dim": 32},
    ],
    backend="auto",
)

x = torch.randn(2, 128, 384)
y, history = model(x, return_history=True)
print(y.shape)

x2 = torch.randn(2, 128, 384)
y2, history = model(x2, history=history, return_history=True)
print(y2.shape)
