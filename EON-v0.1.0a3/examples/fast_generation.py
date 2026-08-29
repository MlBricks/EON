from mlbricks import eon

model = eon(
    dim=512,
    width=32,
    depth=2,
    mixer="esa",
    ffn="saffn",
    backend="auto",
    precision="fp16",
)

# After moving to the final inference device:
model.eval().prepare_generation()
