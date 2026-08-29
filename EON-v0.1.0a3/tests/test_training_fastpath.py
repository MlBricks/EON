import torch
from mlbricks import eon


class BombIdentity(torch.nn.Module):
    def forward(self, x):
        raise AssertionError("registered Identity bridge executed on uniform fast path")


def test_uniform_training_path_skips_identity_bridge_and_backpropagates():
    torch.manual_seed(11)
    model = eon(dim=16, width=24, depth=2, mixer="esa", ffn="saffn", mixer_config={"head": 4}, backend="pytorch", precision="fp32")
    model.state_bridges[0] = BombIdentity()
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_validate_owns_shape_checks():
    model = eon(dim=16, width=24, depth=2, mixer="esa", ffn="saffn", mixer_config={"head": 4}, backend="pytorch", precision="fp32")
    good = torch.randn(2, 5, 16)
    report = model.validate(good)
    assert report["ok"] is True
    assert report["execution_path"] == "uniform"
    try:
        model.validate(torch.randn(2, 5, 15))
    except ValueError:
        pass
    else:
        raise AssertionError("validate() must reject the wrong hidden dimension")
