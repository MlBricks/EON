import inspect

import pytest
import torch
import torch.nn as nn


def test_public_callable_api():
    from mlbricks import eon
    from mlbricks.eon import EON

    assert callable(eon)
    model = eon(
        dim=16,
        width=24,
        depth=1,
        mixer="esa",
        ffn="saffn",
        mixer_config={"head": 4},
        backend="pytorch",
    )
    assert isinstance(model, EON)
    sig = inspect.signature(eon)
    assert list(sig.parameters) == [
        "dim", "width", "depth", "mixer", "ffn", "activation",
        "mixer_config", "ffn_config", "backend", "precision",
    ]


def test_scalar_broadcast_and_history():
    from mlbricks import eon

    model = eon(
        dim=16,
        width=24,
        depth=2,
        mixer="esa",
        ffn="saffn",
        mixer_config={"head": 4},
        backend="pytorch",
    )
    assert model.widths == (24, 24)
    assert model.to_config()["mixer"] == ["esa", "esa"]
    x = torch.randn(2, 5, 16)
    y, history = model(x, return_history=True)
    assert y.shape == x.shape
    y2, history2 = model(x, history=history, return_history=True)
    assert y2.shape == x.shape
    assert type(history2) is type(history)


def test_mixed_esa_bolt_and_ffns():
    from mlbricks import eon

    model = eon(
        dim=16,
        width=[24, 12],
        depth=2,
        mixer=["esa", "bolt"],
        ffn=["saffn", "ffnbrick"],
        activation=["gelu", "silu"],
        mixer_config=[
            {"head": 4, "compass": "auto"},
            {"head": 4, "latent_dim": 4},
        ],
        backend="pytorch",
        precision="fp32",
    )
    x = torch.randn(2, 4, 16)
    y, history = model(x, return_history=True)
    assert y.shape == x.shape
    assert model.layers[0].mixer_name == "esa"
    assert model.layers[1].mixer_name == "bolt"
    assert model.layers[0].ffn_name == "saffn"
    assert model.layers[1].ffn_name == "ffnbrick"
    assert history._state.shape[-1] == 12


def test_per_layer_length_validation():
    from mlbricks import eon
    from mlbricks.eon.core import ConfigurationError

    with pytest.raises(ConfigurationError):
        eon(
            dim=16,
            depth=3,
            mixer=["esa", "bolt"],
            mixer_config={"head": 4},
        )


def test_ffnbrick_rejects_non_silu():
    from mlbricks import eon
    from mlbricks.eon.core import ConfigurationError

    with pytest.raises(ConfigurationError):
        eon(
            dim=16,
            width=24,
            ffn="ffnbrick",
            activation="gelu",
            mixer_config={"head": 4},
            backend="pytorch",
        )


def test_backend_propagates():
    from mlbricks import eon

    model = eon(
        dim=16,
        width=24,
        depth=2,
        mixer=["esa", "bolt"],
        ffn=["saffn", "ffnbrick"],
        mixer_config=[{"head": 4}, {"head": 4, "latent_dim": 4}],
        backend="pytorch",
    )
    assert model.set_backend("auto") is model
    assert model.backend == "auto"
    assert model.layers[0].mixer.backend == "auto"
    assert model.layers[1].mixer.backend == "auto"
    assert model.layers[1].ffn.backend == "auto"


class CustomMixer(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return self.scale * x


class CustomFFN(nn.Module):
    def __init__(self, dim=16, width=24):
        super().__init__()
        self.proj = nn.Linear(width, dim, bias=False)

    def forward(self, x, current_context, previous_context, state):
        del previous_context
        next_state = state + current_context.mean(-1, keepdim=True)
        return self.proj(next_state), next_state


def test_custom_components():
    from mlbricks import eon

    model = eon(
        dim=16,
        width=24,
        mixer=lambda scale=0.25: CustomMixer(scale),
        mixer_config={"scale": 0.25},
        ffn=lambda: CustomFFN(16, 24),
        backend="pytorch",
    )
    x = torch.randn(2, 3, 16)
    assert model(x).shape == x.shape
