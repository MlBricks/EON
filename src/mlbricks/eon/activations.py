from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    input_multiplier: int = 1


_REGISTRY: dict[str, ActivationSpec] = {}


def register_activation(
    name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    input_multiplier: int = 1,
    overwrite: bool = False,
) -> None:
    """Register a shape-aware activation.

    ``input_multiplier`` tells SAFFN how many state-width chunks the activation
    consumes. Most activations use 1. SwiGLU uses 2 and returns one chunk.
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("activation name cannot be empty")
    if input_multiplier < 1:
        raise ValueError("input_multiplier must be >= 1")
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"activation {key!r} is already registered")
    _REGISTRY[key] = ActivationSpec(key, fn, int(input_multiplier))


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


def get_activation(name: str) -> ActivationSpec:
    key = str(name).strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown activation {name!r}. Available: {choices}. "
            "Register custom activations with mlbricks.soup.register_activation()."
        ) from exc


def available_activations() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register_activation("silu", F.silu)
register_activation("gelu", F.gelu)
register_activation("relu", F.relu)
register_activation("relu2", lambda x: F.relu(x).square())
register_activation("mish", F.mish)
register_activation("leaky_relu", lambda x: F.leaky_relu(x, negative_slope=0.10))
register_activation("elu", lambda x: F.elu(x, alpha=1.0))
register_activation("hardswish", F.hardswish)
register_activation("swiglu", _swiglu, input_multiplier=2)
