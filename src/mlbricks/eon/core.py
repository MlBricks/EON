from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .activations import get_activation


_BACKENDS = {"auto", "native", "pytorch"}


class ConfigurationError(ValueError):
    """Raised when an EON architecture configuration is inconsistent."""


def _normalize_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in _BACKENDS:
        raise ConfigurationError(
            f"backend must be one of {sorted(_BACKENDS)}, got {value!r}"
        )
    return backend


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


class _StateAwareFFN(nn.Module):
    """Validated SOUP state-aware FFN retained as EON's built-in ``saffn``.

    It compares the current mixer context with the previous context, updates a
    token-wise state, and emits a state-conditioned feature update.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int,
        *,
        depth_dim: int = 64,
        activation: str = "silu",
    ):
        super().__init__()
        self.dim = int(dim)
        self.state_dim = int(state_dim)
        self.activation_name = str(activation)
        self.activation = get_activation(self.activation_name)

        s = self.state_dim
        value_chunks = self.activation.input_multiplier
        self.x_proj = nn.Linear(dim, (2 + value_chunks) * s, bias=True)
        self.context_candidate = nn.Linear(dim, s, bias=False)
        self.context_write = nn.Linear(dim, s, bias=False)
        self.state_proj = nn.Linear(s, 2 * s, bias=False)
        self.output = nn.Linear(s, dim, bias=False)
        self.depth_embedding = nn.Parameter(torch.empty(depth_dim))
        self.depth_proj = nn.Linear(depth_dim, (2 + value_chunks) * s, bias=False)
        self.retain_logit = nn.Parameter(torch.full((s,), 1.15))
        self.read_logit = nn.Parameter(torch.zeros(s))
        self.candidate_transition_logit = nn.Parameter(torch.tensor(-2.0))
        self.write_transition_logit = nn.Parameter(torch.tensor(-2.0))
        self.retain_delta_scale = nn.Parameter(torch.full((s,), 0.10))
        self.read_delta_scale = nn.Parameter(torch.full((s,), 0.10))
        self.delta_magnitude_log_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.normal_(self.depth_embedding, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        current_context: torch.Tensor,
        previous_context: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.state_dim
        value_chunks = self.activation.input_multiplier

        xparts = self.x_proj(x).split(s, dim=-1)
        xc, xw = xparts[0], xparts[1]
        xv = torch.cat(xparts[2 : 2 + value_chunks], dim=-1)

        sc, sw = self.state_proj(state).split(s, dim=-1)
        dparts = self.depth_proj(self.depth_embedding).split(s, dim=-1)
        dc, dw = dparts[0], dparts[1]
        dv = torch.cat(dparts[2 : 2 + value_chunks], dim=-1)

        delta = current_context - previous_context
        dm = torch.sqrt(
            delta.float().square().mean(-1, keepdim=True) + 1e-6
        ).to(current_context.dtype)
        scaled = torch.exp(self.delta_magnitude_log_scale) * dm

        candidate_scale = torch.sigmoid(self.candidate_transition_logit)
        write_scale = torch.sigmoid(self.write_transition_logit)
        candidate_context = current_context + candidate_scale * delta
        write_context = current_context + write_scale * delta

        candidate = torch.tanh(
            xc + self.context_candidate(candidate_context) + sc + dc
        )
        write = torch.sigmoid(xw + self.context_write(write_context) + sw + dw)
        retain = torch.sigmoid(
            self.retain_logit - scaled * self.retain_delta_scale
        )
        next_state = (1.0 - write) * (retain * state) + write * candidate

        value = self.activation.fn(xv + dv)
        read = torch.sigmoid(self.read_logit + scaled * self.read_delta_scale)
        out = self.output(next_state * value * read)
        return out, next_state


@dataclass(frozen=True)
class _EONHistory:
    """Opaque runtime history returned by EON.

    Applications should pass this object back through ``history=...`` rather
    than depending on its internal representation.
    """

    _state: torch.Tensor
    _context: torch.Tensor

    def detach(self) -> "_EONHistory":
        return _EONHistory(self._state.detach(), self._context.detach())

    def to(self, *args, **kwargs) -> "_EONHistory":
        return _EONHistory(
            self._state.to(*args, **kwargs),
            self._context.to(*args, **kwargs),
        )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalize_width(width: int | Sequence[int], depth: int) -> tuple[int, ...]:
    if isinstance(width, int):
        if width <= 0:
            raise ConfigurationError("width must be > 0")
        return (int(width),) * depth
    values = tuple(int(v) for v in width)
    if len(values) != depth:
        raise ConfigurationError(
            f"width list must have one value per depth: depth={depth}, got {len(values)}"
        )
    if any(v <= 0 for v in values):
        raise ConfigurationError("all widths must be > 0")
    return values


def _normalize_strings(
    value: str | Sequence[str], depth: int, *, name: str
) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) * depth
    values = tuple(str(v) for v in value)
    if len(values) != depth:
        raise ConfigurationError(
            f"{name} list must have one value per depth: depth={depth}, got {len(values)}"
        )
    return values


def _normalize_components(value: Any, depth: int, *, name: str) -> tuple[Any, ...]:
    if _is_sequence(value):
        values = tuple(value)
        if len(values) != depth:
            raise ConfigurationError(
                f"{name} list must have one value per depth: depth={depth}, got {len(values)}"
            )
        return values
    return (value,) * depth


def _normalize_configs(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None,
    depth: int,
    *,
    name: str,
) -> tuple[dict[str, Any], ...]:
    if value is None:
        return tuple({} for _ in range(depth))
    if isinstance(value, Mapping):
        return tuple(copy.deepcopy(dict(value)) for _ in range(depth))
    values = tuple(value)
    if len(values) != depth:
        raise ConfigurationError(
            f"{name} list must have one value per depth: depth={depth}, got {len(values)}"
        )
    out: list[dict[str, Any]] = []
    for item in values:
        if item is None:
            out.append({})
        elif isinstance(item, Mapping):
            out.append(copy.deepcopy(dict(item)))
        else:
            raise ConfigurationError(f"each {name} entry must be a mapping or None")
    return tuple(out)


def _filter_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _clone_module(module: nn.Module) -> nn.Module:
    try:
        return copy.deepcopy(module)
    except Exception as exc:  # pragma: no cover - unusual custom extension modules
        raise ConfigurationError(
            "a module instance broadcast across multiple EON layers must be deepcopy-able; "
            "pass one module instance per layer instead"
        ) from exc


def _custom_component(spec: Any, kwargs: dict[str, Any], *, kind: str) -> nn.Module:
    if isinstance(spec, nn.Module):
        if kwargs:
            raise ConfigurationError(
                f"{kind}_config cannot be used with a pre-built nn.Module instance"
            )
        return spec
    if callable(spec):
        try:
            module = spec(**_filter_kwargs(spec, kwargs))
        except TypeError as exc:
            raise ConfigurationError(f"could not construct custom {kind}: {exc}") from exc
        if not isinstance(module, nn.Module):
            raise ConfigurationError(f"custom {kind} factory must return torch.nn.Module")
        return module
    raise ConfigurationError(
        f"{kind} must be a supported name, nn.Module, or module factory; got {type(spec).__name__}"
    )


def _build_mixer(
    spec: Any,
    *,
    dim: int,
    config: dict[str, Any],
    backend: str,
    precision: str,
) -> tuple[nn.Module, str]:
    if not isinstance(spec, str):
        return _custom_component(spec, config, kind="mixer"), spec.__class__.__name__ if isinstance(spec, nn.Module) else getattr(spec, "__name__", "custom")

    name = spec.strip().lower()
    cfg = dict(config)
    if name == "esa":
        from ..esa import ESA

        reserved = {"embd", "device", "auto_move_input", "auto_compile"}
        conflict = reserved.intersection(cfg)
        if conflict:
            raise ConfigurationError(
                f"ESA mixer_config cannot override EON-managed fields: {sorted(conflict)}"
            )
        head = int(cfg.pop("head", 6))
        kwargs = {
            "embd": dim,
            "head": head,
            "batch": cfg.pop("batch", None),
            "block": cfg.pop("block", None),
            "backend": cfg.pop("backend", backend),
            "precision": cfg.pop("precision", precision),
            "compass": cfg.pop("compass", "auto"),
            "device": None,
            "auto_move_input": False,
            "auto_compile": False,
            **cfg,
        }
        return ESA(**kwargs), "esa"

    if name == "bolt":
        from ..bolt import Bolt

        if "d_model" in cfg:
            raise ConfigurationError("BOLT mixer_config cannot override EON-managed d_model")
        if "head" in cfg and "num_heads" in cfg:
            raise ConfigurationError("BOLT mixer_config must use only one of head or num_heads")
        num_heads = int(cfg.pop("head", cfg.pop("num_heads", 6)))
        kwargs = {
            "d_model": dim,
            "num_heads": num_heads,
            "backend": cfg.pop("backend", backend),
            **cfg,
        }
        return Bolt(**kwargs), "bolt"

    raise ConfigurationError(
        f"unknown mixer {spec!r}; supported built-ins are 'esa' and 'bolt'"
    )


def _build_ffn(
    spec: Any,
    *,
    dim: int,
    width: int,
    activation: str,
    layer_index: int,
    total_layers: int,
    config: dict[str, Any],
    backend: str,
) -> tuple[nn.Module, str]:
    if not isinstance(spec, str):
        return _custom_component(spec, config, kind="ffn"), spec.__class__.__name__ if isinstance(spec, nn.Module) else getattr(spec, "__name__", "custom")

    name = spec.strip().lower()
    cfg = dict(config)
    if name == "saffn":
        depth_dim = int(cfg.pop("depth_dim", cfg.pop("depth_embedding_dim", 64)))
        if cfg:
            raise ConfigurationError(
                f"unsupported saffn ffn_config keys: {sorted(cfg)}"
            )
        return _StateAwareFFN(
            dim,
            width,
            depth_dim=depth_dim,
            activation=activation,
        ), "saffn"

    if name == "ffnbrick":
        from ..ffnbrick import StateAwareFFN

        if activation.strip().lower() != "silu":
            raise ConfigurationError(
                "MLBricks FFNBrick StateAwareFFN uses its fixed SiLU value path; "
                "set activation='silu' for ffn='ffnbrick'"
            )
        reserved = {"d_model", "state_dim", "layer_index", "total_layers"}
        conflict = reserved.intersection(cfg)
        if conflict:
            raise ConfigurationError(
                f"ffnbrick ffn_config cannot override EON-managed fields: {sorted(conflict)}"
            )
        kwargs = {
            "d_model": dim,
            "state_dim": width,
            "layer_index": layer_index,
            "total_layers": total_layers,
            "backend": cfg.pop("backend", backend),
            **cfg,
        }
        return StateAwareFFN(**kwargs), "ffnbrick"

    raise ConfigurationError(
        f"unknown ffn {spec!r}; supported built-ins are 'saffn' and 'ffnbrick'"
    )


class EONLayer(nn.Module):
    """One EON history-comparison layer with swappable mixer and FFN."""

    def __init__(
        self,
        *,
        dim: int,
        width: int,
        mixer: Any,
        ffn: Any,
        activation: str,
        mixer_config: dict[str, Any],
        ffn_config: dict[str, Any],
        layer_index: int,
        total_layers: int,
        backend: str,
        precision: str,
    ):
        super().__init__()
        self.dim = int(dim)
        self.width = int(width)
        self.activation = str(activation)
        self.norm = RMSNorm(self.dim)

        self.mixer, self.mixer_name = _build_mixer(
            mixer,
            dim=self.dim,
            config=mixer_config,
            backend=backend,
            precision=precision,
        )
        self.ffn, self.ffn_name = _build_ffn(
            ffn,
            dim=self.dim,
            width=self.width,
            activation=self.activation,
            layer_index=layer_index,
            total_layers=total_layers,
            config=ffn_config,
            backend=backend,
        )

        # Preserves the validated SOUP residual equation while generalizing the
        # two branches: current mixer context + state-aware FFN output.
        self.mix = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: torch.Tensor | None,
        previous_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.norm(x)
        context = self.mixer(z)
        if not isinstance(context, torch.Tensor) or context.shape != x.shape:
            got = getattr(context, "shape", None)
            raise RuntimeError(
                f"EON mixer {self.mixer_name!r} must return a tensor with shape {tuple(x.shape)}, got {got}"
            )

        if state is None:
            state = torch.zeros(
                *x.shape[:-1], self.width, device=x.device, dtype=x.dtype
            )
        if state.shape[:-1] != x.shape[:-1] or state.shape[-1] != self.width:
            raise ConfigurationError(
                f"EON layer expected state shape [..., {self.width}] matching input prefix, "
                f"got {tuple(state.shape)} for input {tuple(x.shape)}"
            )
        if previous_context is None:
            previous_context = torch.zeros_like(x)
        if previous_context.shape != x.shape:
            raise ConfigurationError(
                f"previous EON context must have shape {tuple(x.shape)}, got {tuple(previous_context.shape)}"
            )

        result = self.ffn(z, context, previous_context, state)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                f"EON FFN {self.ffn_name!r} must return (output, next_state)"
            )
        ffn_out, next_state = result
        if not isinstance(ffn_out, torch.Tensor) or ffn_out.shape != x.shape:
            got = getattr(ffn_out, "shape", None)
            raise RuntimeError(
                f"EON FFN {self.ffn_name!r} output must have shape {tuple(x.shape)}, got {got}"
            )
        if (
            not isinstance(next_state, torch.Tensor)
            or next_state.shape[:-1] != x.shape[:-1]
            or next_state.shape[-1] != self.width
        ):
            got = getattr(next_state, "shape", None)
            raise RuntimeError(
                f"EON FFN {self.ffn_name!r} next_state must have width {self.width}, got {got}"
            )

        delta = torch.sigmoid(self.mix) * (context + ffn_out)
        return x + delta, next_state, context


def _describe_component(spec: Any) -> str:
    if isinstance(spec, str):
        return spec.strip().lower()
    if isinstance(spec, nn.Module):
        return spec.__class__.__name__
    return getattr(spec, "__name__", spec.__class__.__name__)


class EON(nn.Module):
    """EON — Evolving Observational Network.

    EON stores a state/history representation, compares the current mixer
    context with the prior context, and lets a state-aware FFN evolve the state.
    Mixer and FFN selections can be scalar (broadcast to all layers) or lists
    with one selection per physical layer.
    """

    def __init__(
        self,
        *,
        dim: int = 384,
        width: int | Sequence[int] = 1024,
        depth: int = 1,
        mixer: Any = "esa",
        ffn: Any = "saffn",
        activation: str | Sequence[str] = "silu",
        mixer_config: Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None = None,
        ffn_config: Mapping[str, Any] | Sequence[Mapping[str, Any] | None] | None = None,
        backend: str = "auto",
        precision: str = "fp16",
    ):
        super().__init__()
        if int(dim) <= 0:
            raise ConfigurationError("dim must be > 0")
        if int(depth) < 1:
            raise ConfigurationError("depth must be >= 1")

        self.dim = int(dim)
        self.depth = int(depth)
        self.backend = _normalize_backend(backend)
        self.precision = str(precision).lower()
        self.widths = _normalize_width(width, self.depth)
        self.activations = _normalize_strings(activation, self.depth, name="activation")
        for activation_name in self.activations:
            get_activation(activation_name)

        mixer_specs = _normalize_components(mixer, self.depth, name="mixer")
        ffn_specs = _normalize_components(ffn, self.depth, name="ffn")
        mixer_cfgs = _normalize_configs(mixer_config, self.depth, name="mixer_config")
        ffn_cfgs = _normalize_configs(ffn_config, self.depth, name="ffn_config")

        # A single pre-built module passed as a scalar is cloned so broadcasting
        # preserves the SOUP.Cell convention of independent physical layers.
        if self.depth > 1 and isinstance(mixer, nn.Module):
            mixer_specs = tuple(_clone_module(mixer) for _ in range(self.depth))
        if self.depth > 1 and isinstance(ffn, nn.Module):
            ffn_specs = tuple(_clone_module(ffn) for _ in range(self.depth))

        self.layers = nn.ModuleList([
            EONLayer(
                dim=self.dim,
                width=self.widths[i],
                mixer=mixer_specs[i],
                ffn=ffn_specs[i],
                activation=self.activations[i],
                mixer_config=mixer_cfgs[i],
                ffn_config=ffn_cfgs[i],
                layer_index=i,
                total_layers=self.depth,
                backend=self.backend,
                precision=self.precision,
            )
            for i in range(self.depth)
        ])

        self.state_bridges = nn.ModuleList([
            nn.Identity()
            if self.widths[i] == self.widths[i + 1]
            else nn.Linear(self.widths[i], self.widths[i + 1], bias=False)
            for i in range(self.depth - 1)
        ])
        self.cycle_bridge = (
            nn.Identity()
            if self.widths[-1] == self.widths[0]
            else nn.Linear(self.widths[-1], self.widths[0], bias=False)
        )

        self._mixer_specs = tuple(_describe_component(v) for v in mixer_specs)
        self._ffn_specs = tuple(_describe_component(v) for v in ffn_specs)
        self._mixer_configs = tuple(copy.deepcopy(v) for v in mixer_cfgs)
        self._ffn_configs = tuple(copy.deepcopy(v) for v in ffn_cfgs)

    def forward(
        self,
        x: torch.Tensor,
        *,
        history: _EONHistory | None = None,
        return_history: bool = False,
    ):
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"EON input must have shape [B,T,{self.dim}], got {tuple(x.shape)}"
            )

        h = x
        state: torch.Tensor | None = None
        previous_context: torch.Tensor | None = None
        if history is not None:
            if not isinstance(history, _EONHistory):
                raise TypeError("history must be the object returned by EON")
            state = history._state
            previous_context = history._context
            if state.shape[-1] != self.widths[0]:
                if state.shape[-1] == self.widths[-1]:
                    state = self.cycle_bridge(state)
                else:
                    raise ConfigurationError(
                        f"history state width must be {self.widths[0]} or {self.widths[-1]}, "
                        f"got {state.shape[-1]}"
                    )

        for i, layer in enumerate(self.layers):
            h, state, previous_context = layer(
                h,
                state=state,
                previous_context=previous_context,
            )
            if i < self.depth - 1:
                state = self.state_bridges[i](state)

        if return_history:
            assert state is not None and previous_context is not None
            return h, _EONHistory(state, previous_context)
        return h

    def set_backend(self, backend: str, *, recursive: bool = True):
        value = _normalize_backend(backend)
        self.backend = value
        for layer in self.layers:
            for module in (layer.mixer, layer.ffn):
                setter = getattr(module, "set_backend", None)
                if callable(setter):
                    try:
                        setter(value, recursive=recursive)
                    except TypeError:
                        setter(value)
                elif hasattr(module, "backend"):
                    try:
                        module.backend = value
                    except Exception:
                        pass
        return self

    def resolved_backend(self) -> str:
        values: list[str] = []
        for layer in self.layers:
            for module in (layer.mixer, layer.ffn):
                resolver = getattr(module, "resolved_backend", None)
                if callable(resolver):
                    try:
                        values.append(str(resolver()))
                    except Exception:
                        values.append("unavailable")
        if not values:
            return self.backend
        return values[0] if len(set(values)) == 1 else "mixed"

    def to_config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "width": list(self.widths),
            "depth": self.depth,
            "mixer": list(self._mixer_specs),
            "ffn": list(self._ffn_specs),
            "activation": list(self.activations),
            "mixer_config": copy.deepcopy(list(self._mixer_configs)),
            "ffn_config": copy.deepcopy(list(self._ffn_configs)),
            "backend": self.backend,
            "precision": self.precision,
        }

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, depth={self.depth}, widths={list(self.widths)}, "
            f"mixers={list(self._mixer_specs)}, ffns={list(self._ffn_specs)}"
        )
