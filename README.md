# EON — Evolving Observational Network

EON is a history-aware MLBricks architecture that keeps an evolving state,
compares the current sequence context with prior context, and combines the
current mixer output with a state-aware FFN update.

EON is compositional: **the mixer and the state-aware FFN can be selected per
physical layer**.

## Install

EON extends the `mlbricks` package and requires MLBricks 1.0.0+.

```bash
pip install -e .
```

## Public API

```python
from mlbricks import eon

model = eon(
    dim=384,
    width=1024,
    depth=2,
    mixer=["esa", "bolt"],
    ffn=["saffn", "ffnbrick"],
    mixer_config=[
        {"head": 6, "compass": "auto"},
        {"head": 6, "latent_dim": 32},
    ],
    backend="auto",
    precision="fp16",
)
```

`eon(...)` is the canonical constructor. The package does not replace the main
MLBricks `__init__.py`; `mlbricks.eon` is a callable extension module.

## Constructor

```python
eon(
    dim=384,
    width=1024,
    depth=1,
    mixer="esa",
    ffn="saffn",
    activation="silu",
    mixer_config=None,
    ffn_config=None,
    backend="auto",
    precision="fp16",
)
```

The SOUP.Cell broadcasting convention is preserved:

- a scalar `width`, `mixer`, `ffn`, `activation`, or config is broadcast across every layer;
- a list supplies one value/configuration per layer;
- a list length must equal `depth`;
- learned state bridges are inserted when adjacent state widths differ.

### Mixed layers

```python
model = eon(
    dim=384,
    depth=4,
    width=[1024, 1024, 1536, 768],
    mixer=["esa", "bolt", "bolt", "esa"],
    ffn=["saffn", "ffnbrick", "ffnbrick", "saffn"],
    activation=["silu", "silu", "silu", "gelu"],
    mixer_config=[
        {"head": 6},
        {"head": 6, "latent_dim": 32},
        {"head": 6, "latent_dim": 48},
        {"head": 6, "compass": 16},
    ],
)
```

This means:

```text
Layer 1  width 1024  ESA   + SAFFN
Layer 2  width 1024  BOLT  + FFNBrick
Layer 3  width 1536  BOLT  + FFNBrick
Layer 4  width  768  ESA   + SAFFN
```

## Built-in mixers

### `mixer="esa"`

Uses current MLBricks `ESA`. Mixer-specific options belong in `mixer_config`:

```python
mixer_config={
    "head": 6,
    "batch": None,
    "block": None,
    "compass": "auto",
}
```

EON owns parent device placement, so its internal ESA layers use
`device=None`, `auto_move_input=False`, and `auto_compile=False`.

### `mixer="bolt"`

Uses current MLBricks `Bolt`:

```python
mixer_config={
    "head": 6,
    "latent_dim": 32,
}
```

`head` is normalized to Bolt's `num_heads` argument.

## Built-in state-aware FFNs

### `ffn="saffn"`

The validated state-aware FFN inherited from the former SOUP.Cell architecture.
It supports EON's `activation` setting and performs candidate/write/retain/read
state evolution from current context versus previous context.

### `ffn="ffnbrick"`

Uses current MLBricks `ffnbrick.StateAwareFFN`. It follows the same state-aware
four-input contract and currently uses its fixed SiLU value path, so the matching
EON activation must be `"silu"`.

## History

Normal execution:

```python
y = model(x)
```

History-aware/chunked execution:

```python
y1, history = model(x1, return_history=True)
y2, history = model(x2, history=history, return_history=True)
```

Treat `history` as opaque. Pass it back to EON rather than manually managing
state and previous mixer context.

## Layer operation

Each EON layer follows the former SOUP.Cell structure, generalized to swappable
components:

```text
input
  -> RMSNorm
  -> selected mixer (ESA / BOLT / custom)
  -> compare current context with previous context
  -> selected state-aware FFN evolves state
  -> learned scaling of (mixer context + state-aware FFN output)
  -> residual output
```

EON itself owns history flow, state-width bridges, and composition. Mixers and
FFNs remain replaceable components.

## Custom components

A custom mixer must be an `nn.Module` (or factory returning one) with:

```python
context = mixer(x)  # same [B,T,D] shape
```

A custom EON FFN must follow:

```python
output, next_state = ffn(x, current_context, previous_context, state)
```

where `output` has `[B,T,D]` shape and `next_state` has the configured EON
state width.

## Backends

EON follows the MLBricks backend contract:

```text
auto | native | pytorch
```

```python
model.set_backend("native")
print(model.resolved_backend())
```

Backend changes propagate to internal mixers/FFNs that expose MLBricks'
`set_backend()` interface.

## Inspection

```python
print(model.parameter_count)
print(model.to_config())
```

## License

Copyright © 2026 **Zameer Hussain and Akhtar Hussain**.
All rights reserved except as expressly licensed below.

EON is licensed under the **PolyForm Noncommercial License 1.0.0**.
Commercial use requires a separate written commercial license.

Commercial licensing inquiries: `licensing@mlbricks.io`

See [`LICENSE`](LICENSE) for the complete notices and ownership terms.
