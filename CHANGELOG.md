# Changelog

## 0.1.0a0

- Introduced **EON — Evolving Observational Network**.
- Renamed the former standalone SOUP-style state/history architecture to EON.
- Preserved the validated SOUP.Cell residual/state-flow structure.
- Added per-layer mixer selection with built-in `esa` and `bolt` adapters.
- Added per-layer state-aware FFN selection with built-in `saffn` and `ffnbrick` adapters.
- Added scalar-or-list broadcasting for width, mixer, FFN, activation, and component configs.
- Added learned state-width bridges between heterogeneous layers.
- Replaced exposed `state`/`previous_esa` runtime arguments with opaque `history`.
- Added MLBricks `auto | native | pytorch` backend propagation.
- Updated licensing and notices for EON.
