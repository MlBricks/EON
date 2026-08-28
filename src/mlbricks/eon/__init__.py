"""EON — Evolving Observational Network.

Canonical usage::

    from mlbricks import eon
    model = eon(...)

This package is installed as an extension of the MLBricks package.  The
``mlbricks.eon`` module is callable so the compact constructor works without
replacing MLBricks' own ``mlbricks/__init__.py``.
"""

from __future__ import annotations

import inspect
import sys
import types

from .core import EON

__all__ = ["EON"]
__version__ = "0.1.0a0"


class _CallableEONModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return EON(*args, **kwargs)


_module = sys.modules[__name__]
_module.__class__ = _CallableEONModule
_module.__signature__ = inspect.signature(EON)
