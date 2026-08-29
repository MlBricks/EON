from __future__ import annotations

import sys
import types

from .core import EON, eon

__version__ = "0.1.0a3"
__all__ = ["EON", "eon"]


class _CallableModule(types.ModuleType):
    """Allow the intended API: ``from mlbricks import eon; eon(...)``."""

    def __call__(self, *args, **kwargs):
        return eon(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
