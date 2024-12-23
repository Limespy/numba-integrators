# pylint: disable=duplicate-code
"""Numba integrators."""
from importlib import import_module
from sys import modules as _modules
from typing import TYPE_CHECKING

from ._API import *
# ======================================================================
# Hinting types
if TYPE_CHECKING:
    from types import ModuleType

    from . import first
    from . import second
else:
    ModuleType = object
# ======================================================================
__version__ = '0.4.1'
_SELF: ModuleType = _modules[__package__]
_DYNAMIC_MODULES = ('first', 'second')
# ----------------------------------------------------------------------
def __getattr__(name: str) -> ModuleType:
    if name not in _DYNAMIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f'.{name}', __package__)
    setattr(_SELF, name, module)
    return module
