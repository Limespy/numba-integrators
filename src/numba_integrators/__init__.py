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
# ----------------------------------------------------------------------
def __getattr__(name: str) -> ModuleType:
    if name in {'first', 'second'}:
        module = import_module(f'.{name}', __package__)
        setattr(_modules[__package__], name, module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# ======================================================================
def main(*_):
    print(f'Numba Integrators version {__version__}')