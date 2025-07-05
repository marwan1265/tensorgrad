"""tensorGrad.backend

Exposes a single symbol `xp` – the active array backend.  By default this is
NumPy, but it can be switched to CuPy when desired.  All other tensorGrad code
must import and use `xp`; never import NumPy or CuPy directly.

Switching backend
-----------------
1. Environment variable (before import)
       TENSORGRAD_BACKEND=cupy python your_script.py
2. Runtime call
       from tensorGrad import backend
       backend.use_cupy()   # or backend.use_numpy()

If CuPy is requested but not installed, a clear ImportError is raised.
If CuPy is installed but can't initialize (no GPU), falls back to NumPy.
"""

from __future__ import annotations

import importlib
import os
import warnings
from types import ModuleType

import numpy as _np


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _load_cupy() -> ModuleType:
    """Import CuPy lazily and raise a helpful error if it is missing."""
    try:
        return importlib.import_module("cupy")
    except ModuleNotFoundError as e:
        raise ImportError(
            "CuPy backend requested but the 'cupy' package is not installed.\n"
            "Install a wheel compatible with your CUDA version, e.g.\n"
            "    pip install cupy-cuda12x\n"
            "or refer to CuPy's installation guide."
        ) from e


def _try_cupy_fallback() -> ModuleType:
    """Try to load CuPy, but fall back to NumPy if it can't initialize."""
    try:
        cupy = _load_cupy()
        # Test if CuPy can actually create arrays (i.e., CUDA is available)
        cupy.array([1.0])
        return cupy
    except Exception as e:
        warnings.warn(
            f"CuPy backend requested but failed to initialize: {e}. "
            "Falling back to NumPy backend.",
            RuntimeWarning,
            stacklevel=2
        )
        return _np


# -----------------------------------------------------------------------------
# Backend selection logic
# -----------------------------------------------------------------------------

_backend_choice = os.environ.get("TENSORGRAD_BACKEND", "numpy").lower()

if _backend_choice == "cupy":
    _xp = _try_cupy_fallback()
else:
    _xp = _np

# public symbol used by the rest of tensorGrad
xp = _xp  # type: ignore


# -----------------------------------------------------------------------------
# Public helper functions
# -----------------------------------------------------------------------------

def use_cupy() -> None:
    """Switch the active backend to CuPy at runtime (if installed)."""
    global xp
    xp = _try_cupy_fallback()  # type: ignore


def use_numpy() -> None:
    """Switch the active backend to NumPy at runtime."""
    global xp
    xp = _np  # type: ignore


def is_cupy() -> bool:
    """Return True if the current backend is CuPy."""
    return xp.__name__ == "cupy"  # type: ignore 