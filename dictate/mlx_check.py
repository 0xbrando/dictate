"""Pre-flight check for MLX availability.

MLX immediately initializes Metal GPU on import via mlx::core::metal::Device::Device().
On macOS 26.3+ this can crash with SIGABRT (empty device list) — a native crash that
Python try/except cannot catch.

Instead of spawning a subprocess (which triggers macOS CrashReporter dialogs), we call
Apple's MTLCreateSystemDefaultDevice() directly via ctypes.  This API returns nil safely
when no Metal device is available — no crash, no dialog.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import logging

logger = logging.getLogger(__name__)

_mlx_available: bool | None = None  # cached after first check


def _metal_gpu_available() -> bool:
    """Check if a Metal GPU device exists using Apple's C API (no MLX import)."""
    try:
        metal_path = ctypes.util.find_library("Metal")
        if not metal_path:
            return False
        metal = ctypes.cdll.LoadLibrary(metal_path)
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        device = metal.MTLCreateSystemDefaultDevice()
        return device is not None and device != 0
    except (OSError, AttributeError):
        return False


def is_mlx_available() -> bool:
    """Return True if MLX can safely initialize on this system.

    Checks:
      1. ``mlx`` package is installed (pure Python, no native code)
      2. Metal GPU is reachable via ``MTLCreateSystemDefaultDevice()``

    The result is cached for the lifetime of the process.
    """
    global _mlx_available
    if _mlx_available is not None:
        return _mlx_available

    # 1. Is the mlx package even installed?
    if importlib.util.find_spec("mlx") is None:
        _mlx_available = False
        logger.info("MLX package is not installed")
        return False

    # 2. Can Metal find a GPU device?
    if not _metal_gpu_available():
        _mlx_available = False
        logger.warning(
            "Metal GPU not available (MTLCreateSystemDefaultDevice returned nil). "
            "MLX features disabled — use ANE/raw dictation or a localhost LLM server."
        )
        return False

    _mlx_available = True
    logger.debug("MLX pre-flight check passed (Metal GPU found)")
    return True
