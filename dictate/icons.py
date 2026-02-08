"""Menu bar template icons for Dictate.

Generates monochrome waveform PNGs at 36x36 pixels / 144 DPI so macOS
renders them as crisp 18x18-point template images on Retina displays.

Includes 8 animation frames for a rippling waveform during recording.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import zlib

# 144 DPI → 5669 pixels-per-meter (PNG pHYs chunk)
_PPM_144DPI = 5669
_ICON_SIZE = 36
_BOTTOM_PAD = 2  # shifted up slightly vs original 4
_BAR_WIDTH = 3
_BAR_GAP = 3

# Static idle waveform
_IDLE_HEIGHTS = [9, 15, 20, 14, 8]

# Animation: 8 frames generated from a sine wave that ripples across bars
_N_ANIM_FRAMES = 8
_BASE_ACTIVE = [15, 21, 26, 20, 14]
_ANIM_AMPLITUDE = 5


def _make_waveform_grid(
    heights: list[int],
    bar_width: int = _BAR_WIDTH,
    gap: int = _BAR_GAP,
    size: int = _ICON_SIZE,
    bottom_pad: int = _BOTTOM_PAD,
) -> list[str]:
    grid = [["." for _ in range(size)] for _ in range(size)]

    num_bars = len(heights)
    total_w = num_bars * bar_width + (num_bars - 1) * gap
    start_x = (size - total_w) // 2
    bottom_y = size - 1 - bottom_pad

    for i, h in enumerate(heights):
        x0 = start_x + i * (bar_width + gap)
        top_y = bottom_y - h + 1

        for r in range(top_y, bottom_y + 1):
            if r == top_y and bar_width >= 3:
                for c in range(x0 + 1, x0 + bar_width - 1):
                    if 0 <= r < size and 0 <= c < size:
                        grid[r][c] = "X"
            else:
                for c in range(x0, x0 + bar_width):
                    if 0 <= r < size and 0 <= c < size:
                        grid[r][c] = "X"

    return ["".join(row) for row in grid]


def _make_anim_frames() -> dict[str, list[str]]:
    """Generate animation frames using a sine wave that ripples across bars."""
    frames: dict[str, list[str]] = {}
    for f in range(_N_ANIM_FRAMES):
        t = f * 2 * math.pi / _N_ANIM_FRAMES
        heights = []
        for b in range(5):
            phase = b * 2 * math.pi / 5
            offset = math.sin(t + phase) * _ANIM_AMPLITUDE
            h = max(4, min(30, int(_BASE_ACTIVE[b] + offset)))
            heights.append(h)
        frames[f"anim_{f}"] = _make_waveform_grid(heights)
    return frames


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = chunk_type + data
    crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return struct.pack(">I", len(data)) + c + crc


def _grid_to_png(grid: list[str]) -> bytes:
    height = len(grid)
    width = len(grid[0]) if grid else 0

    pixels = bytearray()
    for row in grid:
        for ch in row:
            if ch == "X":
                pixels.extend(b"\x00\x00\x00\xff")
            else:
                pixels.extend(b"\x00\x00\x00\x00")

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        offset = y * width * 4
        raw.extend(pixels[offset : offset + width * 4])

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    phys = _chunk(b"pHYs", struct.pack(">IIB", _PPM_144DPI, _PPM_144DPI, 1))
    idat = _chunk(b"IDAT", zlib.compress(bytes(raw)))
    iend = _chunk(b"IEND", b"")

    return sig + ihdr + phys + idat + iend


N_ANIM_FRAMES = _N_ANIM_FRAMES

_GRIDS: dict[str, list[str]] = {
    "idle": _make_waveform_grid(_IDLE_HEIGHTS),
    **_make_anim_frames(),
}

_icon_cache: dict[str, str] = {}

# Reactive icon: two alternating temp files so rumps always sees a new path
_reactive_paths: list[str] = []
_reactive_idx = 0


def generate_reactive_icon(heights: list[int]) -> str:
    """Generate a waveform icon from actual bar heights. Alternates temp files."""
    global _reactive_idx

    if len(_reactive_paths) < 2:
        for _ in range(2):
            tmp = tempfile.NamedTemporaryFile(
                prefix="dictate_reactive_", suffix=".png", delete=False
            )
            tmp.close()
            _reactive_paths.append(tmp.name)

    grid = _make_waveform_grid(heights)
    png_data = _grid_to_png(grid)

    _reactive_idx = 1 - _reactive_idx
    path = _reactive_paths[_reactive_idx]

    with open(path, "wb") as f:
        f.write(png_data)

    return path


def get_icon_path(name: str) -> str:
    """Return path to a template-icon PNG (created once, then cached)."""
    if name in _icon_cache:
        return _icon_cache[name]

    grid = _GRIDS.get(name)
    if grid is None:
        raise ValueError(f"Unknown icon: {name}")

    png_data = _grid_to_png(grid)

    tmp = tempfile.NamedTemporaryFile(
        prefix=f"dictate_{name}_",
        suffix=".png",
        delete=False,
    )
    tmp.write(png_data)
    tmp.close()

    _icon_cache[name] = tmp.name
    return tmp.name


def cleanup_temp_files() -> None:
    """Remove all temp icon PNGs created during this session."""
    for path in list(_icon_cache.values()):
        try:
            os.remove(path)
        except OSError:
            pass
    _icon_cache.clear()

    for path in _reactive_paths:
        try:
            os.remove(path)
        except OSError:
            pass
    _reactive_paths.clear()
