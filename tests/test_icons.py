"""Tests for dictate.icons — waveform icon generation."""

from __future__ import annotations

import os
import struct

import pytest

from dictate.icons import (
    N_ANIM_FRAMES,
    _grid_to_png,
    _make_waveform_grid,
    cleanup_temp_files,
    generate_reactive_icon,
    get_icon_path,
)


class TestWaveformGrid:
    def test_grid_dimensions(self):
        grid = _make_waveform_grid([10, 15, 20, 15, 10])
        assert len(grid) == 36  # default size
        assert all(len(row) == 36 for row in grid)

    def test_grid_contains_bars(self):
        grid = _make_waveform_grid([10, 15, 20, 15, 10])
        flat = "".join(grid)
        assert "X" in flat, "Grid should contain bar pixels"
        assert "." in flat, "Grid should contain empty pixels"

    def test_taller_bars_have_more_pixels(self):
        short_grid = _make_waveform_grid([5, 5, 5, 5, 5])
        tall_grid = _make_waveform_grid([25, 25, 25, 25, 25])
        short_count = sum(row.count("X") for row in short_grid)
        tall_count = sum(row.count("X") for row in tall_grid)
        assert tall_count > short_count

    def test_custom_size(self):
        grid = _make_waveform_grid([10, 10, 10], size=48)
        assert len(grid) == 48
        assert all(len(row) == 48 for row in grid)


class TestGridToPng:
    def test_produces_valid_png_signature(self):
        grid = _make_waveform_grid([10, 15, 20, 15, 10])
        png = _grid_to_png(grid)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_png_not_empty(self):
        grid = _make_waveform_grid([10, 15, 20, 15, 10])
        png = _grid_to_png(grid)
        assert len(png) > 100  # A minimal PNG is at least ~67 bytes


class TestGetIconPath:
    def test_idle_icon_exists(self):
        path = get_icon_path("idle")
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_anim_icons_exist(self):
        for i in range(N_ANIM_FRAMES):
            path = get_icon_path(f"anim_{i}")
            assert os.path.exists(path)

    def test_unknown_icon_raises(self):
        with pytest.raises(ValueError, match="Unknown icon"):
            get_icon_path("nonexistent_icon_name")

    def test_icons_are_cached(self):
        path1 = get_icon_path("idle")
        path2 = get_icon_path("idle")
        assert path1 == path2


class TestReactiveIcon:
    def test_generates_file(self):
        path = generate_reactive_icon([10, 15, 20, 15, 10])
        assert os.path.exists(path)
        assert path.endswith(".png")
        # Read and verify PNG signature
        with open(path, "rb") as f:
            sig = f.read(8)
        assert sig == b"\x89PNG\r\n\x1a\n"

    def test_alternates_paths(self):
        path1 = generate_reactive_icon([10, 15, 20, 15, 10])
        path2 = generate_reactive_icon([15, 20, 25, 20, 15])
        # Should alternate between two temp files
        assert path1 != path2
        path3 = generate_reactive_icon([10, 15, 20, 15, 10])
        assert path3 == path1  # Back to first path
