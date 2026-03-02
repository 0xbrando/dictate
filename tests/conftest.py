"""Shared pytest fixtures for Dictate tests."""

from unittest.mock import patch

import pytest

import dictate.mlx_check as _mlx_mod


@pytest.fixture(autouse=True)
def mock_mlx_available():
    """Auto-mock MLX availability check so tests don't run subprocess probes."""
    with patch("dictate.mlx_check.is_mlx_available", return_value=True):
        yield
    # Reset the module-level cache so it doesn't leak between tests
    _mlx_mod._mlx_available = None
