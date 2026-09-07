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


@pytest.fixture(autouse=True)
def mock_accessibility_permission():
    """Output unit tests must not depend on the test runner's macOS permissions."""
    with patch("dictate.output._accessibility_trusted", return_value=True):
        yield


@pytest.fixture(autouse=True)
def isolate_app_logging(tmp_path):
    """Tests that initialize the app must not write mock failures into real logs."""
    import logging

    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level
    with patch("dictate.menubar_main.LOG_FILE", tmp_path / "logs" / "dictate.log"):
        yield
    for handler in root.handlers[:]:
        if handler not in previous_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(previous_level)
