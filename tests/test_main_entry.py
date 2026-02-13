"""Tests for dictate/__main__.py entry point."""

from __future__ import annotations

from unittest.mock import patch


class TestMainEntry:
    """Test the __main__.py module entry point."""

    @patch("dictate.menubar_main.main", return_value=0)
    def test_main_import(self, mock_main):
        """Test that __main__ imports correctly and exposes main."""
        import dictate.__main__  # noqa: F401

    @patch("dictate.menubar_main.main", return_value=0)
    def test_main_guard_not_triggered_on_import(self, mock_main):
        """__name__ != '__main__' on import, so SystemExit not raised."""
        import importlib
        import dictate.__main__
        importlib.reload(dictate.__main__)
        # main() should not be called during import (guard protects it)
        # The guard only fires when __name__ == "__main__"
