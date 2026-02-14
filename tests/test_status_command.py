"""Tests for the `dictate status` CLI command."""

import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


class TestStatusCommand:
    """Test the `dictate status` subcommand."""

    def test_status_exits_zero(self):
        """Status command returns 0."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "status"]):
            result = main()
        assert result == 0

    def test_status_shows_version(self, capsys):
        """Status output includes version."""
        from dictate.menubar_main import main
        from dictate import __version__

        with patch.object(sys, "argv", ["dictate", "status"]):
            main()

        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_status_shows_system_info(self, capsys):
        """Status output includes system information."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "status"]):
            main()

        captured = capsys.readouterr()
        assert "macOS" in captured.out or "System" in captured.out
        assert "Python" in captured.out

    def test_status_shows_models(self, capsys):
        """Status output includes model information."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "status"]):
            main()

        captured = capsys.readouterr()
        assert "Models" in captured.out
        assert "Whisper" in captured.out

    def test_status_shows_preferences(self, capsys):
        """Status output includes preferences."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "status"]):
            main()

        captured = capsys.readouterr()
        assert "Preferences" in captured.out

    def test_status_shows_running_state(self, capsys):
        """Status output shows whether Dictate is running."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "status"]):
            main()

        captured = capsys.readouterr()
        assert "Status" in captured.out
        # Should show either "Running" or "Not running"
        assert "Running" in captured.out or "running" in captured.out

    def test_help_includes_status(self, capsys):
        """Help text mentions the status command."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "--help"]):
            main()

        captured = capsys.readouterr()
        assert "status" in captured.out
