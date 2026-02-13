"""Tests for CLI flags (--version, --help)."""

import sys
from unittest import mock

import pytest


class TestVersionFlag:
    """Test --version and -V flags."""

    @mock.patch("dictate.menubar_main.sys")
    def test_version_flag_long(self, mock_sys):
        mock_sys.argv = ["dictate", "--version"]
        # Import after patching
        from dictate.menubar_main import main
        with mock.patch("builtins.print") as mock_print:
            mock_sys.argv = ["dictate", "--version"]
            result = main()
        assert result == 0
        printed = mock_print.call_args[0][0]
        assert "dictate" in printed.lower()

    @mock.patch("dictate.menubar_main.sys")
    def test_version_flag_short(self, mock_sys):
        mock_sys.argv = ["dictate", "-V"]
        from dictate.menubar_main import main
        with mock.patch("builtins.print") as mock_print:
            mock_sys.argv = ["dictate", "-V"]
            result = main()
        assert result == 0
        printed = mock_print.call_args[0][0]
        assert "dictate" in printed.lower()


class TestHelpFlag:
    """Test --help and -h flags."""

    @mock.patch("dictate.menubar_main.sys")
    def test_help_flag_long(self, mock_sys):
        mock_sys.argv = ["dictate", "--help"]
        from dictate.menubar_main import main
        with mock.patch("builtins.print") as mock_print:
            mock_sys.argv = ["dictate", "--help"]
            result = main()
        assert result == 0
        # Should print multiple lines including usage
        assert mock_print.call_count >= 5

    @mock.patch("dictate.menubar_main.sys")
    def test_help_flag_short(self, mock_sys):
        mock_sys.argv = ["dictate", "-h"]
        from dictate.menubar_main import main
        with mock.patch("builtins.print") as mock_print:
            mock_sys.argv = ["dictate", "-h"]
            result = main()
        assert result == 0
        assert mock_print.call_count >= 5

    @mock.patch("dictate.menubar_main.sys")
    def test_help_contains_commands(self, mock_sys):
        mock_sys.argv = ["dictate", "--help"]
        from dictate.menubar_main import main
        outputs = []
        with mock.patch("builtins.print", side_effect=lambda *a, **kw: outputs.append(str(a[0]) if a else "")):
            mock_sys.argv = ["dictate", "--help"]
            result = main()
        all_text = " ".join(outputs)
        assert "update" in all_text.lower()
        assert "foreground" in all_text.lower()
        assert "github" in all_text.lower()
