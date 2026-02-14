"""Tests for TyperOutput clipboard fallback path (output.py lines 45-51).

When clipboard (pyperclip) is unavailable, TyperOutput falls back to direct
keyboard typing. If that also fails, it raises RuntimeError.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyperclip
import pytest

from dictate.output import TyperOutput


class TestTyperOutputClipboardFallback:
    """Tests for the PyperclipException fallback to direct typing."""

    @patch("dictate.output.KeyboardController")
    def test_falls_back_to_direct_typing_on_clipboard_error(self, mock_ctrl_cls):
        """When clipboard raises PyperclipException, falls back to controller.type()."""
        mock_ctrl = MagicMock()
        mock_ctrl_cls.return_value = mock_ctrl

        handler = TyperOutput()

        with patch("dictate.output.pyperclip") as mock_pyperclip:
            # Make pyperclip.copy raise the REAL PyperclipException
            mock_pyperclip.PyperclipException = pyperclip.PyperclipException
            mock_pyperclip.copy.side_effect = pyperclip.PyperclipException("no clipboard")

            handler.output("hello fallback")

        # Should have called controller.type() as fallback
        mock_ctrl.type.assert_called_once_with("hello fallback")

    @patch("dictate.output.KeyboardController")
    def test_raises_runtime_error_when_both_clipboard_and_typing_fail(self, mock_ctrl_cls):
        """When clipboard AND direct typing both fail, raises RuntimeError."""
        mock_ctrl = MagicMock()
        mock_ctrl.type.side_effect = OSError("keyboard locked")
        mock_ctrl_cls.return_value = mock_ctrl

        handler = TyperOutput()

        with patch("dictate.output.pyperclip") as mock_pyperclip:
            mock_pyperclip.PyperclipException = pyperclip.PyperclipException
            mock_pyperclip.copy.side_effect = pyperclip.PyperclipException("no clipboard")

            with pytest.raises(RuntimeError, match="Could not paste or type text"):
                handler.output("will fail")

    @patch("dictate.output.KeyboardController")
    def test_fallback_typing_succeeds_with_empty_string(self, mock_ctrl_cls):
        """Fallback typing works with empty text."""
        mock_ctrl = MagicMock()
        mock_ctrl_cls.return_value = mock_ctrl

        handler = TyperOutput()

        with patch("dictate.output.pyperclip") as mock_pyperclip:
            mock_pyperclip.PyperclipException = pyperclip.PyperclipException
            mock_pyperclip.copy.side_effect = pyperclip.PyperclipException("no clipboard")

            handler.output("")

        mock_ctrl.type.assert_called_once_with("")

    @patch("dictate.output.KeyboardController")
    def test_fallback_typing_succeeds_with_unicode(self, mock_ctrl_cls):
        """Fallback typing works with unicode text."""
        mock_ctrl = MagicMock()
        mock_ctrl_cls.return_value = mock_ctrl

        handler = TyperOutput()

        with patch("dictate.output.pyperclip") as mock_pyperclip:
            mock_pyperclip.PyperclipException = pyperclip.PyperclipException
            mock_pyperclip.copy.side_effect = pyperclip.PyperclipException("no clipboard")

            handler.output("日本語テスト 🎉")

        mock_ctrl.type.assert_called_once_with("日本語テスト 🎉")
