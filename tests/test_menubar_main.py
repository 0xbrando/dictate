"""Tests for dictate.menubar_main — singleton lock, daemonize, logging, main()."""

import errno
import fcntl
import logging
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from dictate.menubar_main import (
    LOCK_FILE,
    _acquire_singleton_lock,
    _daemonize,
    main,
    setup_logging,
)


# ── _acquire_singleton_lock ────────────────────────────────────────

class TestAcquireSingletonLock:
    """Covers _acquire_singleton_lock: success, contention, OS errors."""

    @patch("dictate.menubar_main.os.lseek", return_value=5)
    @patch("dictate.menubar_main.os.ftruncate")
    @patch("dictate.menubar_main.os.write")
    @patch("dictate.menubar_main.fcntl.flock")
    @patch("dictate.menubar_main.os.open", return_value=42)
    @patch("dictate.menubar_main.LOCK_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_success(self, mock_path, mock_os_open, mock_flock, mock_write, mock_trunc, mock_lseek):
        mock_path.parent.mkdir = MagicMock()
        fd = _acquire_singleton_lock()
        assert fd == 42
        mock_flock.assert_called_once_with(42, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mock_write.assert_called_once()

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main.fcntl.flock", side_effect=OSError(errno.EAGAIN, "locked"))
    @patch("dictate.menubar_main.os.open", return_value=7)
    @patch("dictate.menubar_main.LOCK_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_already_locked_eagain(self, mock_path, mock_os_open, mock_flock, mock_close):
        mock_path.parent.mkdir = MagicMock()
        assert _acquire_singleton_lock() is None
        mock_close.assert_called_once_with(7)

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main.fcntl.flock", side_effect=OSError(errno.EWOULDBLOCK, "would block"))
    @patch("dictate.menubar_main.os.open", return_value=8)
    @patch("dictate.menubar_main.LOCK_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_already_locked_ewouldblock(self, mock_path, mock_os_open, mock_flock, mock_close):
        mock_path.parent.mkdir = MagicMock()
        assert _acquire_singleton_lock() is None

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main.fcntl.flock", side_effect=OSError(errno.EACCES, "access denied"))
    @patch("dictate.menubar_main.os.open", return_value=9)
    @patch("dictate.menubar_main.LOCK_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_already_locked_eacces(self, mock_path, mock_os_open, mock_flock, mock_close):
        mock_path.parent.mkdir = MagicMock()
        assert _acquire_singleton_lock() is None

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main.fcntl.flock", side_effect=OSError(errno.EIO, "io error"))
    @patch("dictate.menubar_main.os.open", return_value=10)
    @patch("dictate.menubar_main.LOCK_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_unexpected_os_error(self, mock_path, mock_os_open, mock_flock, mock_close, capsys):
        mock_path.parent.mkdir = MagicMock()
        assert _acquire_singleton_lock() is None
        mock_close.assert_called_once_with(10)


# ── _daemonize ─────────────────────────────────────────────────────

class TestDaemonize:
    """Covers _daemonize: fork paths and error handling."""

    @patch("dictate.menubar_main.signal.signal")
    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main.os.dup2")
    @patch("dictate.menubar_main.os.open", return_value=5)
    @patch("dictate.menubar_main.os.setsid")
    @patch("dictate.menubar_main.os.fork", return_value=0)
    @patch("dictate.menubar_main.LOG_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_child_process(self, mock_log_path, mock_fork, mock_setsid, mock_os_open,
                           mock_dup2, mock_close, mock_signal):
        mock_log_path.parent.mkdir = MagicMock()
        _daemonize()
        mock_fork.assert_called_once()
        mock_setsid.assert_called_once()
        mock_signal.assert_called_once_with(signal.SIGHUP, signal.SIG_IGN)
        # dup2 called 3 times for stdin, stdout, stderr
        assert mock_dup2.call_count == 3

    @patch("dictate.menubar_main.os._exit")
    @patch("dictate.menubar_main.os.fork", return_value=123)
    def test_parent_exits(self, mock_fork, mock_exit):
        _daemonize()
        mock_exit.assert_called_once_with(0)

    @patch("dictate.menubar_main.os.fork", side_effect=OSError("nope"))
    def test_fork_fails(self, mock_fork, capsys):
        _daemonize()  # Should not raise — prints warning and returns
        captured = capsys.readouterr()
        assert "Fork failed" in captured.err

    @patch("dictate.menubar_main.signal.signal")
    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main.os.dup2")
    @patch("dictate.menubar_main.os.open", side_effect=[OSError("log fail"), 5, 6])
    @patch("dictate.menubar_main.os.setsid")
    @patch("dictate.menubar_main.os.fork", return_value=0)
    @patch("dictate.menubar_main.LOG_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_log_open_fails_falls_back_to_devnull(self, mock_log_path, mock_fork,
                                                    mock_setsid, mock_os_open,
                                                    mock_dup2, mock_close, mock_signal):
        """If log file can't be opened, fallback to /dev/null."""
        mock_log_path.parent.mkdir = MagicMock()
        _daemonize()
        # os.open: first fail (log), then devnull for log_fd fallback, then devnull for stdin
        assert mock_os_open.call_count == 3


# ── setup_logging ──────────────────────────────────────────────────

class TestSetupLogging:
    def test_configures_root_logger(self):
        # Reset root logger first
        root = logging.getLogger()
        old_level = root.level
        old_handlers = root.handlers[:]
        try:
            root.handlers.clear()
            root.setLevel(logging.WARNING)
            setup_logging()
            assert root.level == logging.INFO
        finally:
            root.setLevel(old_level)
            root.handlers = old_handlers

    def test_silences_noisy_libraries(self):
        setup_logging()
        for lib in ("urllib3", "httpx", "mlx", "transformers", "tokenizers", "sounddevice"):
            assert logging.getLogger(lib).level == logging.ERROR


# ── main() ─────────────────────────────────────────────────────────

class TestMain:
    """Covers main() entry point flows."""

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=None)
    @patch("dictate.menubar_main._daemonize")
    def test_already_running_returns_1(self, mock_daemon, mock_lock, mock_close):
        with patch.object(sys, "argv", ["dictate", "--foreground"]):
            result = main()
        assert result == 1

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar.DictateMenuBarApp", side_effect=KeyboardInterrupt, create=True)
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=99)
    @patch("dictate.menubar_main._daemonize")
    def test_keyboard_interrupt_returns_130(self, mock_daemon, mock_lock, mock_app, mock_close):
        with patch.object(sys, "argv", ["dictate", "--foreground"]):
            result = main()
        assert result == 130
        mock_close.assert_called_once_with(99)

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar.DictateMenuBarApp", side_effect=RuntimeError("boom"), create=True)
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=50)
    @patch("dictate.menubar_main._daemonize")
    def test_fatal_error_returns_1(self, mock_daemon, mock_lock, mock_app, mock_close):
        with patch.object(sys, "argv", ["dictate", "--foreground"]):
            result = main()
        assert result == 1
        mock_close.assert_called_once_with(50)

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=33)
    @patch("dictate.menubar_main._daemonize")
    def test_success_returns_0(self, mock_daemon, mock_lock, mock_close):
        mock_app = MagicMock()
        with (
            patch.object(sys, "argv", ["dictate", "--foreground"]),
            patch("dictate.menubar.DictateMenuBarApp", return_value=mock_app, create=True),
        ):
            result = main()
        assert result == 0
        mock_app.start_app.assert_called_once()
        mock_close.assert_called_once_with(33)

    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=33)
    @patch("dictate.menubar_main.os.close")
    def test_foreground_flag_skips_daemonize(self, mock_close, mock_lock):
        mock_app = MagicMock()
        with (
            patch.object(sys, "argv", ["dictate", "--foreground"]),
            patch("dictate.menubar.DictateMenuBarApp", return_value=mock_app, create=True),
        ):
            result = main()
        assert result == 0

    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=33)
    @patch("dictate.menubar_main.os.close")
    def test_f_flag_also_skips_daemonize(self, mock_close, mock_lock):
        mock_app = MagicMock()
        with (
            patch.object(sys, "argv", ["dictate", "-f"]),
            patch("dictate.menubar.DictateMenuBarApp", return_value=mock_app, create=True),
        ):
            result = main()
        assert result == 0
