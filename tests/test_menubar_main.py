"""Tests for dictate.menubar_main — singleton lock, daemonize, logging, main(), _run_update()."""

import errno
import fcntl
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from dictate.menubar_main import (
    LOCK_FILE,
    LOG_FILE,
    _acquire_singleton_lock,
    _daemonize,
    _run_update,
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
    def test_unexpected_os_error(self, mock_path, mock_os_open, mock_flock, mock_close):
        mock_path.parent.mkdir = MagicMock()
        assert _acquire_singleton_lock() is None
        mock_close.assert_called_once_with(10)


# ── _daemonize ─────────────────────────────────────────────────────


class TestDaemonize:
    """Covers _daemonize: subprocess.Popen launch + error handling."""

    @patch("dictate.menubar_main.os._exit")
    @patch("dictate.menubar_main.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("dictate.menubar_main.LOG_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_success_launches_subprocess_and_exits(self, mock_log_path, mock_file,
                                                     mock_popen, mock_exit):
        mock_log_path.parent.mkdir = MagicMock()
        _daemonize()
        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["start_new_session"] is True
        assert call_kwargs["stdin"] == subprocess.DEVNULL
        mock_exit.assert_called_once_with(0)

    @patch("dictate.menubar_main.os._exit")
    @patch("dictate.menubar_main.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("dictate.menubar_main.LOG_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_passes_foreground_flag(self, mock_log_path, mock_file, mock_popen, mock_exit):
        mock_log_path.parent.mkdir = MagicMock()
        _daemonize()
        call_args = mock_popen.call_args[0][0]
        assert "--foreground" in call_args

    @patch("dictate.menubar_main.subprocess.Popen", side_effect=OSError("spawn failed"))
    @patch("builtins.open", new_callable=mock_open)
    @patch("dictate.menubar_main.LOG_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_oserror_falls_back_to_foreground(self, mock_log_path, mock_file, mock_popen, capsys):
        mock_log_path.parent.mkdir = MagicMock()
        _daemonize()  # Should NOT call os._exit, returns instead
        captured = capsys.readouterr()
        assert "foreground" in captured.err

    @patch("builtins.open", side_effect=OSError("log dir missing"))
    @patch("dictate.menubar_main.LOG_FILE", new_callable=lambda: MagicMock(spec=Path))
    def test_log_open_failure(self, mock_log_path, mock_file):
        mock_log_path.parent.mkdir = MagicMock()
        with pytest.raises(OSError, match="log dir missing"):
            _daemonize()


# ── _run_update ────────────────────────────────────────────────────


class TestRunUpdate:
    """Covers _run_update: pip upgrade, GitHub fallback, check-only, restart."""

    @patch("dictate.menubar_main.subprocess.Popen")
    @patch("dictate.menubar_main.time.sleep")
    @patch("dictate.menubar_main.subprocess.run")
    def test_pypi_success(self, mock_run, mock_sleep, mock_popen):
        """PyPI install succeeds → update + version check + restart."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),   # pip install (PyPI)
            MagicMock(returncode=0, stdout="2.5.0\n", stderr=""),  # version check
            MagicMock(returncode=0, stdout="12345\n", stderr=""),  # pgrep
            MagicMock(returncode=0, stdout="", stderr=""),   # pkill
        ]
        result = _run_update()
        assert result == 0

    @patch("dictate.menubar_main.subprocess.Popen")
    @patch("dictate.menubar_main.time.sleep")
    @patch("dictate.menubar_main.subprocess.run")
    def test_pypi_fails_github_fallback(self, mock_run, mock_sleep, mock_popen):
        """PyPI fails → falls back to GitHub install."""
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="not found"),  # pip install (PyPI) fails
            MagicMock(returncode=0, stdout="", stderr=""),   # pip install (GitHub) succeeds
            MagicMock(returncode=0, stdout="2.5.0\n", stderr=""),  # version check
            MagicMock(returncode=0, stdout="\n", stderr=""),  # pgrep (no running instance)
        ]
        result = _run_update()
        assert result == 0
        # Should have tried pip twice (PyPI then GitHub)
        calls = mock_run.call_args_list
        # First call: pip install --upgrade dictate-mlx
        assert "dictate-mlx" in str(calls[0])
        # Second call: pip install from git+https://github.com/
        assert "git+" in str(calls[1])

    @patch("dictate.menubar_main.subprocess.run")
    def test_both_fail(self, mock_run):
        """Both PyPI and GitHub fail → returns 1."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = _run_update()
        assert result == 1

    @patch("dictate.menubar_main.subprocess.run", side_effect=Exception("pip broken"))
    def test_exception_returns_1(self, mock_run):
        result = _run_update()
        assert result == 1

    @patch("dictate.menubar_main.subprocess.run")
    def test_from_github_flag(self, mock_run):
        """--github flag skips PyPI, goes straight to GitHub."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),   # pip install (GitHub)
            MagicMock(returncode=0, stdout="2.5.0\n", stderr=""),  # version check
            MagicMock(returncode=0, stdout="\n", stderr=""),  # pgrep (none running)
        ]
        result = _run_update(from_github=True)
        assert result == 0
        # First call should be git+ URL, NOT dictate-mlx
        first_call = str(mock_run.call_args_list[0])
        assert "git+" in first_call

    @patch("dictate.menubar_main.subprocess.run")
    def test_check_only(self, mock_run, capsys):
        """--check only checks PyPI, doesn't install."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Available versions: 2.5.0, 2.4.1\n",
            stderr="",
        )
        result = _run_update(check_only=True)
        assert result == 0
        captured = capsys.readouterr()
        assert "update" in captured.out.lower()

    @patch("dictate.menubar_main.subprocess.run")
    def test_check_only_not_on_pypi(self, mock_run, capsys):
        """--check when not on PyPI shows GitHub instructions."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        result = _run_update(check_only=True)
        assert result == 0
        captured = capsys.readouterr()
        assert "github" in captured.out.lower()

    @patch("dictate.menubar_main.subprocess.Popen")
    @patch("dictate.menubar_main.time.sleep")
    @patch("dictate.menubar_main.subprocess.run")
    def test_already_up_to_date(self, mock_run, mock_sleep, mock_popen, capsys):
        """When installed version matches current → shows 'already up to date'."""
        from dictate import __version__
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),   # pip install
            MagicMock(returncode=0, stdout=f"{__version__}\n", stderr=""),  # version check (same)
            MagicMock(returncode=0, stdout="\n", stderr=""),  # pgrep (none)
        ]
        result = _run_update()
        assert result == 0
        captured = capsys.readouterr()
        assert "up to date" in captured.out.lower()

    @patch("dictate.menubar_main.subprocess.Popen")
    @patch("dictate.menubar_main.time.sleep")
    @patch("dictate.menubar_main.subprocess.run")
    def test_no_running_instance_no_restart(self, mock_run, mock_sleep, mock_popen):
        """When no instance running, doesn't try to restart."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),   # pip install
            MagicMock(returncode=0, stdout="2.5.0\n", stderr=""),  # version check
            MagicMock(returncode=0, stdout="\n", stderr=""),  # pgrep (empty = none running)
        ]
        result = _run_update()
        assert result == 0
        mock_popen.assert_not_called()  # No restart needed


# ── setup_logging ──────────────────────────────────────────────────


class TestSetupLogging:
    def test_configures_root_logger(self):
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
    def test_already_running_returns_1(self, mock_lock, mock_close):
        with patch.object(sys, "argv", ["dictate", "--foreground"]):
            result = main()
        assert result == 1

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar.DictateMenuBarApp", side_effect=KeyboardInterrupt, create=True)
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=99)
    def test_keyboard_interrupt_returns_130(self, mock_lock, mock_app, mock_close):
        with patch.object(sys, "argv", ["dictate", "--foreground"]):
            result = main()
        assert result == 130
        mock_close.assert_called_once_with(99)

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar.DictateMenuBarApp", side_effect=RuntimeError("boom"), create=True)
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=50)
    def test_fatal_error_returns_1(self, mock_lock, mock_app, mock_close):
        with patch.object(sys, "argv", ["dictate", "--foreground"]):
            result = main()
        assert result == 1
        mock_close.assert_called_once_with(50)

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=33)
    def test_success_returns_0(self, mock_lock, mock_close):
        mock_app = MagicMock()
        with (
            patch.object(sys, "argv", ["dictate", "--foreground"]),
            patch("dictate.menubar.DictateMenuBarApp", return_value=mock_app, create=True),
        ):
            result = main()
        assert result == 0
        mock_app.start_app.assert_called_once()
        mock_close.assert_called_once_with(33)

    @patch("dictate.menubar_main._run_update", return_value=0)
    def test_update_flag_runs_update(self, mock_update):
        with patch.object(sys, "argv", ["dictate", "update"]):
            result = main()
        assert result == 0
        mock_update.assert_called_once_with(check_only=False, from_github=False)

    @patch("dictate.menubar_main._run_update", return_value=0)
    def test_dash_update_flag_runs_update(self, mock_update):
        with patch.object(sys, "argv", ["dictate", "--update"]):
            result = main()
        assert result == 0
        mock_update.assert_called_once_with(check_only=False, from_github=False)

    @patch("dictate.menubar_main._run_update", return_value=0)
    def test_update_check_flag(self, mock_update):
        with patch.object(sys, "argv", ["dictate", "update", "--check"]):
            result = main()
        assert result == 0
        mock_update.assert_called_once_with(check_only=True, from_github=False)

    @patch("dictate.menubar_main._run_update", return_value=0)
    def test_update_github_flag(self, mock_update):
        with patch.object(sys, "argv", ["dictate", "update", "--github"]):
            result = main()
        assert result == 0
        mock_update.assert_called_once_with(check_only=False, from_github=True)

    @patch("dictate.menubar_main._daemonize")
    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=33)
    def test_tty_without_foreground_daemonizes(self, mock_lock, mock_close, mock_daemon):
        """When running from a TTY without --foreground, should daemonize."""
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with (
            patch.object(sys, "argv", ["dictate"]),
            patch.object(sys, "stdin", mock_stdin),
        ):
            # _daemonize calls os._exit so main() won't continue
            # But with mock it returns, then main continues
            mock_app = MagicMock()
            with patch("dictate.menubar.DictateMenuBarApp", return_value=mock_app, create=True):
                main()
        mock_daemon.assert_called_once()

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=33)
    def test_foreground_flag_skips_daemonize(self, mock_lock, mock_close):
        mock_app = MagicMock()
        with (
            patch.object(sys, "argv", ["dictate", "--foreground"]),
            patch("dictate.menubar.DictateMenuBarApp", return_value=mock_app, create=True),
        ):
            result = main()
        assert result == 0

    @patch("dictate.menubar_main.os.close")
    @patch("dictate.menubar_main._acquire_singleton_lock", return_value=33)
    def test_f_flag_also_skips_daemonize(self, mock_lock, mock_close):
        mock_app = MagicMock()
        with (
            patch.object(sys, "argv", ["dictate", "-f"]),
            patch("dictate.menubar.DictateMenuBarApp", return_value=mock_app, create=True),
        ):
            result = main()
        assert result == 0
