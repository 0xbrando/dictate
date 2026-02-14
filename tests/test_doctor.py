"""Tests for the dictate doctor diagnostic command and status."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _run_doctor tests
# ---------------------------------------------------------------------------

class TestRunDoctor:
    """Tests for the _run_doctor() diagnostic function."""

    def _import_doctor(self):
        from dictate.menubar_main import _run_doctor
        return _run_doctor

    def _base_patches(self, **overrides):
        """Return a dict of common patches for doctor tests."""
        defaults = {
            "mac_ver": ("15.0.0", ("", "", ""), ""),
            "processor": "arm",
            "python_version": "3.12.0",
            "model_cached": True,
            "disk_free": 300e9,
            "pgrep_stdout": "\n",
            "sounddevice_ok": True,
        }
        defaults.update(overrides)
        return defaults

    def _run_with_patches(self, patches):
        doctor = self._import_doctor()
        
        sd_mock = MagicMock()
        if patches.get("sounddevice_ok"):
            # First call: list of all devices; second call: default input device
            sd_mock.query_devices = MagicMock(side_effect=[
                [{"name": "MacBook Mic", "max_input_channels": 1}],
                {"name": "MacBook Mic", "max_input_channels": 1},
            ])
        else:
            sd_mock.query_devices = MagicMock(side_effect=OSError("No audio"))

        modules = {
            "parakeet_mlx": MagicMock(),
            "Quartz": MagicMock(),
            "sounddevice": sd_mock,
        }

        pgrep_result = MagicMock(returncode=0, stdout=patches["pgrep_stdout"])

        with patch("platform.mac_ver", return_value=patches["mac_ver"]), \
             patch("platform.processor", return_value=patches["processor"]), \
             patch("platform.python_version", return_value=patches["python_version"]), \
             patch("dictate.config.is_model_cached", return_value=patches["model_cached"]), \
             patch("shutil.disk_usage", return_value=(500e9, 500e9 - patches["disk_free"], patches["disk_free"])), \
             patch("subprocess.run", return_value=pgrep_result), \
             patch.dict(sys.modules, modules):
            return doctor()

    def test_doctor_returns_zero_on_healthy_system(self):
        patches = self._base_patches()
        assert self._run_with_patches(patches) == 0

    def test_doctor_detects_old_macos(self):
        patches = self._base_patches(mac_ver=("12.0.0", ("", "", ""), ""))
        assert self._run_with_patches(patches) == 1

    def test_doctor_detects_intel_chip(self):
        patches = self._base_patches(processor="i386")
        assert self._run_with_patches(patches) == 1

    def test_doctor_detects_old_python(self):
        patches = self._base_patches(python_version="3.9.7")
        assert self._run_with_patches(patches) == 1

    def test_doctor_warns_on_python314(self):
        patches = self._base_patches(python_version="3.14.0")
        assert self._run_with_patches(patches) == 0  # Warning only

    def test_doctor_detects_low_disk_space(self):
        patches = self._base_patches(disk_free=2e9)
        assert self._run_with_patches(patches) == 1

    def test_doctor_warns_model_not_cached(self, capsys):
        patches = self._base_patches(model_cached=False)
        result = self._run_with_patches(patches)
        assert result == 0  # Warning only
        captured = capsys.readouterr()
        assert "not downloaded" in captured.out

    def test_doctor_handles_sounddevice_error(self):
        patches = self._base_patches(sounddevice_ok=False)
        assert self._run_with_patches(patches) == 0

    def test_doctor_warns_macos_13(self, capsys):
        patches = self._base_patches(mac_ver=("13.6.0", ("", "", ""), ""))
        result = self._run_with_patches(patches)
        assert result == 0  # Warning only
        captured = capsys.readouterr()
        assert "14+" in captured.out or "recommended" in captured.out

    def test_doctor_detects_multiple_instances(self, capsys):
        patches = self._base_patches(pgrep_stdout="12345\n67890\n")
        result = self._run_with_patches(patches)
        captured = capsys.readouterr()
        assert "Multiple instances" in captured.out

    def test_doctor_shows_all_checks_passed(self, capsys):
        patches = self._base_patches()
        result = self._run_with_patches(patches)
        captured = capsys.readouterr()
        assert "All checks passed" in captured.out

    def test_doctor_shows_no_parakeet(self, capsys):
        """When parakeet is not installed, show informational message."""
        doctor = self._import_doctor()

        sd_mock = MagicMock()
        sd_mock.query_devices = MagicMock(return_value={"name": "Mic", "max_input_channels": 1})

        modules = {
            "sounddevice": sd_mock,
            "Quartz": MagicMock(),
        }
        # Remove parakeet from available modules
        with patch("platform.mac_ver", return_value=("15.0.0", ("", "", ""), "")), \
             patch("platform.processor", return_value="arm"), \
             patch("platform.python_version", return_value="3.12.0"), \
             patch("dictate.config.is_model_cached", return_value=True), \
             patch("shutil.disk_usage", return_value=(500e9, 200e9, 300e9)), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="\n")), \
             patch.dict(sys.modules, modules), \
             patch.dict(sys.modules, {"parakeet_mlx": None}):
            # Need to make the import actually fail
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "parakeet_mlx":
                    raise ImportError("no parakeet")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                result = doctor()

        captured = capsys.readouterr()
        assert "Parakeet" in captured.out


# ---------------------------------------------------------------------------
# _show_status tests
# ---------------------------------------------------------------------------

class TestShowStatus:
    """Tests for the _show_status() function."""

    def _import_status(self):
        from dictate.menubar_main import _show_status
        return _show_status

    def test_status_returns_zero(self, capsys):
        status = self._import_status()

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="\n")), \
             patch("dictate.config.is_model_cached", return_value=False), \
             patch("dictate.config.get_cached_model_disk_size", return_value="0 B"), \
             patch.dict(sys.modules, {"parakeet_mlx": None}):
            import builtins
            real_import = builtins.__import__
            def fake_import(name, *args, **kwargs):
                if name == "parakeet_mlx":
                    raise ImportError("no parakeet")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=fake_import):
                result = status()

        captured = capsys.readouterr()
        assert "Dictate Status" in captured.out
        assert result == 0

    def test_status_shows_running_instance(self, capsys):
        status = self._import_status()

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="99999\n")), \
             patch("dictate.config.is_model_cached", return_value=True), \
             patch("dictate.config.get_cached_model_disk_size", return_value="1.2 GB"), \
             patch.dict(sys.modules, {"parakeet_mlx": None}):
            import builtins
            real_import = builtins.__import__
            def fake_import(name, *args, **kwargs):
                if name == "parakeet_mlx":
                    raise ImportError("no parakeet")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=fake_import):
                result = status()

        captured = capsys.readouterr()
        assert "Running" in captured.out

    def test_status_shows_not_running(self, capsys):
        status = self._import_status()

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=f"{os.getpid()}\n")), \
             patch("dictate.config.is_model_cached", return_value=False), \
             patch.dict(sys.modules, {"parakeet_mlx": None}):
            import builtins
            real_import = builtins.__import__
            def fake_import(name, *args, **kwargs):
                if name == "parakeet_mlx":
                    raise ImportError("no parakeet")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=fake_import):
                result = status()

        captured = capsys.readouterr()
        assert "Not running" in captured.out

    def test_status_shows_log_file(self, capsys, tmp_path):
        status = self._import_status()

        log_file = tmp_path / "dictate.log"
        log_file.write_text("test log content " * 100)

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="\n")), \
             patch("dictate.config.is_model_cached", return_value=False), \
             patch("dictate.menubar_main.LOG_FILE", log_file), \
             patch.dict(sys.modules, {"parakeet_mlx": None}):
            import builtins
            real_import = builtins.__import__
            def fake_import(name, *args, **kwargs):
                if name == "parakeet_mlx":
                    raise ImportError("no parakeet")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=fake_import):
                result = status()

        captured = capsys.readouterr()
        assert "Logs" in captured.out


# ---------------------------------------------------------------------------
# CLI routing tests
# ---------------------------------------------------------------------------

class TestCLIRouting:
    """Test that CLI commands route correctly."""

    def test_doctor_in_help_text(self, capsys):
        """Doctor command should appear in help text."""
        with patch("sys.argv", ["dictate", "--help"]):
            from dictate.menubar_main import main
            result = main()
        captured = capsys.readouterr()
        assert "doctor" in captured.out
        assert result == 0

    def test_doctor_command_routes(self):
        """'dictate doctor' should call _run_doctor."""
        with patch("sys.argv", ["dictate", "doctor"]), \
             patch("dictate.menubar_main._run_doctor", return_value=0) as mock_doctor:
            from dictate.menubar_main import main
            result = main()
            mock_doctor.assert_called_once()
            assert result == 0

    def test_status_command_routes(self):
        """'dictate status' should call _show_status."""
        with patch("sys.argv", ["dictate", "status"]), \
             patch("dictate.menubar_main._show_status", return_value=0) as mock_status:
            from dictate.menubar_main import main
            result = main()
            mock_status.assert_called_once()
            assert result == 0

    def test_config_command_routes(self):
        """'dictate config' should call _config_command."""
        with patch("sys.argv", ["dictate", "config"]), \
             patch("dictate.menubar_main._config_command", return_value=0) as mock_config:
            from dictate.menubar_main import main
            result = main()
            mock_config.assert_called_once()

    def test_stats_command_routes(self):
        """'dictate stats' should call _show_stats."""
        with patch("sys.argv", ["dictate", "stats"]), \
             patch("dictate.menubar_main._show_stats", return_value=0) as mock_stats:
            from dictate.menubar_main import main
            result = main()
            mock_stats.assert_called_once()
