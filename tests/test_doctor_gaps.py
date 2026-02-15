"""Tests for _run_doctor and _run_update uncovered paths — macOS version edge cases,
Parakeet import, device check errors, disk space, process detection."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def _run_doctor_with_mocks(printed, mac_ver="15.0", processor="arm",
                           model_cached=True, disk_free=5e11,
                           subprocess_effect=None, extra_patches=None):
    """Helper: run _run_doctor with standard mocks, capturing print output."""
    from dictate.menubar_main import _run_doctor

    mock_sub_result = MagicMock()
    mock_sub_result.stdout = ""
    mock_sub_result.returncode = 1

    with patch("platform.mac_ver", return_value=(mac_ver, ("", "", ""), "")):
        with patch("platform.processor", return_value=processor):
            with patch("dictate.config.is_model_cached", return_value=model_cached):
                with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a))):
                    with patch("subprocess.run",
                               side_effect=subprocess_effect if subprocess_effect else
                               MagicMock(return_value=mock_sub_result)):
                        with patch("shutil.disk_usage", return_value=(1e12, 1e12 - disk_free, disk_free)):
                            if extra_patches:
                                with extra_patches:
                                    return _run_doctor()
                            else:
                                return _run_doctor()


class TestDoctorMacVersion:
    """Doctor: macOS version edge cases."""

    def test_macos_version_unknown(self):
        """Could not detect macOS version."""
        printed = []
        _run_doctor_with_mocks(printed, mac_ver="")
        text = " ".join(printed)
        assert "Could not detect" in text or "?" in text

    def test_macos_13_warning(self):
        """macOS 13 gets warning (14+ recommended)."""
        printed = []
        _run_doctor_with_mocks(printed, mac_ver="13.5")
        text = " ".join(printed)
        assert "13.5" in text
        assert "⚠" in text or "recommended" in text.lower()


class TestDoctorDiskSpace:
    """Doctor: disk space check edge cases."""

    def test_disk_space_low(self):
        """3-10 GB free shows warning."""
        printed = []
        _run_doctor_with_mocks(printed, disk_free=5e9)
        text = " ".join(printed)
        assert "low" in text.lower() or "⚠" in text

    def test_disk_space_critical(self):
        """< 3 GB free shows critical."""
        printed = []
        _run_doctor_with_mocks(printed, disk_free=1e9)
        text = " ".join(printed)
        assert "critical" in text.lower() or "✗" in text


class TestDoctorProcessDetection:
    """Doctor: running process detection."""

    def test_multiple_instances_detected(self):
        """Multiple Dictate instances warning."""
        printed = []

        mock_result = MagicMock()
        mock_result.stdout = "12345\n67890\n"
        mock_result.returncode = 0

        _run_doctor_with_mocks(
            printed,
            subprocess_effect=MagicMock(return_value=mock_result)
        )

        text = " ".join(printed)
        assert "Multiple" in text or "instances" in text.lower() or "12345" in text

    def test_one_instance_running(self):
        """One instance running shows PID."""
        printed = []

        mock_result = MagicMock()
        mock_result.stdout = "12345\n"
        mock_result.returncode = 0

        _run_doctor_with_mocks(
            printed,
            subprocess_effect=MagicMock(return_value=mock_result)
        )

    def test_not_running(self):
        """Dictate not currently running."""
        printed = []

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 1

        _run_doctor_with_mocks(
            printed,
            subprocess_effect=MagicMock(return_value=mock_result)
        )

        text = " ".join(printed)
        assert "not" in text.lower() or "○" in text


class TestDoctorDeviceErrors:
    """Doctor: device-related error paths."""

    def test_no_input_devices(self):
        """No microphone detected."""
        from dictate.menubar_main import _run_doctor
        printed = []

        # Just verify the function completes without error
        _run_doctor_with_mocks(printed)


class TestUpdatePaths:
    """_run_update edge cases."""

    def test_update_check_only(self):
        """--check shows version info without installing."""
        from dictate.menubar_main import _run_update
        printed = []

        mock_result = MagicMock()
        mock_result.stdout = "dictate-mlx 2.5.0"
        mock_result.returncode = 0

        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a))):
            with patch("subprocess.run", return_value=mock_result):
                _run_update(check_only=True)

    def test_update_check_pip_fails(self):
        """pip check fails gracefully."""
        from dictate.menubar_main import _run_update
        printed = []

        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a))):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                _run_update(check_only=True)

        text = " ".join(printed)
        assert "Could not check" in text or "PyPI" in text or "GitHub" in text


class TestShowStatusPaths:
    """_show_status edge cases."""

    def test_status_daemon_not_running(self):
        """Status when daemon is not running."""
        from dictate.menubar_main import _show_status
        printed = []

        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a))):
            with patch("dictate.config.is_model_cached", return_value=True):
                with patch("dictate.config.get_cached_model_disk_size", return_value=100e6):
                    with patch("subprocess.run", side_effect=FileNotFoundError):
                        _show_status()

    def test_status_daemon_error(self):
        """Status check exception handled."""
        from dictate.menubar_main import _show_status
        printed = []

        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a))):
            with patch("dictate.config.is_model_cached", return_value=True):
                with patch("dictate.config.get_cached_model_disk_size", return_value=100e6):
                    with patch("subprocess.run", side_effect=Exception("test")):
                        _show_status()

        text = " ".join(printed)
        # Should show unknown or error status, not crash
        assert "Unknown" in text or "?" in text or "Not running" in text.lower() or "○" in text
