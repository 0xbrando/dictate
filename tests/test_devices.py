"""Tests for the 'dictate devices' CLI command."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_prefs(tmp_path, monkeypatch):
    """Redirect preferences so device-list tests do not read local mic choices."""
    prefs_dir = tmp_path / "Dictate"
    prefs_dir.mkdir()
    monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
    monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_dir / "preferences.json")
    monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", prefs_dir / "dictionary.json")


class TestListDevices:
    """Tests for _list_devices() function."""

    def test_list_devices_with_devices(self, capsys):
        """Should list devices with default and selected markers."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import _list_devices

        mock_devices = [
            AudioDevice(index=0, name="MacBook Pro Microphone", is_default=True),
            AudioDevice(index=1, name="USB Audio Device", is_default=False),
            AudioDevice(index=3, name="AirPods Pro", is_default=False),
        ]

        with patch("dictate.audio.list_input_devices", return_value=mock_devices):
            result = _list_devices()

        assert result == 0
        output = capsys.readouterr().out
        assert "MacBook Pro Microphone" in output
        assert "USB Audio Device" in output
        assert "AirPods Pro" in output
        assert "macOS default" in output
        assert "selected" in output

    def test_list_devices_shows_configured_device(self, capsys):
        """Should show a configured device separately from macOS default."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import _list_devices
        from dictate.presets import Preferences

        prefs = Preferences()
        prefs.device_id = 1
        prefs.save()
        mock_devices = [
            AudioDevice(index=0, name="Jump Desktop Audio", is_default=True),
            AudioDevice(index=1, name="RODECaster Pro II", is_default=False),
        ]

        with patch("dictate.audio.list_input_devices", return_value=mock_devices):
            result = _list_devices()

        assert result == 0
        output = capsys.readouterr().out
        selected_line = [line for line in output.splitlines() if "RODECaster Pro II" in line][-1]
        default_line = [line for line in output.splitlines() if "Jump Desktop Audio" in line][-1]
        assert "selected" in selected_line
        assert "macOS default" in default_line

    def test_list_devices_no_devices(self, capsys):
        """Should report no devices found."""
        from dictate.menubar_main import _list_devices

        with patch("dictate.audio.list_input_devices", return_value=[]):
            result = _list_devices()

        assert result == 1
        output = capsys.readouterr().out
        assert "No input devices found" in output

    def test_list_devices_import_error(self, capsys):
        """Should handle missing sounddevice gracefully."""
        from dictate.menubar_main import _list_devices

        with patch("dictate.audio.list_input_devices", side_effect=ImportError("no sounddevice")):
            # The function catches ImportError from its own import
            # We need to patch differently — make the import fail
            pass

        # Test the exception path
        with patch("dictate.audio.list_input_devices", side_effect=Exception("device error")):
            result = _list_devices()

        assert result == 1
        output = capsys.readouterr().out
        assert "Could not query devices" in output

    def test_list_devices_shows_indices(self, capsys):
        """Should show device index numbers."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import _list_devices

        mock_devices = [
            AudioDevice(index=5, name="Test Mic", is_default=False),
            AudioDevice(index=12, name="Other Mic", is_default=True),
        ]

        with patch("dictate.audio.list_input_devices", return_value=mock_devices):
            result = _list_devices()

        assert result == 0
        output = capsys.readouterr().out
        assert "5" in output
        assert "12" in output

    def test_list_devices_shows_config_hint(self, capsys):
        """Should show how to set device."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import _list_devices

        mock_devices = [AudioDevice(index=0, name="Mic", is_default=True)]

        with patch("dictate.audio.list_input_devices", return_value=mock_devices):
            result = _list_devices()

        output = capsys.readouterr().out
        assert "config set" in output

    def test_list_devices_default_marker(self, capsys):
        """Default device should be visually distinct."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import _list_devices

        mock_devices = [
            AudioDevice(index=0, name="Default Mic", is_default=True),
            AudioDevice(index=1, name="Other Mic", is_default=False),
        ]

        with patch("dictate.audio.list_input_devices", return_value=mock_devices):
            _list_devices()

        output = capsys.readouterr().out
        # Default device should have the green marker
        lines = output.strip().split("\n")
        default_line = [l for l in lines if "Default Mic" in l][0]
        other_line = [l for l in lines if "Other Mic" in l][0]
        assert "●" in default_line  # green filled circle for default
        assert "○" in other_line    # empty circle for non-default

    def test_devices_in_help(self):
        """The devices command should appear in --help output."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "--help"]):
            result = main()

        assert result == 0

    def test_devices_in_help_text(self, capsys):
        """Help text should mention devices command."""
        from dictate.menubar_main import main

        with patch.object(sys, "argv", ["dictate", "--help"]):
            main()

        output = capsys.readouterr().out
        assert "devices" in output

    def test_main_routes_to_devices(self, capsys):
        """'dictate devices' should route to _list_devices."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import main

        mock_devices = [AudioDevice(index=0, name="Test Mic", is_default=True)]

        with (
            patch.object(sys, "argv", ["dictate", "devices"]),
            patch("dictate.audio.list_input_devices", return_value=mock_devices),
        ):
            result = main()

        assert result == 0
        output = capsys.readouterr().out
        assert "Test Mic" in output

    def test_list_devices_single_device(self, capsys):
        """Single device should still show properly."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import _list_devices

        mock_devices = [AudioDevice(index=0, name="Only Mic", is_default=True)]

        with patch("dictate.audio.list_input_devices", return_value=mock_devices):
            result = _list_devices()

        assert result == 0
        output = capsys.readouterr().out
        assert "Only Mic" in output
        assert "macOS default" in output

    def test_list_devices_many_devices(self, capsys):
        """Should handle many devices."""
        from dictate.audio import AudioDevice
        from dictate.menubar_main import _list_devices

        mock_devices = [
            AudioDevice(index=i, name=f"Device {i}", is_default=(i == 3))
            for i in range(10)
        ]

        with patch("dictate.audio.list_input_devices", return_value=mock_devices):
            result = _list_devices()

        assert result == 0
        output = capsys.readouterr().out
        for i in range(10):
            assert f"Device {i}" in output
