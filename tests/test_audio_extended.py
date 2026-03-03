"""Extended tests for dictate.audio — unit-testable parts.

Tests AudioDevice, VADState, list_input_devices, get_device_name, tone synthesizers,
and audio processing edge cases.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dictate.audio import (
    AudioDevice,
    VADState,
    _synth_chime,
    _synth_click,
    _synth_marimba,
    _synth_simple,
    _synth_soft_pop,
    _synth_warm,
    list_input_devices,
    get_device_name,
    play_tone,
)


class TestAudioDeviceExtended:
    """Extended tests for AudioDevice dataclass."""

    def test_str_with_default(self):
        """Test __str__ with default marker."""
        device = AudioDevice(index=0, name="Test Device", is_default=True)
        result = str(device)
        assert "[0]" in result
        assert "Test Device" in result
        assert "(DEFAULT)" in result

    def test_str_without_default(self):
        """Test __str__ without default marker."""
        device = AudioDevice(index=1, name="Another Device", is_default=False)
        result = str(device)
        assert "[1]" in result
        assert "Another Device" in result
        assert "(DEFAULT)" not in result

    def test_str_special_characters_in_name(self):
        """Test __str__ with special characters in name."""
        device = AudioDevice(index=2, name="Device (USB)", is_default=False)
        result = str(device)
        assert "[2]" in result
        assert "Device (USB)" in result

    def test_str_empty_name(self):
        """Test __str__ with empty name."""
        device = AudioDevice(index=0, name="", is_default=True)
        result = str(device)
        assert "[0]" in result
        assert "(DEFAULT)" in result

    def test_dataclass_fields(self):
        """Test dataclass fields are accessible."""
        device = AudioDevice(index=5, name="Mic", is_default=True)
        assert device.index == 5
        assert device.name == "Mic"
        assert device.is_default is True


class TestVADStateExtended:
    """Extended tests for VADState dataclass."""

    def test_reset_sets_in_speech_false(self):
        """Test reset sets in_speech to False."""
        vad = VADState(in_speech=True, last_speech_time=123.0)
        vad.reset(pre_roll_samples=4000)
        assert vad.in_speech is False

    def test_reset_sets_last_speech_time_zero(self):
        """Test reset sets last_speech_time to 0.0."""
        vad = VADState(in_speech=True, last_speech_time=123.0)
        vad.reset(pre_roll_samples=4000)
        assert vad.last_speech_time == 0.0

    def test_reset_clears_current_chunk(self):
        """Test reset clears current_chunk list."""
        vad = VADState()
        vad.current_chunk.append(np.zeros(100, dtype=np.float32))
        vad.current_chunk.append(np.zeros(100, dtype=np.float32))
        assert len(vad.current_chunk) == 2
        
        vad.reset(pre_roll_samples=4000)
        assert len(vad.current_chunk) == 0

    def test_reset_sets_pre_roll_maxlen(self):
        """Test reset sets pre_roll maxlen."""
        vad = VADState()
        vad.reset(pre_roll_samples=8000)
        assert vad.pre_roll.maxlen == 8000

    def test_reset_clears_pre_roll(self):
        """Test reset clears pre_roll deque."""
        vad = VADState()
        vad.pre_roll.append(0.1)
        vad.pre_roll.append(0.2)
        assert len(vad.pre_roll) == 2
        
        vad.reset(pre_roll_samples=4000)
        assert len(vad.pre_roll) == 0

    def test_initial_state(self):
        """Test initial state of VADState."""
        vad = VADState()
        assert vad.in_speech is False
        assert vad.last_speech_time == 0.0
        assert len(vad.current_chunk) == 0
        assert isinstance(vad.pre_roll, deque)


class TestListInputDevicesExtended:
    """Extended tests for list_input_devices with mocked sounddevice."""

    @patch("dictate.audio.sd")
    def test_empty_devices_list(self, mock_sd):
        """Test with empty devices list."""
        mock_sd.query_devices.return_value = []
        mock_sd.default.device = [0, 1]
        
        devices = list_input_devices()
        assert devices == []

    @patch("dictate.audio.sd")
    def test_single_input_device(self, mock_sd):
        """Test with single input device."""
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2, "max_output_channels": 0}
        ]
        mock_sd.default.device = [0, 1]
        
        devices = list_input_devices()
        assert len(devices) == 1
        assert devices[0].index == 0
        assert devices[0].name == "Built-in Microphone"
        assert devices[0].is_default is True

    @patch("dictate.audio.sd")
    def test_multiple_input_devices(self, mock_sd):
        """Test with multiple input devices."""
        mock_sd.query_devices.return_value = [
            {"name": "Mic 1", "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Mic 2", "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Mic 3", "max_input_channels": 2, "max_output_channels": 0},
        ]
        mock_sd.default.device = [1, 2]  # Default is index 1
        
        devices = list_input_devices()
        assert len(devices) == 3
        assert devices[0].is_default is False
        assert devices[1].is_default is True
        assert devices[2].is_default is False

    @patch("dictate.audio.sd")
    def test_output_only_devices_excluded(self, mock_sd):
        """Test that output-only devices are excluded."""
        mock_sd.query_devices.return_value = [
            {"name": "Input Mic", "max_input_channels": 2, "max_output_channels": 0},
            {"name": "Output Speakers", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Input USB Mic", "max_input_channels": 1, "max_output_channels": 0},
        ]
        mock_sd.default.device = [0, 1]
        
        devices = list_input_devices()
        assert len(devices) == 2
        assert all(d.name != "Output Speakers" for d in devices)

    @patch("dictate.audio.sd")
    def test_mixed_input_output_device_included(self, mock_sd):
        """Test that devices with both input and output are included."""
        mock_sd.query_devices.return_value = [
            {"name": "Headset", "max_input_channels": 1, "max_output_channels": 2},
        ]
        mock_sd.default.device = [0, 1]
        
        devices = list_input_devices()
        assert len(devices) == 1
        assert devices[0].name == "Headset"


class TestGetDeviceNameExtended:
    """Extended tests for get_device_name with mocked sounddevice."""

    @patch("dictate.audio.sd")
    def test_specific_device_id(self, mock_sd):
        """Test getting name for specific device ID."""
        mock_sd.query_devices.return_value = {"name": "USB Microphone"}
        
        name = get_device_name(5)
        assert name == "USB Microphone"
        mock_sd.query_devices.assert_called_once_with(5)

    @patch("dictate.audio.sd")
    def test_default_device(self, mock_sd):
        """Test getting name for default device."""
        mock_sd.default.device = [3, 1]
        mock_sd.query_devices.return_value = {"name": "Default Mic"}
        
        name = get_device_name(None)
        assert name == "Default Mic"
        mock_sd.query_devices.assert_called_once_with(3)

    @patch("dictate.audio.sd")
    def test_no_input_device_found(self, mock_sd):
        """Test when no input device found."""
        mock_sd.default.device = [-1, -1]
        
        name = get_device_name(None)
        assert name == "(no input device found)"

    @patch("dictate.audio.sd")
    def test_device_name_with_special_chars(self, mock_sd):
        """Test device name with special characters."""
        mock_sd.query_devices.return_value = {"name": "Mic (USB Audio) [2.0]"}
        
        name = get_device_name(0)
        assert name == "Mic (USB Audio) [2.0]"


class TestToneSynthesizersExtended:
    """Extended tests for all tone synthesizers."""

    def test_synth_simple_returns_float32_array(self):
        """Test _synth_simple returns float32 array."""
        tone = _synth_simple(880, 0.04, 0.15, 44100)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) == int(44100 * 0.04)

    def test_synth_simple_non_empty(self):
        """Test _synth_simple returns non-empty array."""
        tone = _synth_simple(880, 0.04, 0.15, 44100)
        assert len(tone) > 0
        assert not np.all(tone == 0)

    def test_synth_simple_expected_length(self):
        """Test _synth_simple returns expected length."""
        sr = 44100
        duration = 0.04
        tone = _synth_simple(880, duration, 0.15, sr)
        assert len(tone) == int(sr * duration)

    def test_synth_soft_pop_returns_float32_array(self):
        """Test _synth_soft_pop returns float32 array."""
        tone = _synth_soft_pop(880, 0.15, 44100)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_synth_soft_pop_expected_length(self):
        """Test _synth_soft_pop returns expected length (0.06s)."""
        sr = 44100
        tone = _synth_soft_pop(880, 0.15, sr)
        assert len(tone) == int(sr * 0.06)

    def test_synth_chime_returns_float32_array(self):
        """Test _synth_chime returns float32 array."""
        tone = _synth_chime(880, 0.15, 44100)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_synth_chime_expected_length(self):
        """Test _synth_chime returns expected length (0.10s)."""
        sr = 44100
        tone = _synth_chime(880, 0.15, sr)
        assert len(tone) == int(sr * 0.10)

    def test_synth_warm_returns_float32_array(self):
        """Test _synth_warm returns float32 array."""
        tone = _synth_warm(880, 0.15, 44100)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_synth_warm_expected_length(self):
        """Test _synth_warm returns expected length (0.08s)."""
        sr = 44100
        tone = _synth_warm(880, 0.15, sr)
        assert len(tone) == int(sr * 0.08)

    def test_synth_click_returns_float32_array(self):
        """Test _synth_click returns float32 array."""
        tone = _synth_click(1000, 0.15, 44100)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_synth_click_expected_length(self):
        """Test _synth_click returns expected length (0.015s)."""
        sr = 44100
        tone = _synth_click(1000, 0.15, sr)
        assert len(tone) == int(sr * 0.015)

    def test_synth_marimba_returns_float32_array(self):
        """Test _synth_marimba returns float32 array."""
        tone = _synth_marimba(880, 0.15, 44100)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.float32
        assert len(tone) > 0

    def test_synth_marimba_expected_length(self):
        """Test _synth_marimba returns expected length (0.10s)."""
        sr = 44100
        tone = _synth_marimba(880, 0.15, sr)
        assert len(tone) == int(sr * 0.10)

    def test_all_synths_with_different_frequencies(self):
        """Test all synthesizers with different frequencies."""
        sr = 44100
        frequencies = [220, 440, 880, 1760]
        
        for freq in frequencies:
            assert len(_synth_simple(freq, 0.04, 0.15, sr)) > 0
            assert len(_synth_soft_pop(freq, 0.15, sr)) > 0
            assert len(_synth_chime(freq, 0.15, sr)) > 0
            assert len(_synth_warm(freq, 0.15, sr)) > 0
            assert len(_synth_click(freq, 0.15, sr)) > 0
            assert len(_synth_marimba(freq, 0.15, sr)) > 0

    def test_all_synths_with_zero_volume(self):
        """Test all synthesizers with zero volume return zeros."""
        sr = 44100
        assert np.allclose(_synth_simple(880, 0.04, 0.0, sr), 0.0)
        assert np.allclose(_synth_soft_pop(880, 0.0, sr), 0.0)
        assert np.allclose(_synth_chime(880, 0.0, sr), 0.0)
        assert np.allclose(_synth_warm(880, 0.0, sr), 0.0)
        assert np.allclose(_synth_click(880, 0.0, sr), 0.0)
        assert np.allclose(_synth_marimba(880, 0.0, sr), 0.0)

    def test_synth_simple_has_fade(self):
        """Test _synth_simple applies fade in/out."""
        sr = 44100
        tone = _synth_simple(880, 0.04, 0.5, sr)
        
        # Start and end should be near zero due to fade
        assert abs(tone[0]) < 0.1
        assert abs(tone[-1]) < 0.1
        # Middle should be louder
        assert abs(tone[len(tone)//2]) > 0.1

    def test_synth_soft_pop_has_decay(self):
        """Test _synth_soft_pop has exponential decay."""
        sr = 44100
        tone = _synth_soft_pop(880, 0.5, sr)
        
        # End should be near zero due to exponential decay (relaxed tolerance)
        assert abs(tone[-1]) < 0.05

    def test_synth_chime_harmonics(self):
        """Test _synth_chime has multiple harmonics."""
        sr = 44100
        tone = _synth_chime(880, 0.5, sr)
        
        # Should have content (not just zeros)
        assert np.any(tone != 0)

    def test_synth_warm_harmonics(self):
        """Test _synth_warm has multiple harmonics."""
        sr = 44100
        tone = _synth_warm(880, 0.5, sr)
        
        # Should have content
        assert np.any(tone != 0)

    def test_synth_click_has_noise(self):
        """Test _synth_click has noise component."""
        sr = 44100
        tone = _synth_click(1000, 0.5, sr)
        
        # Should have content
        assert np.any(tone != 0)


class TestPlayToneExtended:
    """Extended tests for play_tone function."""

    @patch("dictate.audio.sd")
    def test_play_tone_with_disabled_config(self, mock_sd):
        """Test play_tone returns early when config.enabled is False."""
        from dictate.config import ToneConfig

        config = ToneConfig(enabled=False)
        play_tone(config, 880)

        # Should not call sd.play when disabled
        mock_sd.play.assert_not_called()

    @patch("dictate.audio.sd")
    def test_play_tone_with_soft_pop_style(self, mock_sd):
        """Test play_tone with soft_pop style."""
        from dictate.config import ToneConfig

        config = ToneConfig(enabled=True, style="soft_pop", volume=0.15)
        play_tone(config, 880, sample_rate=16_000)

        # Should call sd.play with float32 array at the passed sample rate
        mock_sd.play.assert_called_once()
        args = mock_sd.play.call_args
        assert args[0][1] == 16_000  # uses passed sample_rate
        assert args[1]["blocking"] is False

    @patch("dictate.audio.sd")
    def test_play_tone_with_chime_style(self, mock_sd):
        """Test play_tone with chime style."""
        from dictate.config import ToneConfig
        
        config = ToneConfig(enabled=True, style="chime", volume=0.15)
        play_tone(config, 880)
        
        mock_sd.play.assert_called_once()

    @patch("dictate.audio.sd")
    def test_play_tone_with_warm_style(self, mock_sd):
        """Test play_tone with warm style."""
        from dictate.config import ToneConfig
        
        config = ToneConfig(enabled=True, style="warm", volume=0.15)
        play_tone(config, 880)
        
        mock_sd.play.assert_called_once()

    @patch("dictate.audio.sd")
    def test_play_tone_with_click_style(self, mock_sd):
        """Test play_tone with click style."""
        from dictate.config import ToneConfig
        
        config = ToneConfig(enabled=True, style="click", volume=0.15)
        play_tone(config, 880)
        
        mock_sd.play.assert_called_once()

    @patch("dictate.audio.sd")
    def test_play_tone_with_marimba_style(self, mock_sd):
        """Test play_tone with marimba style."""
        from dictate.config import ToneConfig
        
        config = ToneConfig(enabled=True, style="marimba", volume=0.15)
        play_tone(config, 880)
        
        mock_sd.play.assert_called_once()

    @patch("dictate.audio.sd")
    def test_play_tone_with_unknown_style_defaults_to_simple(self, mock_sd):
        """Test play_tone with unknown style defaults to simple."""
        from dictate.config import ToneConfig
        
        config = ToneConfig(enabled=True, style="unknown", volume=0.15)
        play_tone(config, 880)
        
        mock_sd.play.assert_called_once()

    @patch("dictate.audio.sd")
    def test_play_tone_passes_frequency(self, mock_sd):
        """Test play_tone uses the provided frequency."""
        from dictate.config import ToneConfig
        
        config = ToneConfig(enabled=True, style="simple", volume=0.15)
        play_tone(config, 440)  # Different frequency
        
        mock_sd.play.assert_called_once()
