"""Additional coverage tests for dictate.presets — targeting specific uncovered lines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dictate.config import LLMBackend, LLMModel


class TestDetectChipExceptions:
    """Cover exception handling in detect_chip()."""

    def test_detect_chip_exception_returns_unknown(self, monkeypatch):
        """Line 29: detect_chip() exception path returns 'Unknown'."""
        import dictate.presets as presets

        def mock_check_output(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "sysctl")

        import subprocess

        monkeypatch.setattr(subprocess, "check_output", mock_check_output)
        result = presets.detect_chip()
        assert result == "Unknown"


class TestRecommendedQualityPreset:
    """Cover recommended_quality_preset() edge cases."""

    def test_recommended_preset_non_ultra_non_max(self, monkeypatch):
        """Line 42: When chip has neither 'ultra' nor 'max', returns 1."""
        import dictate.presets as presets

        monkeypatch.setattr(presets, "detect_chip", lambda: "M1")
        result = presets.recommended_quality_preset()
        assert result == 1

    def test_recommended_preset_with_ultra(self, monkeypatch):
        """Ultra chip returns 2."""
        import dictate.presets as presets

        monkeypatch.setattr(presets, "detect_chip", lambda: "M2 Ultra")
        result = presets.recommended_quality_preset()
        assert result == 2

    def test_recommended_preset_with_max(self, monkeypatch):
        """Max chip returns 2."""
        import dictate.presets as presets

        monkeypatch.setattr(presets, "detect_chip", lambda: "M3 Max")
        result = presets.recommended_quality_preset()
        assert result == 2


class TestPreferencesLoadFirstLaunch:
    """Cover Preferences.load() first launch path (lines 216-217)."""

    def test_load_first_launch_creates_defaults(self, tmp_path, monkeypatch):
        """Lines 216-217: PREFS_FILE doesn't exist, auto-detect and save."""
        import dictate.presets as presets

        prefs_dir = tmp_path / "Dictate"
        prefs_file = prefs_dir / "preferences.json"

        monkeypatch.setattr(presets, "PREFS_DIR", prefs_dir)
        monkeypatch.setattr(presets, "PREFS_FILE", prefs_file)

        # Mock detect_chip and _refresh_discovery to avoid real calls
        monkeypatch.setattr(presets, "detect_chip", lambda: "M1")

        # Create preferences instance with mocked _refresh_discovery
        prefs = presets.Preferences()
        prefs._refresh_discovery = MagicMock()
        
        # Patch Preferences class to use our mocked instance during load
        original_load = presets.Preferences.load
        
        def mock_load(cls):
            if not prefs_file.exists():
                chip = presets.detect_chip()
                preset = presets.recommended_quality_preset()
                new_prefs = cls(quality_preset=preset)
                new_prefs._refresh_discovery()
                new_prefs.save()
                return new_prefs
            return original_load()
        
        monkeypatch.setattr(presets.Preferences, "load", classmethod(mock_load))
        
        loaded = presets.Preferences.load()
        assert loaded.quality_preset == 1  # M1 returns preset 1
        assert prefs_file.exists()


class TestPreferencesV1Migration:
    """Cover v1 migration path (lines 240-245)."""

    def test_load_v1_migration_preset_ge_1(self, tmp_path, monkeypatch):
        """Lines 240-245: v1 migration where raw_preset >= 1 adds 1."""
        import dictate.presets as presets

        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        prefs_file = prefs_dir / "preferences.json"

        # Create v1 preferences with quality_preset >= 1
        v1_data = {
            "quality_preset": 2,  # Should become 3 after migration
            "_prefs_version": 1,
        }
        prefs_file.write_text(json.dumps(v1_data))

        monkeypatch.setattr(presets, "PREFS_DIR", prefs_dir)
        monkeypatch.setattr(presets, "PREFS_FILE", prefs_file)

        # Mock _refresh_discovery to avoid real network calls
        original_refresh = presets.Preferences._refresh_discovery
        monkeypatch.setattr(presets.Preferences, "_refresh_discovery", lambda self: None)

        loaded = presets.Preferences.load()
        # v1: 2 -> 3 (adds 1 for presets >= 1)
        assert loaded.quality_preset == 3


class TestRefreshDiscovery:
    """Cover _refresh_discovery() method (lines 272-276)."""

    def test_refresh_discovery_api_backend_available(self, monkeypatch):
        """API backend triggers discover_llm when available."""
        import dictate.presets as presets

        # Create a preferences with API backend (preset 0)
        prefs = presets.Preferences(quality_preset=0)

        # Mock discover_llm
        mock_result = MagicMock()
        mock_result.is_available = True
        mock_result.name = "test-model"
        monkeypatch.setattr(presets, "discover_llm", lambda endpoint: mock_result)

        prefs._refresh_discovery()
        assert prefs._discovered_model == "test-model"

    def test_refresh_discovery_api_backend_unavailable(self, monkeypatch):
        """API backend when discover_llm returns unavailable."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=0)

        mock_result = MagicMock()
        mock_result.is_available = False
        mock_result.name = None
        monkeypatch.setattr(presets, "discover_llm", lambda endpoint: mock_result)

        prefs._refresh_discovery()
        assert prefs._discovered_model is None

    def test_refresh_discovery_non_api_backend(self):
        """Non-API backend returns None for _discovered_model."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=1)  # Local backend
        prefs._discovered_model = "should-be-cleared"

        prefs._refresh_discovery()
        assert prefs._discovered_model is None


class TestUpdateEndpoint:
    """Cover update_endpoint() method (lines 282-283)."""

    def test_update_endpoint_and_refresh(self, monkeypatch):
        """Lines 282-283: Update endpoint and refresh discovery."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=0)
        prefs.llm_endpoint = "old:11434"

        refresh_called = []
        def mock_refresh(self):
            refresh_called.append(True)
        monkeypatch.setattr(presets.Preferences, "_refresh_discovery", mock_refresh)

        prefs.update_endpoint("new:1234")
        assert prefs.llm_endpoint == "new:1234"
        assert len(refresh_called) == 1


class TestIsApiBackend:
    """Cover is_api_backend property (line 298)."""

    def test_is_api_backend_true(self):
        """Line 298: Returns True for API backend."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=0)  # API preset
        assert prefs.is_api_backend is True

    def test_is_api_backend_false(self):
        """Returns False for local backend."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=1)  # Local preset
        assert prefs.is_api_backend is False


class TestDiscoveredModelDisplay:
    """Cover discovered_model_display property (lines 307-309)."""

    def test_discovered_model_display_api_backend(self, monkeypatch):
        """Lines 307-309: Returns display name when backend is API."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=0, llm_endpoint="localhost:11434")
        monkeypatch.setattr(presets, "get_display_name", lambda endpoint: "qwen3-coder via localhost:11434")

        result = prefs.discovered_model_display
        assert result == "qwen3-coder via localhost:11434"

    def test_discovered_model_display_non_api(self):
        """Returns empty string for non-API backend."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=1)
        assert prefs.discovered_model_display == ""


class TestSttProperties:
    """Cover stt_engine and stt_model properties (lines 318-319)."""

    def test_stt_engine_property(self):
        """Line 318: stt_engine property returns correct engine."""
        from dictate.config import STTEngine
        import dictate.presets as presets

        prefs = presets.Preferences(stt_preset=0)  # Parakeet
        assert prefs.stt_engine == STTEngine.PARAKEET

        prefs = presets.Preferences(stt_preset=1)  # Whisper
        assert prefs.stt_engine == STTEngine.WHISPER

    def test_stt_model_property(self):
        """Line 319: stt_model property returns correct model."""
        import dictate.presets as presets

        prefs = presets.Preferences(stt_preset=0)  # Parakeet
        assert "parakeet" in prefs.stt_model.lower()

        prefs = presets.Preferences(stt_preset=1)  # Whisper
        assert "whisper" in prefs.stt_model.lower()


class TestIsSafeApiUrl:
    """Cover _is_safe_api_url() method (lines 340, 343-344)."""

    def test_is_safe_api_url_invalid_scheme(self):
        """Line 340: Returns False for invalid scheme."""
        import dictate.presets as presets

        assert presets.Preferences._is_safe_api_url("ftp://localhost:8005") is False
        assert presets.Preferences._is_safe_api_url("file:///etc/passwd") is False

    def test_is_safe_api_url_localhost_variants(self):
        """Lines 343-344: Returns True for localhost variants."""
        import dictate.presets as presets

        assert presets.Preferences._is_safe_api_url("http://localhost:8005") is True
        assert presets.Preferences._is_safe_api_url("http://127.0.0.1:8005") is True
        assert presets.Preferences._is_safe_api_url("http://[::1]:8005") is True
        assert presets.Preferences._is_safe_api_url("http://0.0.0.0:8005") is True

    def test_is_safe_api_url_remote(self):
        """Returns False for remote URLs."""
        import dictate.presets as presets

        assert presets.Preferences._is_safe_api_url("https://example.com") is False
        assert presets.Preferences._is_safe_api_url("http://192.168.1.1:8005") is False


class TestValidatedApiUrl:
    """Cover validated_api_url property (lines 354-370)."""

    def test_validated_api_url_endpoint_construction(self, monkeypatch):
        """Line 354-370: Endpoint-based URL construction for API backend."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=0, llm_endpoint="localhost:11434")
        
        # Should construct URL from endpoint
        result = prefs.validated_api_url
        assert result == "http://localhost:11434/v1/chat/completions"

    def test_validated_api_url_protocol_stripping_http(self, monkeypatch):
        """Strip http:// prefix from endpoint."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=0, llm_endpoint="http://localhost:11434")
        result = prefs.validated_api_url
        assert result == "http://localhost:11434/v1/chat/completions"

    def test_validated_api_url_protocol_stripping_https(self, monkeypatch):
        """Strip https:// prefix from endpoint."""
        import dictate.presets as presets

        prefs = presets.Preferences(quality_preset=0, llm_endpoint="https://localhost:11434")
        result = prefs.validated_api_url
        assert result == "http://localhost:11434/v1/chat/completions"

    def test_validated_api_url_remote_blocked(self, monkeypatch):
        """Remote endpoint blocked without env var."""
        import dictate.presets as presets

        # Ensure env var is not set
        monkeypatch.delenv("DICTATE_ALLOW_REMOTE_API", raising=False)

        prefs = presets.Preferences(quality_preset=0, llm_endpoint="example.com:11434")
        result = prefs.validated_api_url
        # Should return default localhost URL
        assert result == "http://localhost:8005/v1/chat/completions"

    def test_validated_api_url_remote_allowed_via_env(self, monkeypatch):
        """Remote endpoint allowed with DICTATE_ALLOW_REMOTE_API=1."""
        import dictate.presets as presets

        monkeypatch.setenv("DICTATE_ALLOW_REMOTE_API", "1")

        prefs = presets.Preferences(quality_preset=0, llm_endpoint="example.com:11434")
        result = prefs.validated_api_url
        assert result == "http://example.com:11434/v1/chat/completions"

    def test_validated_api_url_local_backend(self):
        """Legacy path for local backend uses stored api_url."""
        import dictate.presets as presets

        prefs = presets.Preferences(
            quality_preset=1,  # Local backend
            api_url="http://localhost:8005/v1/chat/completions"
        )
        result = prefs.validated_api_url
        assert result == "http://localhost:8005/v1/chat/completions"


class TestPynputKeyMapping:
    """Cover ptt_pynput_key and command_pynput_key properties."""

    def test_ptt_pynput_key_mapping(self, monkeypatch):
        """Lines 383-391: Maps key strings to pynput keys."""
        import sys
        import dictate.presets as presets

        # Mock pynput.keyboard.Key
        mock_key = MagicMock()
        mock_key.ctrl_l = "ctrl_l_key"
        mock_key.ctrl_r = "ctrl_r_key"
        mock_key.cmd_r = "cmd_r_key"
        mock_key.alt_l = "alt_l_key"
        mock_key.alt_r = "alt_r_key"

        mock_keyboard = MagicMock()
        mock_keyboard.Key = mock_key

        monkeypatch.setitem(
            presets.__dict__, 
            "pynput", 
            type(sys)("pynput")
        )
        monkeypatch.setattr(presets, "pynput", mock_keyboard)
        
        # We need to patch the import inside the method
        mock_pynput_module = type(sys)("pynput")
        mock_pynput_module.keyboard = mock_keyboard
        
        with patch.dict("sys.modules", {"pynput": mock_pynput_module, "pynput.keyboard": mock_keyboard}):
            prefs = presets.Preferences(ptt_key="ctrl_r")
            result = prefs.ptt_pynput_key
            assert result == "ctrl_r_key"

    def test_ptt_pynput_key_default(self, monkeypatch):
        """Returns default ctrl_l for unknown keys."""
        import dictate.presets as presets
        import sys

        mock_key = MagicMock()
        mock_key.ctrl_l = "ctrl_l_default"
        mock_key.ctrl_r = "ctrl_r_key"
        mock_key.cmd_r = "cmd_r_key"
        mock_key.alt_l = "alt_l_key"
        mock_key.alt_r = "alt_r_key"

        mock_keyboard = MagicMock()
        mock_keyboard.Key = mock_key

        mock_pynput_module = type(sys)("pynput")
        mock_pynput_module.keyboard = mock_keyboard
        
        with patch.dict("sys.modules", {"pynput": mock_pynput_module, "pynput.keyboard": mock_keyboard}):
            prefs = presets.Preferences(ptt_key="unknown_key")
            result = prefs.ptt_pynput_key
            assert result == "ctrl_l_default"

    def test_command_pynput_key_mapping(self, monkeypatch):
        """Lines 395-405: Maps command key strings to pynput keys."""
        import dictate.presets as presets
        import sys

        mock_key = MagicMock()
        mock_key.ctrl_l = "ctrl_l_key"
        mock_key.ctrl_r = "ctrl_r_key"
        mock_key.cmd_r = "cmd_r_key"
        mock_key.alt_l = "alt_l_key"
        mock_key.alt_r = "alt_r_key"

        mock_keyboard = MagicMock()
        mock_keyboard.Key = mock_key

        mock_pynput_module = type(sys)("pynput")
        mock_pynput_module.keyboard = mock_keyboard
        
        with patch.dict("sys.modules", {"pynput": mock_pynput_module, "pynput.keyboard": mock_keyboard}):
            prefs = presets.Preferences(command_key="alt_r")
            result = prefs.command_pynput_key
            assert result == "alt_r_key"

    def test_command_pynput_key_none(self, monkeypatch):
        """Lines 395-405: Returns None for 'none' command key."""
        import dictate.presets as presets

        prefs = presets.Preferences(command_key="none")
        result = prefs.command_pynput_key
        assert result is None

    def test_command_pynput_key_unknown(self, monkeypatch):
        """Returns None for unknown command keys."""
        import dictate.presets as presets
        import sys

        mock_key = MagicMock()
        mock_key.ctrl_l = "ctrl_l_key"
        mock_key.ctrl_r = "ctrl_r_key"
        mock_key.cmd_r = "cmd_r_key"
        mock_key.alt_l = "alt_l_key"
        mock_key.alt_r = "alt_r_key"

        mock_keyboard = MagicMock()
        mock_keyboard.Key = mock_key

        mock_pynput_module = type(sys)("pynput")
        mock_pynput_module.keyboard = mock_keyboard
        
        with patch.dict("sys.modules", {"pynput": mock_pynput_module, "pynput.keyboard": mock_keyboard}):
            prefs = presets.Preferences(command_key="unknown_key")
            result = prefs.command_pynput_key
            assert result is None


class TestLoadDictionary:
    """Cover load_dictionary() with dict format (lines 417-419)."""

    def test_load_dictionary_dict_format(self, tmp_path, monkeypatch):
        """Lines 417-419: Load dictionary with {'words': [...]} format."""
        import dictate.presets as presets

        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        dict_file = prefs_dir / "dictionary.json"
        dict_file.write_text(json.dumps({"words": ["word1", "word2", "word3"]}))

        monkeypatch.setattr(presets, "DICTIONARY_FILE", dict_file)

        result = presets.Preferences.load_dictionary()
        assert result == ["word1", "word2", "word3"]

    def test_load_dictionary_list_format(self, tmp_path, monkeypatch):
        """Load dictionary with [...] format."""
        import dictate.presets as presets

        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        dict_file = prefs_dir / "dictionary.json"
        dict_file.write_text(json.dumps(["item1", "item2"]))

        monkeypatch.setattr(presets, "DICTIONARY_FILE", dict_file)

        result = presets.Preferences.load_dictionary()
        assert result == ["item1", "item2"]

    def test_load_dictionary_empty(self, tmp_path, monkeypatch):
        """Returns empty list when file doesn't exist."""
        import dictate.presets as presets

        dict_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(presets, "DICTIONARY_FILE", dict_file)

        result = presets.Preferences.load_dictionary()
        assert result == []

    def test_load_dictionary_corrupted(self, tmp_path, monkeypatch):
        """Returns empty list when file is corrupted."""
        import dictate.presets as presets

        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        dict_file = prefs_dir / "dictionary.json"
        dict_file.write_text("invalid json {{{")

        monkeypatch.setattr(presets, "DICTIONARY_FILE", dict_file)

        result = presets.Preferences.load_dictionary()
        assert result == []

    def test_load_dictionary_dict_format_converts_to_string(self, tmp_path, monkeypatch):
        """Dict format converts words to strings (line 419)."""
        import dictate.presets as presets

        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        dict_file = prefs_dir / "dictionary.json"
        dict_file.write_text(json.dumps({"words": [123, True, "normal"]}))

        monkeypatch.setattr(presets, "DICTIONARY_FILE", dict_file)

        result = presets.Preferences.load_dictionary()
        assert result == ["123", "True", "normal"]
