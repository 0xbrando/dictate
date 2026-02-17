"""Tests for dictate.presets — quality presets, preferences, and hardware detection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dictate.config import LLMBackend, LLMModel, STTEngine
from dictate.presets import (
    COMMAND_KEYS,
    INPUT_LANGUAGES,
    OUTPUT_LANGUAGES,
    PTT_KEYS,
    QUALITY_PRESETS,
    SOUND_PRESETS,
    STT_PRESETS,
    WRITING_STYLES,
    Preferences,
    detect_chip,
    recommended_quality_preset,
)


# ── Preset data integrity ─────────────────────────────────────


class TestPresetData:
    def test_quality_presets_not_empty(self):
        assert len(QUALITY_PRESETS) >= 3

    def test_quality_presets_have_labels(self):
        for p in QUALITY_PRESETS:
            assert p.label, "Quality preset missing label"
            assert p.llm_model is not None

    def test_sound_presets_not_empty(self):
        assert len(SOUND_PRESETS) >= 2

    def test_sound_presets_include_none(self):
        """There should be a 'None' / silent option."""
        labels = [p.label.lower() for p in SOUND_PRESETS]
        assert any("none" in l for l in labels), "No silent sound preset"

    def test_stt_presets_have_whisper(self):
        engines = [p.engine for p in STT_PRESETS]
        assert STTEngine.WHISPER in engines

    def test_stt_presets_have_parakeet(self):
        engines = [p.engine for p in STT_PRESETS]
        assert STTEngine.PARAKEET in engines

    def test_input_languages_have_auto(self):
        codes = [code for code, _ in INPUT_LANGUAGES]
        assert "auto" in codes

    def test_output_languages_have_auto(self):
        codes = [code for code, _ in OUTPUT_LANGUAGES]
        assert "auto" in codes

    def test_ptt_keys_not_empty(self):
        assert len(PTT_KEYS) >= 2

    def test_writing_styles_include_clean(self):
        keys = [k for k, _, _ in WRITING_STYLES]
        assert "clean" in keys

    def test_writing_styles_include_formal(self):
        keys = [k for k, _, _ in WRITING_STYLES]
        assert "formal" in keys

    def test_writing_styles_include_bullets(self):
        keys = [k for k, _, _ in WRITING_STYLES]
        assert "bullets" in keys


# ── Hardware detection ─────────────────────────────────────────


class TestHardwareDetection:
    def test_detect_chip_returns_string(self):
        result = detect_chip()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_recommended_preset_is_valid_index(self):
        idx = recommended_quality_preset()
        assert 0 <= idx < len(QUALITY_PRESETS)


# ── Preferences ───────────────────────────────────────────────


class TestPreferences:
    def test_defaults(self):
        p = Preferences()
        assert p.device_id is None
        assert p.quality_preset == 1
        assert p.stt_preset == 0
        assert p.input_language == "auto"
        assert p.output_language == "auto"
        assert p.llm_cleanup is True
        assert p.writing_style == "clean"
        assert p.ptt_key == "ctrl_l"

    def test_llm_model_property(self):
        p = Preferences(quality_preset=1)
        assert isinstance(p.llm_model, LLMModel)

    def test_llm_model_clamps_out_of_range(self):
        p = Preferences(quality_preset=999)
        # Should clamp to last preset, not crash
        assert isinstance(p.llm_model, LLMModel)

    def test_backend_property(self):
        # Preset 0 is API (Smart routing)
        p = Preferences(quality_preset=0)
        assert p.backend == LLMBackend.API

        # Preset 1+ should be LOCAL
        p = Preferences(quality_preset=1)
        assert p.backend == LLMBackend.LOCAL

    def test_stt_engine_property(self):
        p = Preferences(stt_preset=0)
        assert p.stt_engine == STTEngine.PARAKEET

    def test_stt_engine_clamps(self):
        p = Preferences(stt_preset=999)
        assert isinstance(p.stt_engine, STTEngine)

    def test_whisper_language_auto_is_none(self):
        p = Preferences(input_language="auto")
        assert p.whisper_language is None

    def test_whisper_language_specific(self):
        p = Preferences(input_language="ja")
        assert p.whisper_language == "ja"

    def test_llm_output_language_auto_is_none(self):
        p = Preferences(output_language="auto")
        assert p.llm_output_language is None

    def test_sound_property(self):
        p = Preferences(sound_preset=0)
        s = p.sound
        assert s.label
        assert isinstance(s.start_hz, int)

    def test_api_url_validation_localhost(self):
        p = Preferences(api_url="http://localhost:8005/v1/chat/completions")
        assert p.validated_api_url == p.api_url

    def test_api_url_validation_blocks_remote(self):
        p = Preferences(api_url="https://evil.com/v1/chat/completions")
        assert "localhost" in p.validated_api_url

    def test_api_url_validation_allows_remote_with_env(self, monkeypatch):
        monkeypatch.setenv("DICTATE_ALLOW_REMOTE_API", "1")
        p = Preferences(api_url="https://api.example.com/v1/chat/completions")
        assert p.validated_api_url == "https://api.example.com/v1/chat/completions"


# ── Preferences save/load ─────────────────────────────────────


class TestPreferencesPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        prefs_dir = tmp_path / "Dictate"
        prefs_file = prefs_dir / "preferences.json"
        monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
        monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)

        p = Preferences(
            quality_preset=3,
            input_language="ja",
            writing_style="formal",
            ptt_key="ctrl_r",
        )
        p.save()

        assert prefs_file.exists()
        loaded = Preferences.load()
        assert loaded.quality_preset == 3
        assert loaded.input_language == "ja"
        assert loaded.writing_style == "formal"
        assert loaded.ptt_key == "ctrl_r"

    def test_load_missing_file_creates_defaults(self, tmp_path, monkeypatch):
        prefs_dir = tmp_path / "Dictate"
        prefs_file = prefs_dir / "preferences.json"
        monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
        monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)

        loaded = Preferences.load()
        assert loaded.input_language == "auto"
        # Should have auto-detected and saved
        assert prefs_file.exists()

    def test_load_corrupted_file_returns_defaults(self, tmp_path, monkeypatch):
        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        prefs_file = prefs_dir / "preferences.json"
        prefs_file.write_text("not valid json {{{")
        monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
        monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)

        loaded = Preferences.load()
        assert loaded.input_language == "auto"  # defaults


# ── Dictionary persistence ────────────────────────────────────


class TestDictionaryPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        prefs_dir = tmp_path / "Dictate"
        dict_file = prefs_dir / "dictionary.json"
        monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
        monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)

        words = ["OpenClaw", "MLX", "Brando"]
        Preferences.save_dictionary(words)

        assert dict_file.exists()
        loaded = Preferences.load_dictionary()
        assert loaded == words

    def test_load_empty(self, tmp_path, monkeypatch):
        dict_file = tmp_path / "dictionary.json"
        monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)

        loaded = Preferences.load_dictionary()
        assert loaded == []

    def test_load_legacy_list_format(self, tmp_path, monkeypatch):
        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        dict_file = prefs_dir / "dictionary.json"
        dict_file.write_text(json.dumps(["Word1", "Word2"]))
        monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)

        loaded = Preferences.load_dictionary()
        assert loaded == ["Word1", "Word2"]

    def test_load_dict_format(self, tmp_path, monkeypatch):
        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir(parents=True)
        dict_file = prefs_dir / "dictionary.json"
        dict_file.write_text(json.dumps({"words": ["A", "B"]}))
        monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)

        loaded = Preferences.load_dictionary()
        assert loaded == ["A", "B"]


class TestPrefsEdgeCases:
    """Cover remaining uncovered lines in presets.py."""

    def test_save_handles_os_error(self, tmp_path, monkeypatch):
        """save() should not raise on write failure."""
        from dictate.presets import Preferences
        prefs = Preferences()
        monkeypatch.setattr("dictate.presets.PREFS_FILE", Path("/nonexistent/stats.json"))
        prefs.save()  # Should not raise

    def test_detect_chip_exception(self, monkeypatch):
        """detect_chip should return 'Unknown' on failure."""
        from dictate.presets import detect_chip
        import subprocess
        monkeypatch.setattr(
            subprocess, "check_output",
            lambda *a, **kw: (_ for _ in ()).throw(Exception("no sysctl")),
        )
        result = detect_chip()
        assert result == "Unknown"

    def test_is_safe_api_url_invalid(self):
        """_is_safe_api_url should handle invalid URLs gracefully."""
        from dictate.presets import Preferences
        # Should return False for clearly invalid URLs
        assert Preferences._is_safe_api_url("not-a-url") is False
        assert Preferences._is_safe_api_url("ftp://localhost:8080") is False
        # Should return True for localhost
        assert Preferences._is_safe_api_url("http://localhost:8080/v1") is True
        assert Preferences._is_safe_api_url("http://127.0.0.1:8080") is True

    def test_load_v1_migration_quality_preset(self, tmp_path, monkeypatch):
        """Load v1 preferences — quality_preset migrates from v1."""
        from dictate.presets import Preferences
        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir()
        prefs_file = prefs_dir / "preferences.json"
        dict_file = prefs_dir / "dictionary.json"
        monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
        monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)
        monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)
        # v1 uses _prefs_version=1; quality_preset 1 → 1+1=2
        prefs_file.write_text(json.dumps({
            "_prefs_version": 1,
            "quality_preset": 1,
            "stt_preset": 0,
        }))
        loaded = Preferences.load()
        assert loaded.quality_preset == 2

    def test_load_v2_migration_high_quality_preset(self, tmp_path, monkeypatch):
        """Load v2 preferences with high quality preset index."""
        from dictate.presets import Preferences
        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir()
        prefs_file = prefs_dir / "preferences.json"
        dict_file = prefs_dir / "dictionary.json"
        monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
        monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)
        monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)
        # v2 uses _prefs_version=2; quality_preset 3 (>=2) → 3-1=2
        prefs_file.write_text(json.dumps({
            "_prefs_version": 2,
            "quality_preset": 3,
        }))
        loaded = Preferences.load()
        assert loaded.quality_preset == 2

    def test_load_v2_migration_low_quality_preset(self, tmp_path, monkeypatch):
        """Load v2 preferences with low quality preset (0.5B→1.5B)."""
        from dictate.presets import Preferences
        prefs_dir = tmp_path / "Dictate"
        prefs_dir.mkdir()
        prefs_file = prefs_dir / "preferences.json"
        dict_file = prefs_dir / "dictionary.json"
        monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
        monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)
        monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)
        # v2: quality_preset 1 (<=1) → max(0, 1) = 1
        prefs_file.write_text(json.dumps({
            "_prefs_version": 2,
            "quality_preset": 1,
        }))
        loaded = Preferences.load()
        assert loaded.quality_preset == 1
