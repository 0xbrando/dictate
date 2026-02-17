"""Tests for the 'dictate config' CLI command."""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest


# ── Helpers ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_prefs(tmp_path, monkeypatch):
    """Redirect preferences to a temp directory so tests don't touch real prefs."""
    prefs_dir = tmp_path / "Dictate"
    prefs_dir.mkdir()
    prefs_file = prefs_dir / "preferences.json"
    dict_file = prefs_dir / "dictionary.json"
    monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
    monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)
    monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)
    return prefs_dir, prefs_file


def _run_config(*args):
    """Import and run the config command with given args."""
    from dictate.menubar_main import _config_command
    return _config_command(list(args))


# ── Show / display ──────────────────────────────────────────────


class TestConfigShow:
    def test_show_defaults(self, capsys):
        """Show config with no prefs file → uses defaults."""
        rc = _run_config("show")
        assert rc == 0
        out = capsys.readouterr().out
        assert "writing_style" in out
        assert "clean" in out
        assert "quality" in out
        assert "ptt_key" in out

    def test_show_no_arg(self, capsys):
        """No subcommand defaults to show."""
        rc = _run_config()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dictate Config" in out

    def test_show_with_existing_prefs(self, _isolate_prefs, capsys):
        """Show after setting a preference."""
        _run_config("set", "writing_style", "formal")
        rc = _run_config("show")
        assert rc == 0
        out = capsys.readouterr().out
        assert "formal" in out

    def test_show_after_reset(self, capsys):
        """Show after reset returns defaults."""
        _run_config("set", "writing_style", "bullets")
        _run_config("reset")
        rc = _run_config("show")
        assert rc == 0
        out = capsys.readouterr().out
        assert "clean" in out


# ── Set command ─────────────────────────────────────────────────


class TestConfigSet:
    def test_set_writing_style(self, capsys):
        rc = _run_config("set", "writing_style", "formal")
        assert rc == 0
        out = capsys.readouterr().out
        assert "writing_style" in out
        assert "formal" in out

    def test_set_writing_style_bullets(self, capsys):
        rc = _run_config("set", "writing_style", "bullets")
        assert rc == 0
        out = capsys.readouterr().out
        assert "bullets" in out

    def test_set_quality_by_index(self, capsys):
        rc = _run_config("set", "quality", "2")
        assert rc == 0
        # Verify it persisted
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.quality_preset == 2

    def test_set_quality_by_alias(self, capsys):
        rc = _run_config("set", "quality", "speedy")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.quality_preset == 1

    def test_set_quality_api(self, capsys):
        rc = _run_config("set", "quality", "api")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.quality_preset == 0

    def test_set_quality_balanced(self, capsys):
        rc = _run_config("set", "quality", "balanced")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.quality_preset == 3

    def test_set_stt_by_alias(self, capsys):
        rc = _run_config("set", "stt", "whisper")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.stt_preset == 1

    def test_set_stt_by_index(self, capsys):
        rc = _run_config("set", "stt", "0")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.stt_preset == 0

    def test_set_input_language(self, capsys):
        rc = _run_config("set", "input_language", "ja")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.input_language == "ja"

    def test_set_output_language(self, capsys):
        rc = _run_config("set", "output_language", "es")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.output_language == "es"

    def test_set_ptt_key(self, capsys):
        rc = _run_config("set", "ptt_key", "cmd_r")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.ptt_key == "cmd_r"

    def test_set_command_key(self, capsys):
        rc = _run_config("set", "command_key", "alt_r")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.command_key == "alt_r"

    def test_set_llm_cleanup_off(self, capsys):
        rc = _run_config("set", "llm_cleanup", "off")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.llm_cleanup is False

    def test_set_llm_cleanup_on(self, capsys):
        _run_config("set", "llm_cleanup", "off")
        rc = _run_config("set", "llm_cleanup", "on")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.llm_cleanup is True

    def test_set_llm_cleanup_true(self, capsys):
        rc = _run_config("set", "llm_cleanup", "true")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.llm_cleanup is True

    def test_set_llm_cleanup_false(self, capsys):
        rc = _run_config("set", "llm_cleanup", "false")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.llm_cleanup is False

    def test_set_llm_cleanup_numeric(self, capsys):
        rc = _run_config("set", "llm_cleanup", "0")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.llm_cleanup is False

    def test_set_sound_by_alias(self, capsys):
        rc = _run_config("set", "sound", "chime")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.sound_preset == 1

    def test_set_sound_by_index(self, capsys):
        rc = _run_config("set", "sound", "3")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.sound_preset == 3

    def test_set_llm_endpoint(self, capsys):
        rc = _run_config("set", "llm_endpoint", "localhost:8080")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.llm_endpoint == "localhost:8080"

    def test_set_advanced_mode_on(self, capsys):
        rc = _run_config("set", "advanced_mode", "on")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.advanced_mode is True

    def test_set_advanced_mode_off(self, capsys):
        _run_config("set", "advanced_mode", "on")
        rc = _run_config("set", "advanced_mode", "off")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.advanced_mode is False


# ── Error handling ──────────────────────────────────────────────


class TestConfigErrors:
    def test_invalid_key(self, capsys):
        rc = _run_config("set", "nonexistent_key", "value")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Unknown key" in err

    def test_invalid_writing_style(self, capsys):
        rc = _run_config("set", "writing_style", "nonexistent")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid value" in err

    def test_invalid_quality_value(self, capsys):
        rc = _run_config("set", "quality", "99")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid value" in err

    def test_invalid_quality_alias(self, capsys):
        rc = _run_config("set", "quality", "ultra_mega")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid value" in err

    def test_invalid_stt(self, capsys):
        rc = _run_config("set", "stt", "deepgram")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid value" in err

    def test_invalid_language(self, capsys):
        rc = _run_config("set", "input_language", "klingon")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid value" in err

    def test_invalid_ptt_key(self, capsys):
        rc = _run_config("set", "ptt_key", "spacebar")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid value" in err

    def test_invalid_llm_cleanup_value(self, capsys):
        rc = _run_config("set", "llm_cleanup", "maybe")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid value" in err

    def test_set_missing_value(self, capsys):
        rc = _run_config("set", "writing_style")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Usage" in err

    def test_set_no_args(self, capsys):
        rc = _run_config("set")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Usage" in err

    def test_unknown_subcommand(self, capsys):
        rc = _run_config("delete")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Unknown subcommand" in err


# ── Reset ───────────────────────────────────────────────────────


class TestConfigReset:
    def test_reset_restores_defaults(self, capsys):
        _run_config("set", "writing_style", "formal")
        _run_config("set", "llm_cleanup", "off")
        _run_config("set", "quality", "4")
        rc = _run_config("reset")
        assert rc == 0
        from dictate.presets import Preferences
        prefs = Preferences.load()
        assert prefs.writing_style == "clean"
        assert prefs.llm_cleanup is True
        assert prefs.quality_preset == 1  # default

    def test_reset_message(self, capsys):
        rc = _run_config("reset")
        assert rc == 0
        out = capsys.readouterr().out
        assert "reset" in out.lower()


# ── Path ────────────────────────────────────────────────────────


class TestConfigPath:
    def test_path_output(self, _isolate_prefs, capsys):
        prefs_dir, prefs_file = _isolate_prefs
        rc = _run_config("path")
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert out == str(prefs_file)


# ── Persistence ─────────────────────────────────────────────────


class TestConfigPersistence:
    def test_set_persists_to_file(self, _isolate_prefs):
        prefs_dir, prefs_file = _isolate_prefs
        _run_config("set", "writing_style", "bullets")
        assert prefs_file.exists()
        data = json.loads(prefs_file.read_text())
        assert data["writing_style"] == "bullets"

    def test_multiple_sets_accumulate(self, _isolate_prefs):
        prefs_dir, prefs_file = _isolate_prefs
        _run_config("set", "writing_style", "formal")
        _run_config("set", "input_language", "de")
        _run_config("set", "llm_cleanup", "off")
        data = json.loads(prefs_file.read_text())
        assert data["writing_style"] == "formal"
        assert data["input_language"] == "de"
        assert data["llm_cleanup"] is False

    def test_reset_creates_default_file(self, _isolate_prefs):
        prefs_dir, prefs_file = _isolate_prefs
        _run_config("reset")
        assert prefs_file.exists()
        data = json.loads(prefs_file.read_text())
        assert data["writing_style"] == "clean"
        assert data["llm_cleanup"] is True


# ── Integration with main() ─────────────────────────────────────


class TestConfigMainIntegration:
    def test_help_mentions_config(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["dictate", "--help"])
        from dictate.menubar_main import main
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "config" in out
        assert "View and modify preferences" in out

    def test_main_routes_config_show(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["dictate", "config"])
        from dictate.menubar_main import main
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dictate Config" in out

    def test_main_routes_config_set(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["dictate", "config", "set", "writing_style", "formal"])
        from dictate.menubar_main import main
        rc = main()
        assert rc == 0

    def test_main_routes_config_reset(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["dictate", "config", "reset"])
        from dictate.menubar_main import main
        rc = main()
        assert rc == 0

    def test_main_routes_config_path(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["dictate", "config", "path"])
        from dictate.menubar_main import main
        rc = main()
        assert rc == 0


# ── Edge cases ──────────────────────────────────────────────────


class TestConfigEdgeCases:
    def test_list_is_alias_for_show(self, capsys):
        rc = _run_config("list")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dictate Config" in out

    def test_get_is_alias_for_show(self, capsys):
        rc = _run_config("get")
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dictate Config" in out

    def test_set_all_languages(self, capsys):
        """Test every valid input language can be set."""
        from dictate.presets import INPUT_LANGUAGES
        for code, name in INPUT_LANGUAGES:
            rc = _run_config("set", "input_language", code)
            assert rc == 0, f"Failed to set input_language={code}"

    def test_set_all_output_languages(self, capsys):
        """Test every valid output language can be set."""
        from dictate.presets import OUTPUT_LANGUAGES
        for code, name in OUTPUT_LANGUAGES:
            rc = _run_config("set", "output_language", code)
            assert rc == 0, f"Failed to set output_language={code}"

    def test_set_all_ptt_keys(self, capsys):
        """Test every valid PTT key can be set."""
        from dictate.presets import PTT_KEYS
        for code, name in PTT_KEYS:
            rc = _run_config("set", "ptt_key", code)
            assert rc == 0, f"Failed to set ptt_key={code}"

    def test_set_all_command_keys(self, capsys):
        """Test every valid command key can be set."""
        from dictate.presets import COMMAND_KEYS
        for code, name in COMMAND_KEYS:
            rc = _run_config("set", "command_key", code)
            assert rc == 0, f"Failed to set command_key={code}"

    def test_set_all_quality_aliases(self, capsys):
        """Test every quality alias works."""
        aliases = ["api", "speedy", "fast", "balanced", "quality"]
        for alias in aliases:
            rc = _run_config("set", "quality", alias)
            assert rc == 0, f"Failed to set quality={alias}"

    def test_show_no_prefs_file(self, _isolate_prefs, capsys):
        """Show works even without any saved preferences."""
        prefs_dir, prefs_file = _isolate_prefs
        if prefs_file.exists():
            prefs_file.unlink()
        rc = _run_config("show")
        assert rc == 0
        out = capsys.readouterr().out
        assert "No preferences file" in out

    def test_set_error_shows_available_keys(self, capsys):
        """'config set' with missing args shows available keys."""
        rc = _run_config("set")
        assert rc == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "writing_style" in combined
        assert "quality" in combined
        assert "ptt_key" in combined
