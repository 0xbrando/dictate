"""Tests for shell completion scripts."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

COMPLETIONS_DIR = Path(__file__).resolve().parent.parent / "completions"


class TestBashCompletions:
    """Verify bash completion script structure."""

    def test_bash_file_exists(self):
        assert (COMPLETIONS_DIR / "dictate.bash").exists()

    def test_bash_completion_has_commands(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for cmd in ["config", "stats", "status", "doctor", "update"]:
            assert cmd in content, f"Missing command: {cmd}"

    def test_bash_completion_has_config_keys(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for key in [
            "writing_style", "quality", "stt", "input_language",
            "output_language", "ptt_key", "command_key", "llm_cleanup",
            "sound", "llm_endpoint", "advanced_mode",
        ]:
            assert key in content, f"Missing config key: {key}"

    def test_bash_completion_has_writing_styles(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for style in ["clean", "formal", "bullets", "email", "slack", "technical", "tweet", "raw"]:
            assert style in content, f"Missing writing style: {style}"

    def test_bash_completion_has_quality_aliases(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for alias in ["api", "speedy", "fast", "balanced", "quality"]:
            assert alias in content, f"Missing quality alias: {alias}"

    def test_bash_completion_has_languages(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for lang in ["auto", "en", "ja", "zh", "ko", "es", "fr", "de"]:
            assert lang in content, f"Missing language: {lang}"

    def test_bash_completion_has_ptt_keys(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for key in ["ctrl_l", "ctrl_r", "cmd_r", "alt_l", "alt_r"]:
            assert key in content, f"Missing PTT key: {key}"

    def test_bash_completion_has_stt_engines(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        assert "parakeet" in content
        assert "whisper" in content

    def test_bash_completion_has_bool_values(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        assert "on off" in content or ("on" in content and "off" in content)

    def test_bash_completion_registers_function(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        assert "complete -F" in content
        assert "_dictate_completions" in content

    def test_bash_syntax_valid(self):
        """Check bash completion script has valid syntax."""
        bash_file = COMPLETIONS_DIR / "dictate.bash"
        result = subprocess.run(
            ["bash", "-n", str(bash_file)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_bash_completion_has_flags(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for flag in ["--help", "--version", "--foreground", "-h", "-V", "-f"]:
            assert flag in content, f"Missing flag: {flag}"

    def test_bash_completion_handles_config_set(self):
        """Verify config set path completes key then value."""
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        # Should have case for words[2] == "set"
        assert '"set"' in content or "'set'" in content

    def test_bash_completion_has_sound_values(self):
        content = (COMPLETIONS_DIR / "dictate.bash").read_text()
        for sound in ["soft_pop", "chime", "warm", "click", "marimba"]:
            assert sound in content, f"Missing sound: {sound}"


class TestZshCompletions:
    """Verify zsh completion script structure."""

    def test_zsh_file_exists(self):
        assert (COMPLETIONS_DIR / "dictate.zsh").exists()

    def test_zsh_completion_has_compdef(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        assert "#compdef dictate" in content

    def test_zsh_completion_has_commands(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        for cmd in ["config", "stats", "status", "doctor", "update"]:
            assert cmd in content, f"Missing command: {cmd}"

    def test_zsh_completion_has_descriptions(self):
        """Zsh completions should include descriptions."""
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        # Writing styles should have descriptions
        assert "Fixes punctuation" in content
        assert "Professional tone" in content
        assert "Distills into key points" in content

    def test_zsh_completion_has_writing_styles(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        for style in ["clean", "formal", "bullets", "email", "slack", "technical", "tweet", "raw"]:
            assert style in content, f"Missing writing style: {style}"

    def test_zsh_completion_has_config_keys(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        for key in [
            "writing_style", "quality", "stt", "input_language",
            "output_language", "ptt_key", "command_key", "llm_cleanup",
            "sound", "llm_endpoint", "advanced_mode",
        ]:
            assert key in content, f"Missing config key: {key}"

    def test_zsh_completion_has_quality_descriptions(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        assert "1.5B" in content  # speedy
        assert "3B" in content    # fast
        assert "8B" in content    # quality

    def test_zsh_completion_has_language_descriptions(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        assert "English" in content
        assert "Japanese" in content
        assert "Chinese" in content

    def test_zsh_completion_has_ptt_key_descriptions(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        assert "Left Control" in content
        assert "Right Command" in content

    def test_zsh_completion_has_config_subcmds(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        for subcmd in ["show", "set", "reset", "path"]:
            assert subcmd in content, f"Missing config subcommand: {subcmd}"

    def test_zsh_completion_has_flags(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        for flag in ["--help", "--version", "--foreground"]:
            assert flag in content, f"Missing flag: {flag}"

    def test_zsh_completion_has_sound_descriptions(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        for sound in ["soft_pop", "chime", "warm", "click", "marimba"]:
            assert sound in content, f"Missing sound: {sound}"

    def test_zsh_main_function_name(self):
        content = (COMPLETIONS_DIR / "dictate.zsh").read_text()
        assert "_dictate()" in content or "_dictate() {" in content
        assert '_dictate "$@"' in content
