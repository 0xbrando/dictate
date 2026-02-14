"""Tests for writing styles feature — system prompts and raw mode."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from dictate.config import LLMConfig, LLMBackend
from dictate.presets import WRITING_STYLES


class TestWritingStyles:
    """Test that all writing styles are properly defined and mapped."""

    def test_all_styles_have_keys(self):
        """Every style tuple has (key, display_name, description)."""
        for style in WRITING_STYLES:
            assert len(style) == 3
            key, display, desc = style
            assert isinstance(key, str)
            assert isinstance(display, str)
            assert isinstance(desc, str)
            assert len(key) > 0
            assert len(display) > 0
            assert len(desc) > 0

    def test_style_keys_are_unique(self):
        """No duplicate style keys."""
        keys = [s[0] for s in WRITING_STYLES]
        assert len(keys) == len(set(keys))

    def test_default_styles_present(self):
        """Original 3 styles still exist."""
        keys = [s[0] for s in WRITING_STYLES]
        assert "clean" in keys
        assert "formal" in keys
        assert "bullets" in keys

    def test_new_styles_present(self):
        """New styles are registered."""
        keys = [s[0] for s in WRITING_STYLES]
        assert "email" in keys
        assert "slack" in keys
        assert "technical" in keys
        assert "tweet" in keys
        assert "raw" in keys

    def test_at_least_8_styles(self):
        """We have at least 8 writing styles."""
        assert len(WRITING_STYLES) >= 8


class TestSystemPrompts:
    """Test that each writing style generates a valid system prompt."""

    @pytest.fixture
    def llm_config(self):
        return LLMConfig()

    def test_clean_prompt(self, llm_config):
        llm_config.writing_style = "clean"
        prompt = llm_config.get_system_prompt()
        assert "post-processor" in prompt.lower()
        assert "punctuation" in prompt.lower()

    def test_formal_prompt(self, llm_config):
        llm_config.writing_style = "formal"
        prompt = llm_config.get_system_prompt()
        assert "formal" in prompt.lower()
        assert "professional" in prompt.lower()

    def test_bullets_prompt(self, llm_config):
        llm_config.writing_style = "bullets"
        prompt = llm_config.get_system_prompt()
        assert "bullet" in prompt.lower()

    def test_email_prompt(self, llm_config):
        llm_config.writing_style = "email"
        prompt = llm_config.get_system_prompt()
        assert "email" in prompt.lower()

    def test_slack_prompt(self, llm_config):
        llm_config.writing_style = "slack"
        prompt = llm_config.get_system_prompt()
        assert "chat" in prompt.lower() or "casual" in prompt.lower()

    def test_technical_prompt(self, llm_config):
        llm_config.writing_style = "technical"
        prompt = llm_config.get_system_prompt()
        assert "technical" in prompt.lower()

    def test_tweet_prompt(self, llm_config):
        llm_config.writing_style = "tweet"
        prompt = llm_config.get_system_prompt()
        assert "280" in prompt or "tweet" in prompt.lower()

    def test_unknown_style_falls_back_to_clean(self, llm_config):
        llm_config.writing_style = "nonexistent_style"
        prompt = llm_config.get_system_prompt()
        # Should fall back to clean
        assert "punctuation" in prompt.lower()

    def test_all_prompts_contain_only(self, llm_config):
        """Every style prompt instructs to output ONLY the processed text."""
        known_styles = ["clean", "formal", "bullets", "email", "slack", "technical", "tweet"]
        for style in known_styles:
            llm_config.writing_style = style
            prompt = llm_config.get_system_prompt()
            assert "ONLY" in prompt, f"Style '{style}' missing ONLY instruction"

    def test_translation_with_styles(self, llm_config):
        """Translation instruction is prepended regardless of style."""
        llm_config.writing_style = "email"
        llm_config.output_language = "ja"
        prompt = llm_config.get_system_prompt()
        assert "Japanese" in prompt
        assert "email" in prompt.lower()

    def test_dictionary_with_styles(self, llm_config):
        """Dictionary words are included regardless of style."""
        llm_config.writing_style = "technical"
        llm_config.dictionary = ["Kubernetes", "PostgreSQL"]
        prompt = llm_config.get_system_prompt()
        assert "Kubernetes" in prompt
        assert "PostgreSQL" in prompt
        assert "technical" in prompt.lower()


class TestRawMode:
    """Test that raw mode bypasses LLM cleanup."""

    def test_raw_style_in_presets(self):
        """Raw mode is registered as a writing style."""
        keys = [s[0] for s in WRITING_STYLES]
        assert "raw" in keys

    def test_raw_mode_skips_cleanup(self):
        """TranscriptionPipeline.process() skips LLM when style is 'raw'."""
        from dictate.transcribe import TranscriptionPipeline
        from dictate.config import WhisperConfig, LLMConfig, STTEngine

        whisper_config = WhisperConfig(engine=STTEngine.WHISPER)
        llm_config = LLMConfig(writing_style="raw")

        with patch("dictate.transcribe.WhisperTranscriber") as mock_whisper_cls, \
             patch("dictate.transcribe.TextCleaner") as mock_cleaner_cls:
            mock_whisper = MagicMock()
            mock_whisper.transcribe.return_value = "hello world"
            mock_whisper_cls.return_value = mock_whisper

            mock_cleaner = MagicMock()
            mock_cleaner_cls.return_value = mock_cleaner

            pipeline = TranscriptionPipeline(whisper_config, llm_config)

            audio = np.zeros(16000, dtype=np.int16)
            result = pipeline.process(audio)

            assert result == "hello world"
            # Cleaner should NOT have been called
            mock_cleaner.cleanup.assert_not_called()

    def test_raw_mode_preserves_duplicate_detection(self):
        """Raw mode still deduplicates consecutive identical outputs."""
        from dictate.transcribe import TranscriptionPipeline
        from dictate.config import WhisperConfig, LLMConfig, STTEngine

        whisper_config = WhisperConfig(engine=STTEngine.WHISPER)
        llm_config = LLMConfig(writing_style="raw")

        with patch("dictate.transcribe.WhisperTranscriber") as mock_whisper_cls, \
             patch("dictate.transcribe.TextCleaner") as mock_cleaner_cls:
            mock_whisper = MagicMock()
            mock_whisper.transcribe.return_value = "hello world"
            mock_whisper_cls.return_value = mock_whisper
            mock_cleaner_cls.return_value = MagicMock()

            pipeline = TranscriptionPipeline(whisper_config, llm_config)

            audio = np.zeros(16000, dtype=np.int16)
            result1 = pipeline.process(audio)
            result2 = pipeline.process(audio)

            assert result1 == "hello world"
            assert result2 is None  # Deduplicated
