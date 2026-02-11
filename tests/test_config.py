"""Tests for dictate.config — configuration dataclasses and system prompts."""

from __future__ import annotations

import pytest

from dictate.config import (
    AudioConfig,
    Config,
    LANGUAGE_NAMES,
    LLMBackend,
    LLMConfig,
    LLMModel,
    OutputMode,
    STTEngine,
    ToneConfig,
    VADConfig,
    WhisperConfig,
)


# ── AudioConfig ───────────────────────────────────────────────


class TestAudioConfig:
    def test_defaults(self):
        c = AudioConfig()
        assert c.sample_rate == 16_000
        assert c.channels == 1
        assert c.block_ms == 30
        assert c.device_id is None

    def test_block_size(self):
        c = AudioConfig(sample_rate=16_000, block_ms=30)
        assert c.block_size == 480  # 16000 * 0.03

    def test_block_size_custom(self):
        c = AudioConfig(sample_rate=48_000, block_ms=20)
        assert c.block_size == 960  # 48000 * 0.02


# ── VADConfig ─────────────────────────────────────────────────


class TestVADConfig:
    def test_defaults(self):
        c = VADConfig()
        assert c.rms_threshold == 0.012
        assert c.silence_timeout_s == 2.0
        assert c.pre_roll_s == 0.25
        assert c.post_roll_s == 0.15


# ── ToneConfig ────────────────────────────────────────────────


class TestToneConfig:
    def test_defaults(self):
        c = ToneConfig()
        assert c.enabled is True
        assert c.start_hz == 880
        assert c.stop_hz == 440
        assert c.duration_s == 0.04
        assert c.volume == 0.15
        assert c.style == "soft_pop"


# ── WhisperConfig ─────────────────────────────────────────────


class TestWhisperConfig:
    def test_defaults(self):
        c = WhisperConfig()
        assert "whisper" in c.model.lower()
        assert c.language is None
        assert c.engine == STTEngine.PARAKEET

    def test_custom_engine(self):
        c = WhisperConfig(engine=STTEngine.PARAKEET, model="test-model")
        assert c.engine == STTEngine.PARAKEET
        assert c.model == "test-model"


# ── LLMModel ──────────────────────────────────────────────────


class TestLLMModel:
    def test_hf_repo_mapping(self):
        assert "1.5B" in LLMModel.QWEN_1_5B.hf_repo
        assert "3B" in LLMModel.QWEN.hf_repo
        assert "7B" in LLMModel.QWEN_7B.hf_repo
        assert "14B" in LLMModel.QWEN_14B.hf_repo

    def test_all_models_have_repos(self):
        for model in LLMModel:
            repo = model.hf_repo
            assert repo, f"{model.value} has no hf_repo"
            assert "/" in repo, f"{model.value} repo missing org prefix"


# ── LLMConfig ─────────────────────────────────────────────────


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.enabled is True
        assert c.backend == LLMBackend.LOCAL
        assert c.temperature == 0.0
        assert c.max_tokens == 300
        assert c.output_language is None
        assert c.writing_style == "clean"
        assert c.dictionary is None

    def test_model_property(self):
        c = LLMConfig(model_choice=LLMModel.QWEN_7B)
        assert "7B" in c.model

    def test_system_prompt_clean(self):
        c = LLMConfig(writing_style="clean")
        prompt = c.get_system_prompt()
        assert "dictation post-processor" in prompt.lower()
        assert "NEVER answer questions" in prompt
        assert "Fix punctuation" in prompt

    def test_system_prompt_formal(self):
        c = LLMConfig(writing_style="formal")
        prompt = c.get_system_prompt()
        assert "professional" in prompt.lower() or "formal" in prompt.lower()

    def test_system_prompt_bullets(self):
        c = LLMConfig(writing_style="bullets")
        prompt = c.get_system_prompt()
        assert "bullet" in prompt.lower()

    def test_system_prompt_translation(self):
        c = LLMConfig(writing_style="clean")
        prompt = c.get_system_prompt(output_language="es")
        assert "Spanish" in prompt
        assert "TRANSLATE" in prompt

    def test_system_prompt_translation_via_config(self):
        c = LLMConfig(writing_style="clean", output_language="ja")
        prompt = c.get_system_prompt()
        assert "Japanese" in prompt

    def test_system_prompt_dictionary(self):
        c = LLMConfig(dictionary=["OpenClaw", "MLX", "Brando"])
        prompt = c.get_system_prompt()
        assert "OpenClaw" in prompt
        assert "MLX" in prompt
        assert "Brando" in prompt
        assert "exact spellings" in prompt.lower()

    def test_system_prompt_no_dictionary(self):
        c = LLMConfig(dictionary=None)
        prompt = c.get_system_prompt()
        assert "exact spellings" not in prompt.lower()

    def test_system_prompt_dictionary_limit(self):
        """Dictionary is limited to 50 words in the prompt."""
        words = [f"word_{i}" for i in range(100)]
        c = LLMConfig(dictionary=words)
        prompt = c.get_system_prompt()
        # Should contain word_0 through word_49 but not word_50+
        assert "word_0" in prompt
        assert "word_49" in prompt
        # The 51st word should be cut off by the [:50] slice
        assert "word_50" not in prompt

    def test_command_prompt(self):
        c = LLMConfig()
        prompt = c.get_command_prompt()
        assert "text editing assistant" in prompt.lower()
        assert "CLIPBOARD" in prompt


# ── LANGUAGE_NAMES ────────────────────────────────────────────


class TestLanguageNames:
    def test_common_languages_present(self):
        for code in ("en", "es", "fr", "de", "ja", "zh", "ko", "it", "pt"):
            assert code in LANGUAGE_NAMES, f"Missing language: {code}"

    def test_values_are_strings(self):
        for code, name in LANGUAGE_NAMES.items():
            assert isinstance(name, str)
            assert len(name) > 0


# ── Config.from_env ───────────────────────────────────────────


class TestConfigFromEnv:
    def test_defaults(self):
        c = Config()
        assert c.output_mode == OutputMode.TYPE
        assert c.verbose is True
        assert c.min_hold_to_process_s == 0.25

    def test_from_env_with_overrides(self, monkeypatch):
        monkeypatch.setenv("DICTATE_OUTPUT_MODE", "clipboard")
        monkeypatch.setenv("DICTATE_INPUT_LANGUAGE", "ja")
        monkeypatch.setenv("DICTATE_OUTPUT_LANGUAGE", "en")
        monkeypatch.setenv("DICTATE_LLM_CLEANUP", "false")
        monkeypatch.setenv("DICTATE_LLM_MODEL", "qwen-7b")
        monkeypatch.setenv("DICTATE_LLM_BACKEND", "api")

        c = Config.from_env()
        assert c.output_mode == OutputMode.CLIPBOARD
        assert c.whisper.language == "ja"
        assert c.llm.output_language == "en"
        assert c.llm.enabled is False
        assert c.llm.model_choice == LLMModel.QWEN_7B
        assert c.llm.backend == LLMBackend.API

    def test_auto_language_is_none(self, monkeypatch):
        monkeypatch.setenv("DICTATE_INPUT_LANGUAGE", "auto")
        monkeypatch.setenv("DICTATE_OUTPUT_LANGUAGE", "auto")
        c = Config.from_env()
        assert c.whisper.language is None
        assert c.llm.output_language is None

    def test_invalid_model_keeps_default(self, monkeypatch):
        monkeypatch.setenv("DICTATE_LLM_MODEL", "nonexistent-model")
        c = Config.from_env()
        assert c.llm.model_choice == LLMModel.QWEN  # default preserved


class TestAudioConfigValidation:
    """Test AudioConfig edge cases."""

    def test_block_size_zero_raises(self):
        from dictate.config import AudioConfig
        c = AudioConfig(block_ms=0)
        with pytest.raises(ValueError, match="block_ms must be positive"):
            _ = c.block_size

    def test_block_size_negative_raises(self):
        from dictate.config import AudioConfig
        c = AudioConfig(block_ms=-5)
        with pytest.raises(ValueError, match="block_ms must be positive"):
            _ = c.block_size

    def test_block_size_normal(self):
        from dictate.config import AudioConfig
        c = AudioConfig(sample_rate=16000, block_ms=30)
        assert c.block_size == 480  # 16000 * 0.03

    def test_get_model_size_str(self):
        from dictate.config import get_model_size_str
        from unittest.mock import patch
        with patch("dictate.model_download.get_model_size", return_value="1.8GB"):
            result = get_model_size_str("mlx-community/Qwen2.5-3B-Instruct-4bit")
            assert result == "1.8GB"
