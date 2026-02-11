"""Extended tests for dictate.config — additional coverage for Config.from_env() and LLMConfig.

Tests environment variable handling and LLM configuration edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dictate.config import (
    LLMBackend,
    LLMConfig,
    LLMModel,
    OutputMode,
    is_model_cached,
)


class TestConfigFromEnvExtended:
    """Extended tests for Config.from_env() with various environment variables."""

    def test_audio_device_env(self, monkeypatch):
        """Test DICTATE_AUDIO_DEVICE sets audio device_id."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_AUDIO_DEVICE", "2")
        config = Config.from_env()
        assert config.audio.device_id == 2

    def test_whisper_model_env(self, monkeypatch):
        """Test DICTATE_WHISPER_MODEL sets custom whisper model."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_WHISPER_MODEL", "mlx-community/whisper-custom")
        config = Config.from_env()
        assert config.whisper.model == "mlx-community/whisper-custom"

    def test_input_language_specific_env(self, monkeypatch):
        """Test DICTATE_INPUT_LANGUAGE with specific language code."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_INPUT_LANGUAGE", "pl")
        config = Config.from_env()
        assert config.whisper.language == "pl"

    def test_output_language_specific_env(self, monkeypatch):
        """Test DICTATE_OUTPUT_LANGUAGE with specific language code."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_OUTPUT_LANGUAGE", "de")
        config = Config.from_env()
        assert config.llm.output_language == "de"

    def test_verbose_true_env(self, monkeypatch):
        """Test DICTATE_VERBOSE=true enables verbose mode."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_VERBOSE", "true")
        config = Config.from_env()
        assert config.verbose is True

    def test_verbose_yes_env(self, monkeypatch):
        """Test DICTATE_VERBOSE=yes enables verbose mode."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_VERBOSE", "yes")
        config = Config.from_env()
        assert config.verbose is True

    def test_verbose_one_env(self, monkeypatch):
        """Test DICTATE_VERBOSE=1 enables verbose mode."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_VERBOSE", "1")
        config = Config.from_env()
        assert config.verbose is True

    def test_verbose_false_env(self, monkeypatch):
        """Test DICTATE_VERBOSE=false disables verbose mode."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_VERBOSE", "false")
        config = Config.from_env()
        assert config.verbose is False

    def test_verbose_zero_env(self, monkeypatch):
        """Test DICTATE_VERBOSE=0 disables verbose mode."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_VERBOSE", "0")
        config = Config.from_env()
        assert config.verbose is False

    def test_llm_cleanup_true_env(self, monkeypatch):
        """Test DICTATE_LLM_CLEANUP=true enables LLM cleanup."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_CLEANUP", "true")
        config = Config.from_env()
        assert config.llm.enabled is True

    def test_llm_cleanup_false_env(self, monkeypatch):
        """Test DICTATE_LLM_CLEANUP=false disables LLM cleanup."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_CLEANUP", "false")
        config = Config.from_env()
        assert config.llm.enabled is False

    def test_llm_model_qwen_1_5b_env(self, monkeypatch):
        """Test DICTATE_LLM_MODEL=qwen-1.5b sets correct model."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_MODEL", "qwen-1.5b")
        config = Config.from_env()
        assert config.llm.model_choice == LLMModel.QWEN_1_5B

    def test_llm_model_phi3_env(self, monkeypatch):
        """Test DICTATE_LLM_MODEL=phi3 sets correct model."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_MODEL", "phi3")
        config = Config.from_env()
        assert config.llm.model_choice == LLMModel.PHI3

    def test_llm_model_qwen_env(self, monkeypatch):
        """Test DICTATE_LLM_MODEL=qwen sets correct model."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_MODEL", "qwen")
        config = Config.from_env()
        assert config.llm.model_choice == LLMModel.QWEN

    def test_llm_model_qwen_7b_env(self, monkeypatch):
        """Test DICTATE_LLM_MODEL=qwen-7b sets correct model."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_MODEL", "qwen-7b")
        config = Config.from_env()
        assert config.llm.model_choice == LLMModel.QWEN_7B

    def test_llm_model_qwen_14b_env(self, monkeypatch):
        """Test DICTATE_LLM_MODEL=qwen-14b sets correct model."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_MODEL", "qwen-14b")
        config = Config.from_env()
        assert config.llm.model_choice == LLMModel.QWEN_14B

    def test_llm_model_invalid_env(self, monkeypatch):
        """Test invalid DICTATE_LLM_MODEL keeps default."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_MODEL", "invalid-model")
        config = Config.from_env()
        assert config.llm.model_choice == LLMModel.QWEN  # default

    def test_llm_backend_local_env(self, monkeypatch):
        """Test DICTATE_LLM_BACKEND=local sets local backend."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_BACKEND", "local")
        config = Config.from_env()
        assert config.llm.backend == LLMBackend.LOCAL

    def test_llm_backend_api_env(self, monkeypatch):
        """Test DICTATE_LLM_BACKEND=api sets API backend."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_BACKEND", "api")
        config = Config.from_env()
        assert config.llm.backend == LLMBackend.API

    def test_llm_backend_invalid_env(self, monkeypatch):
        """Test invalid DICTATE_LLM_BACKEND keeps default."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_LLM_BACKEND", "invalid")
        config = Config.from_env()
        assert config.llm.backend == LLMBackend.LOCAL  # default

    def test_llm_api_url_env(self, monkeypatch):
        """Test DICTATE_LLM_API_URL sets custom API URL."""
        from dictate.config import Config
        
        custom_url = "http://custom-endpoint:8080/v1/chat/completions"
        monkeypatch.setenv("DICTATE_LLM_API_URL", custom_url)
        config = Config.from_env()
        assert config.llm.api_url == custom_url

    def test_output_mode_type_env(self, monkeypatch):
        """Test DICTATE_OUTPUT_MODE=type sets type mode."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_OUTPUT_MODE", "type")
        config = Config.from_env()
        assert config.output_mode == OutputMode.TYPE

    def test_output_mode_clipboard_env(self, monkeypatch):
        """Test DICTATE_OUTPUT_MODE=clipboard sets clipboard mode."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_OUTPUT_MODE", "clipboard")
        config = Config.from_env()
        assert config.output_mode == OutputMode.CLIPBOARD

    def test_all_env_vars_together(self, monkeypatch):
        """Test all environment variables work together."""
        from dictate.config import Config
        
        monkeypatch.setenv("DICTATE_AUDIO_DEVICE", "3")
        monkeypatch.setenv("DICTATE_OUTPUT_MODE", "clipboard")
        monkeypatch.setenv("DICTATE_WHISPER_MODEL", "mlx-community/custom-model")
        monkeypatch.setenv("DICTATE_INPUT_LANGUAGE", "fr")
        monkeypatch.setenv("DICTATE_OUTPUT_LANGUAGE", "es")
        monkeypatch.setenv("DICTATE_VERBOSE", "false")
        monkeypatch.setenv("DICTATE_LLM_CLEANUP", "false")
        monkeypatch.setenv("DICTATE_LLM_MODEL", "phi3")
        monkeypatch.setenv("DICTATE_LLM_BACKEND", "api")
        monkeypatch.setenv("DICTATE_LLM_API_URL", "http://api:8000/v1")
        
        config = Config.from_env()
        
        assert config.audio.device_id == 3
        assert config.output_mode == OutputMode.CLIPBOARD
        assert config.whisper.model == "mlx-community/custom-model"
        assert config.whisper.language == "fr"
        assert config.llm.output_language == "es"
        assert config.verbose is False
        assert config.llm.enabled is False
        assert config.llm.model_choice == LLMModel.PHI3
        assert config.llm.backend == LLMBackend.API
        assert config.llm.api_url == "http://api:8000/v1"


class TestLLMConfigExtended:
    """Extended tests for LLMConfig class."""

    def test_get_system_prompt_english(self):
        """Test system prompt with English output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="en")
        assert "English" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_polish(self):
        """Test system prompt with Polish output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="pl")
        assert "Polish" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_german(self):
        """Test system prompt with German output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="de")
        assert "German" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_french(self):
        """Test system prompt with French output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="fr")
        assert "French" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_spanish(self):
        """Test system prompt with Spanish output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="es")
        assert "Spanish" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_italian(self):
        """Test system prompt with Italian output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="it")
        assert "Italian" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_portuguese(self):
        """Test system prompt with Portuguese output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="pt")
        assert "Portuguese" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_dutch(self):
        """Test system prompt with Dutch output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="nl")
        assert "Dutch" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_japanese(self):
        """Test system prompt with Japanese output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="ja")
        assert "Japanese" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_chinese(self):
        """Test system prompt with Chinese output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="zh")
        assert "Chinese" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_korean(self):
        """Test system prompt with Korean output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="ko")
        assert "Korean" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_russian(self):
        """Test system prompt with Russian output."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="ru")
        assert "Russian" in prompt
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_unknown_language(self):
        """Test system prompt with unknown language code (passes through)."""
        config = LLMConfig(writing_style="clean")
        prompt = config.get_system_prompt(output_language="xx")
        assert "xx" in prompt  # Passes through unknown code
        assert "TRANSLATE" in prompt

    def test_get_system_prompt_formal_style(self):
        """Test formal writing style prompt."""
        config = LLMConfig(writing_style="formal")
        prompt = config.get_system_prompt()
        assert "professional" in prompt.lower() or "formal" in prompt.lower()
        assert "NEVER answer questions" in prompt

    def test_get_system_prompt_bullets_style(self):
        """Test bullets writing style prompt."""
        config = LLMConfig(writing_style="bullets")
        prompt = config.get_system_prompt()
        assert "bullet" in prompt.lower()
        assert "NEVER answer questions" in prompt

    def test_get_system_prompt_unknown_style_defaults_to_clean(self):
        """Test unknown writing style defaults to clean."""
        config = LLMConfig(writing_style="unknown_style")
        prompt = config.get_system_prompt()
        assert "Fix punctuation" in prompt  # clean style

    def test_endpoint_api_url_with_plain_host(self):
        """Test endpoint_api_url with plain host:port."""
        config = LLMConfig(endpoint="localhost:11434")
        assert config.endpoint_api_url == "http://localhost:11434/v1/chat/completions"

    def test_endpoint_api_url_with_http_prefix(self):
        """Test endpoint_api_url strips http:// prefix."""
        config = LLMConfig(endpoint="http://localhost:11434")
        assert config.endpoint_api_url == "http://localhost:11434/v1/chat/completions"

    def test_endpoint_api_url_with_https_prefix(self):
        """Test endpoint_api_url strips https:// prefix."""
        config = LLMConfig(endpoint="https://localhost:11434")
        assert config.endpoint_api_url == "http://localhost:11434/v1/chat/completions"

    def test_endpoint_api_url_with_path(self):
        """Test endpoint_api_url strips path."""
        config = LLMConfig(endpoint="localhost:11434/v1/models")
        assert config.endpoint_api_url == "http://localhost:11434/v1/chat/completions"

    def test_endpoint_api_url_with_whitespace(self):
        """Test endpoint_api_url handles whitespace."""
        config = LLMConfig(endpoint="  localhost:11434  ")
        assert config.endpoint_api_url == "http://localhost:11434/v1/chat/completions"

    def test_endpoint_api_url_with_trailing_slash(self):
        """Test endpoint_api_url handles trailing slash."""
        config = LLMConfig(endpoint="localhost:11434/")
        assert config.endpoint_api_url == "http://localhost:11434/v1/chat/completions"

    def test_get_command_prompt_content(self):
        """Test get_command_prompt returns expected content."""
        config = LLMConfig()
        prompt = config.get_command_prompt()
        assert "text editing assistant" in prompt.lower()
        assert "CLIPBOARD" in prompt
        assert "make it shorter" in prompt
        assert "make it formal" in prompt
        assert "fix the grammar" in prompt
        assert "Output ONLY the final text" in prompt

    def test_system_prompt_property(self):
        """Test system_prompt property calls get_system_prompt."""
        config = LLMConfig(writing_style="clean")
        assert config.system_prompt == config.get_system_prompt()

    def test_model_property_for_all_models(self):
        """Test model property returns correct HF repo for all models."""
        for model in LLMModel:
            config = LLMConfig(model_choice=model)
            assert config.model == model.hf_repo


class TestIsModelCached:
    """Tests for is_model_cached function using real filesystem via tmp_path."""

    def test_model_cached_returns_true(self, tmp_path, monkeypatch):
        """Test is_model_cached returns True when model snapshot dir exists with contents."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        snap_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen2.5-3B-Instruct-4bit" / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "abc123").mkdir()  # a snapshot
        assert is_model_cached("mlx-community/Qwen2.5-3B-Instruct-4bit") is True

    def test_model_not_cached_dir_not_exists(self, tmp_path, monkeypatch):
        """Test is_model_cached returns False when directory doesn't exist."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert is_model_cached("mlx-community/Qwen2.5-3B-Instruct-4bit") is False

    def test_model_not_cached_empty_dir(self, tmp_path, monkeypatch):
        """Test is_model_cached returns False when snapshot dir is empty."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        snap_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--mlx-community--Qwen2.5-3B-Instruct-4bit" / "snapshots"
        snap_dir.mkdir(parents=True)
        assert is_model_cached("mlx-community/Qwen2.5-3B-Instruct-4bit") is False

    def test_is_model_cached_with_real_path(self):
        """Test is_model_cached with actual Path (won't find anything)."""
        # This tests with real filesystem - should return False for non-existent model
        result = is_model_cached("nonexistent/model-test-12345")
        assert result is False
