"""Configuration for the Dictate application."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynput.keyboard import Key

from pynput import keyboard

logger = logging.getLogger(__name__)


class STTEngine(str, Enum):
    WHISPER = "whisper"
    PARAKEET = "parakeet"


class OutputMode(str, Enum):
    TYPE = "type"
    CLIPBOARD = "clipboard"


class LLMBackend(str, Enum):
    LOCAL = "local"
    API = "api"


class LLMModel(str, Enum):
    QWEN_1_5B = "qwen-1.5b"
    PHI3 = "phi3"
    QWEN = "qwen"
    QWEN_7B = "qwen-7b"
    QWEN_14B = "qwen-14b"

    @property
    def hf_repo(self) -> str:
        repos = {
            LLMModel.QWEN_1_5B: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            LLMModel.PHI3: "mlx-community/Phi-3-mini-4k-instruct-4bit",
            LLMModel.QWEN: "mlx-community/Qwen2.5-3B-Instruct-4bit",
            LLMModel.QWEN_7B: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            LLMModel.QWEN_14B: "mlx-community/Qwen2.5-14B-Instruct-4bit",
        }
        return repos.get(self, repos[LLMModel.QWEN])


# STT Models
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
PARAKEET_V2_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"  # English only
PARAKEET_V3_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"  # 25 European languages, same speed


def is_model_cached(hf_repo: str) -> bool:
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{hf_repo.replace('/', '--')}" / "snapshots"
    return model_dir.is_dir() and any(model_dir.iterdir())


def get_model_size_str(hf_repo: str) -> str:
    """Return approximate size string for known models.

    Args:
        hf_repo: HuggingFace repository name

    Returns:
        Size string like '1.8GB' or 'Unknown'
    """
    from dictate.model_download import get_model_size

    return get_model_size(hf_repo)


def delete_cached_model(hf_repo: str) -> bool:
    """Delete a cached model from the HuggingFace cache.

    Args:
        hf_repo: HuggingFace repository name (e.g., 'mlx-community/Qwen2.5-3B-Instruct-4bit')

    Returns:
        True if the model directory was found and deleted, False otherwise
    """
    from pathlib import Path
    import shutil

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{hf_repo.replace('/', '--')}"

    if model_dir.is_dir():
        try:
            shutil.rmtree(model_dir)
            logger.info("Deleted cached model: %s", hf_repo)
            return True
        except Exception as e:
            logger.error("Failed to delete model %s: %s", hf_repo, e)
            return False
    return False


def get_cached_model_disk_size(hf_repo: str) -> str:
    """Calculate the actual disk usage of a cached model.

    Args:
        hf_repo: HuggingFace repository name

    Returns:
        Human-readable size string like '1.8 GB' or 'Unknown'
    """
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{hf_repo.replace('/', '--')}"

    if not model_dir.is_dir():
        return "Unknown"

    try:
        total_size = 0
        for path in model_dir.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size

        # Convert to human-readable format
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if total_size < 1024:
                return f"{total_size:.1f} {unit}" if unit != "B" else f"{total_size} {unit}"
            total_size /= 1024
        return f"{total_size:.1f} PB"
    except Exception as e:
        logger.error("Failed to calculate disk size for %s: %s", hf_repo, e)
        return "Unknown"


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    block_ms: int = 30
    device_id: int | None = None

    @property
    def block_size(self) -> int:
        if self.block_ms <= 0:
            raise ValueError(f"block_ms must be positive, got {self.block_ms}")
        return int(self.sample_rate * (self.block_ms / 1000.0))


@dataclass
class VADConfig:
    rms_threshold: float = 0.012
    silence_timeout_s: float = 2.0
    pre_roll_s: float = 0.25
    post_roll_s: float = 0.15


@dataclass
class ToneConfig:
    enabled: bool = True
    start_hz: int = 880
    stop_hz: int = 440
    duration_s: float = 0.04
    volume: float = 0.15
    style: str = "soft_pop"


@dataclass
class WhisperConfig:
    model: str = "mlx-community/whisper-large-v3-turbo"
    language: str | None = None
    engine: STTEngine = STTEngine.PARAKEET


# Language name mapping for LLM prompts
LANGUAGE_NAMES = {
    "en": "English",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ru": "Russian",
}


@dataclass
class LLMConfig:
    enabled: bool = True
    backend: LLMBackend = LLMBackend.LOCAL
    model_choice: LLMModel = LLMModel.QWEN
    api_url: str = "http://localhost:8005/v1/chat/completions"
    endpoint: str = "localhost:11434"  # LLM endpoint for local API servers
    max_tokens: int = 300
    temperature: float = 0.0
    output_language: str | None = None
    writing_style: str = "clean"
    dictionary: list[str] | None = None

    @property
    def endpoint_api_url(self) -> str:
        """Get the OpenAI-compatible API URL from the endpoint."""
        endpoint = self.endpoint.strip()
        # Remove protocol prefix if present
        if endpoint.startswith("http://"):
            endpoint = endpoint[7:]
        elif endpoint.startswith("https://"):
            endpoint = endpoint[8:]
        # Remove trailing slash and path
        endpoint = endpoint.split("/")[0]
        host = endpoint.split(":")[0]
        scheme = "http" if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") else "https"
        return f"{scheme}://{endpoint}/v1/chat/completions"

    @property
    def model(self) -> str:
        models = {
            LLMModel.QWEN_1_5B: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            LLMModel.PHI3: "mlx-community/Phi-3-mini-4k-instruct-4bit",
            LLMModel.QWEN: "mlx-community/Qwen2.5-3B-Instruct-4bit",
            LLMModel.QWEN_7B: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            LLMModel.QWEN_14B: "mlx-community/Qwen2.5-14B-Instruct-4bit",
        }
        return models.get(self.model_choice, models[LLMModel.QWEN])

    def get_system_prompt(self, output_language: str | None = None) -> str:
        target_lang = output_language if output_language is not None else self.output_language

        if target_lang:
            lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
            translation_instruction = (
                f"TRANSLATE the input text to {lang_name}. "
                f"Output the translation in {lang_name} language only. "
            )
        else:
            translation_instruction = ""

        base = (
            "The input is speech-to-text output from a human dictating. "
            "NEVER answer questions. NEVER add your own words. "
            "NEVER respond conversationally. NEVER offer suggestions. "
        )

        style_prompts = {
            "clean": (
                "You are a dictation post-processor. "
                "Fix punctuation and capitalization. "
                "Output ONLY the cleaned-up text exactly as they said it."
            ),
            "formal": (
                "You are a dictation post-processor. "
                "Rewrite in a professional, formal tone. "
                "Use proper grammar and complete sentences. "
                "Output ONLY the rewritten text."
            ),
            "bullets": (
                "You are a dictation post-processor. "
                "Convert the dictation into concise bullet points. "
                "Strip filler words and extract key ideas. "
                "Each bullet should be one clear action or point. "
                "Output ONLY the bullet points."
            ),
            "email": (
                "You are a dictation post-processor. "
                "Format the dictation as a professional email. "
                "Add appropriate greeting and sign-off if not present. "
                "Fix grammar, punctuation, and paragraph breaks. "
                "Output ONLY the formatted email text."
            ),
            "slack": (
                "You are a dictation post-processor. "
                "Rewrite as a casual, concise chat message. "
                "Keep it short and conversational. "
                "Remove filler words and unnecessary formality. "
                "Output ONLY the chat message."
            ),
            "technical": (
                "You are a dictation post-processor. "
                "Rewrite in precise technical documentation style. "
                "Use clear, unambiguous language. "
                "Preserve technical terms and code references exactly. "
                "Output ONLY the technical text."
            ),
            "tweet": (
                "You are a dictation post-processor. "
                "Condense the dictation into a tweet (max 280 characters). "
                "Make it engaging and clear. Remove filler words. "
                "Output ONLY the tweet text, nothing else."
            ),
        }

        style = style_prompts.get(self.writing_style, style_prompts["clean"])

        dict_instruction = ""
        if self.dictionary:
            words = ", ".join(self.dictionary[:50])
            dict_instruction = (
                f"IMPORTANT: Always use these exact spellings when they appear: {words}. "
            )

        return f"{translation_instruction}{base}{dict_instruction}{style}"

    @property
    def system_prompt(self) -> str:
        return self.get_system_prompt()

    def get_command_prompt(self) -> str:
        return (
            "You are a text editing assistant. The user will speak a command describing "
            "how to modify text. The CLIPBOARD contains the text to modify. "
            "Apply the spoken command to the clipboard text and output ONLY the modified result. "
            "Common commands: 'make it shorter', 'make it formal', 'fix the grammar', "
            "'translate to Spanish', 'rewrite as bullet points', 'delete the last sentence', "
            "'add a greeting'. Output ONLY the final text, no explanations."
        )


@dataclass
class KeybindConfig:
    ptt_key: "Key" = field(default_factory=lambda: keyboard.Key.ctrl_l)
    quit_key: "Key" = field(default_factory=lambda: keyboard.Key.esc)
    quit_modifier: "Key" = field(default_factory=lambda: keyboard.Key.cmd)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    tones: ToneConfig = field(default_factory=ToneConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    keybinds: KeybindConfig = field(default_factory=KeybindConfig)
    output_mode: OutputMode = OutputMode.TYPE
    min_hold_to_process_s: float = 0.25
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        config = cls()

        if device := os.environ.get("DICTATE_AUDIO_DEVICE"):
            config.audio.device_id = int(device)

        if mode := os.environ.get("DICTATE_OUTPUT_MODE"):
            config.output_mode = OutputMode(mode.lower())

        if whisper_model := os.environ.get("DICTATE_WHISPER_MODEL"):
            if whisper_model.startswith("mlx-community/"):
                config.whisper.model = whisper_model
            else:
                import logging as _log

                _log.getLogger(__name__).warning(
                    "Ignoring DICTATE_WHISPER_MODEL=%s — only mlx-community/ repos allowed",
                    whisper_model,
                )

        if lang := os.environ.get("DICTATE_INPUT_LANGUAGE"):
            config.whisper.language = None if lang.lower() == "auto" else lang

        if lang := os.environ.get("DICTATE_OUTPUT_LANGUAGE"):
            config.llm.output_language = None if lang.lower() == "auto" else lang

        if verbose := os.environ.get("DICTATE_VERBOSE"):
            config.verbose = verbose.lower() in ("1", "true", "yes")

        # LLM cleanup
        if llm_enabled := os.environ.get("DICTATE_LLM_CLEANUP"):
            config.llm.enabled = llm_enabled.lower() in ("1", "true", "yes")

        # LLM model choice
        if llm_model := os.environ.get("DICTATE_LLM_MODEL"):
            try:
                config.llm.model_choice = LLMModel(llm_model.lower())
            except ValueError:
                logger.warning(
                    "Invalid DICTATE_LLM_MODEL=%r, using default %r",
                    llm_model,
                    config.llm.model_choice,
                )

        # LLM backend (local or api)
        if llm_backend := os.environ.get("DICTATE_LLM_BACKEND"):
            try:
                config.llm.backend = LLMBackend(llm_backend.lower())
            except ValueError:
                logger.warning(
                    "Invalid DICTATE_LLM_BACKEND=%r, using default %r",
                    llm_backend,
                    config.llm.backend,
                )

        # LLM API URL (for api backend)
        if api_url := os.environ.get("DICTATE_LLM_API_URL"):
            config.llm.api_url = api_url

        return config
