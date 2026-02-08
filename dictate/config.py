"""Configuration for the Dictate application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynput.keyboard import Key

from pynput import keyboard


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


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    block_ms: int = 30
    device_id: int | None = None

    @property
    def block_size(self) -> int:
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
    engine: STTEngine = STTEngine.WHISPER


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
    max_tokens: int = 300
    temperature: float = 0.0
    output_language: str | None = None
    writing_style: str = "clean"
    dictionary: list[str] | None = None

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
            config.whisper.model = whisper_model

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
                pass  # Keep default if invalid value

        # LLM backend (local or api)
        if llm_backend := os.environ.get("DICTATE_LLM_BACKEND"):
            try:
                config.llm.backend = LLMBackend(llm_backend.lower())
            except ValueError:
                pass

        # LLM API URL (for api backend)
        if api_url := os.environ.get("DICTATE_LLM_API_URL"):
            config.llm.api_url = api_url

        return config