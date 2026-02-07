"""Quality presets and preferences persistence for the menu bar app."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dictate.config import LLMModel

logger = logging.getLogger(__name__)

PREFS_DIR = Path.home() / "Library" / "Application Support" / "Dictate"
PREFS_FILE = PREFS_DIR / "preferences.json"

INPUT_LANGUAGES = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("pl", "Polish"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("ja", "Japanese"),
    ("zh", "Chinese"),
    ("ko", "Korean"),
    ("ru", "Russian"),
]

OUTPUT_LANGUAGES = [
    ("auto", "Same as input"),
    ("en", "English"),
    ("pl", "Polish"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("ja", "Japanese"),
    ("zh", "Chinese"),
    ("ko", "Korean"),
    ("ru", "Russian"),
]


@dataclass
class QualityPreset:
    label: str
    llm_model: LLMModel
    description: str


QUALITY_PRESETS: list[QualityPreset] = [
    QualityPreset(
        label="Speed (3B)",
        llm_model=LLMModel.QWEN,
        description="Fastest — Qwen 3B",
    ),
    QualityPreset(
        label="Balanced (7B)",
        llm_model=LLMModel.QWEN_7B,
        description="Better cleanup — Qwen 7B",
    ),
    QualityPreset(
        label="Quality (14B)",
        llm_model=LLMModel.QWEN_14B,
        description="Best accuracy — Qwen 14B",
    ),
]


@dataclass
class SoundPreset:
    label: str
    start_hz: int
    stop_hz: int


SOUND_PRESETS: list[SoundPreset] = [
    SoundPreset(label="Default (880/440 Hz)", start_hz=880, stop_hz=440),
    SoundPreset(label="Soft (660/330 Hz)", start_hz=660, stop_hz=330),
    SoundPreset(label="High (1320/660 Hz)", start_hz=1320, stop_hz=660),
    SoundPreset(label="Click (1000/500 Hz)", start_hz=1000, stop_hz=500),
    SoundPreset(label="None", start_hz=0, stop_hz=0),
]


@dataclass
class Preferences:
    device_id: int | None = None
    quality_preset: int = 0  # index into QUALITY_PRESETS
    input_language: str = "auto"
    output_language: str = "auto"
    llm_cleanup: bool = True
    sound_preset: int = 0  # index into SOUND_PRESETS

    def save(self) -> None:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        try:
            PREFS_FILE.write_text(json.dumps(data, indent=2))
        except OSError:
            logger.exception("Failed to save preferences")

    @classmethod
    def load(cls) -> Preferences:
        if not PREFS_FILE.exists():
            return cls()
        try:
            data = json.loads(PREFS_FILE.read_text())
            return cls(
                device_id=data.get("device_id"),
                quality_preset=data.get("quality_preset", 0),
                input_language=data.get("input_language", "auto"),
                output_language=data.get("output_language", "auto"),
                llm_cleanup=data.get("llm_cleanup", True),
                sound_preset=data.get("sound_preset", 0),
            )
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load preferences, using defaults")
            return cls()

    @property
    def llm_model(self) -> LLMModel:
        idx = max(0, min(self.quality_preset, len(QUALITY_PRESETS) - 1))
        return QUALITY_PRESETS[idx].llm_model

    @property
    def whisper_language(self) -> str | None:
        return None if self.input_language == "auto" else self.input_language

    @property
    def llm_output_language(self) -> str | None:
        return None if self.output_language == "auto" else self.output_language

    @property
    def sound(self) -> SoundPreset:
        idx = max(0, min(self.sound_preset, len(SOUND_PRESETS) - 1))
        return SOUND_PRESETS[idx]
