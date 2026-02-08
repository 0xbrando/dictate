"""Quality presets and preferences persistence for the menu bar app."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dictate.config import LLMBackend, LLMModel, STTEngine

logger = logging.getLogger(__name__)


# ── Hardware detection ─────────────────────────────────────────

def detect_chip() -> str:
    """Detect Apple Silicon chip family (e.g. 'M1', 'M2', 'M3', 'M4', 'M5', 'Ultra')."""
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=2,
        ).strip()
        # e.g. "Apple M2", "Apple M3 Pro", "Apple M4 Ultra"
        return raw.replace("Apple ", "")
    except Exception:
        return "Unknown"


def recommended_quality_preset() -> int:
    """Return the best quality preset index for this hardware.

    Presets: [0]=API, [1]=1.5B, [2]=3B, [3]=7B, [4]=14B
    """
    chip = detect_chip().lower()
    if "ultra" in chip or "max" in chip:
        return 2  # Fast - 3B (plenty fast on Ultra/Max)
    return 1  # Speedy - 1.5B (good default for all chips)

PREFS_DIR = Path.home() / "Library" / "Application Support" / "Dictate"
PREFS_FILE = PREFS_DIR / "preferences.json"
DICTIONARY_FILE = PREFS_DIR / "dictionary.json"

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
    backend: LLMBackend = LLMBackend.LOCAL
    description: str = ""


QUALITY_PRESETS: list[QualityPreset] = [
    QualityPreset(
        label="Smart (~250ms, 0 RAM)",
        llm_model=LLMModel.QWEN,
        backend=LLMBackend.API,
        description="Fast local for short, API server for long",
    ),
    QualityPreset(
        label="Speedy - 1.5B (~120ms, 1GB)",
        llm_model=LLMModel.QWEN_1_5B,
        description="Fast cleanup, great for any M chip",
    ),
    QualityPreset(
        label="Fast - 3B (~250ms, 2GB)",
        llm_model=LLMModel.QWEN,
        description="Quick cleanup, Qwen 3B",
    ),
    QualityPreset(
        label="Balanced - 7B (~350ms, 5GB)",
        llm_model=LLMModel.QWEN_7B,
        description="Better for long-form, Qwen 7B",
    ),
    QualityPreset(
        label="Quality - 14B (~500ms, 9GB)",
        llm_model=LLMModel.QWEN_14B,
        description="Best for rewriting, Qwen 14B",
    ),
]


@dataclass
class SoundPreset:
    label: str
    start_hz: int
    stop_hz: int
    style: str = "simple"


SOUND_PRESETS: list[SoundPreset] = [
    SoundPreset(label="Soft Pop", start_hz=880, stop_hz=660, style="soft_pop"),
    SoundPreset(label="Chime", start_hz=880, stop_hz=660, style="chime"),
    SoundPreset(label="Warm Harmonic", start_hz=880, stop_hz=660, style="warm"),
    SoundPreset(label="Subtle Click", start_hz=1000, stop_hz=800, style="click"),
    SoundPreset(label="Marimba", start_hz=880, stop_hz=660, style="marimba"),
    SoundPreset(label="Simple Beep", start_hz=880, stop_hz=440, style="simple"),
    SoundPreset(label="None", start_hz=0, stop_hz=0, style="simple"),
]


PTT_KEYS: list[tuple[str, str]] = [
    ("ctrl_l", "Left Control"),
    ("ctrl_r", "Right Control"),
    ("cmd_r", "Right Command"),
    ("alt_l", "Left Option"),
    ("alt_r", "Right Option"),
]

COMMAND_KEYS: list[tuple[str, str]] = [
    ("none", "Disabled"),
    ("alt_r", "Right Option"),
    ("alt_l", "Left Option"),
    ("cmd_r", "Right Command"),
    ("ctrl_r", "Right Control"),
]


@dataclass
class STTPreset:
    label: str
    engine: STTEngine
    model: str
    description: str = ""


STT_PRESETS: list[STTPreset] = [
    STTPreset(
        label="Whisper (default)",
        engine=STTEngine.WHISPER,
        model="mlx-community/whisper-large-v3-turbo",
        description="99+ languages",
    ),
    STTPreset(
        label="Parakeet (faster)",
        engine=STTEngine.PARAKEET,
        model="mlx-community/parakeet-tdt-0.6b-v3",
        description="pip install parakeet-mlx",
    ),
]


WRITING_STYLES: list[tuple[str, str, str]] = [
    ("clean", "Clean Up", "Fixes punctuation, keeps your words"),
    ("formal", "Formal", "Professional tone and grammar"),
    ("bullets", "Bullet Points", "Distills into key points"),
]


@dataclass
class Preferences:
    device_id: int | None = None
    quality_preset: int = 1  # index into QUALITY_PRESETS (default: Speedy 1.5B)
    stt_preset: int = 0  # index into STT_PRESETS (default: Whisper)
    input_language: str = "auto"
    output_language: str = "auto"
    llm_cleanup: bool = True
    sound_preset: int = 0  # index into SOUND_PRESETS (default: Soft Pop)
    writing_style: str = "clean"  # key into WRITING_STYLES
    ptt_key: str = "ctrl_l"  # key into PTT_KEYS
    command_key: str = "none"  # key into COMMAND_KEYS
    api_url: str = "http://localhost:8005/v1/chat/completions"
    advanced_mode: bool = False

    def save(self) -> None:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(PREFS_DIR, 0o700)
        data = asdict(self)
        data["_prefs_version"] = 3
        try:
            PREFS_FILE.write_text(json.dumps(data, indent=2))
            os.chmod(PREFS_FILE, 0o600)
        except OSError:
            logger.exception("Failed to save preferences")

    @classmethod
    def load(cls) -> Preferences:
        if not PREFS_FILE.exists():
            # First launch: auto-detect best settings for this hardware
            chip = detect_chip()
            preset = recommended_quality_preset()
            logger.info("First launch on %s — auto-selected quality preset %d", chip, preset)
            prefs = cls(quality_preset=preset)
            prefs.save()
            return prefs
        try:
            data = json.loads(PREFS_FILE.read_text())
            # Migrate preset indexes across versions:
            # v1: [0]=API, [1]=3B, [2]=7B, [3]=14B
            # v2: [0]=API, [1]=0.5B, [2]=1.5B, [3]=3B, [4]=7B, [5]=14B
            # v3: [0]=API, [1]=1.5B, [2]=3B, [3]=7B, [4]=14B  (0.5B removed)
            raw_preset = data.get("quality_preset", 1)
            version = data.get("_prefs_version", 1)
            if version == 1 and raw_preset >= 1:
                raw_preset += 1  # v1 1→2, 2→3, 3→4
            elif version == 2:
                if raw_preset <= 1:
                    raw_preset = max(0, raw_preset)  # API stays 0, 0.5B→1.5B(1)
                elif raw_preset >= 2:
                    raw_preset -= 1  # 2→1, 3→2, 4→3, 5→4
            return cls(
                device_id=data.get("device_id"),
                quality_preset=raw_preset,
                stt_preset=data.get("stt_preset", 0),
                input_language=data.get("input_language", "auto"),
                output_language=data.get("output_language", "auto"),
                llm_cleanup=data.get("llm_cleanup", True),
                sound_preset=data.get("sound_preset", 0),
                writing_style=data.get("writing_style", "clean"),
                ptt_key=data.get("ptt_key", "ctrl_l"),
                command_key=data.get("command_key", "none"),
                api_url=data.get("api_url", "http://localhost:8005/v1/chat/completions"),
                advanced_mode=data.get("advanced_mode", False),
            )
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load preferences, using defaults")
            return cls()

    @property
    def llm_model(self) -> LLMModel:
        idx = max(0, min(self.quality_preset, len(QUALITY_PRESETS) - 1))
        return QUALITY_PRESETS[idx].llm_model

    @property
    def backend(self) -> LLMBackend:
        idx = max(0, min(self.quality_preset, len(QUALITY_PRESETS) - 1))
        return QUALITY_PRESETS[idx].backend

    @property
    def stt_engine(self) -> STTEngine:
        idx = max(0, min(self.stt_preset, len(STT_PRESETS) - 1))
        return STT_PRESETS[idx].engine

    @property
    def stt_model(self) -> str:
        idx = max(0, min(self.stt_preset, len(STT_PRESETS) - 1))
        return STT_PRESETS[idx].model

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

    @staticmethod
    def _is_safe_api_url(url: str) -> bool:
        """Only allow localhost API URLs unless explicitly overridden."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            host = parsed.hostname or ""
            return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
        except Exception:
            return False

    @property
    def validated_api_url(self) -> str:
        if self._is_safe_api_url(self.api_url):
            return self.api_url
        if os.environ.get("DICTATE_ALLOW_REMOTE_API") == "1":
            logger.warning("Remote API URL allowed via DICTATE_ALLOW_REMOTE_API: %s", self.api_url)
            return self.api_url
        logger.warning("Blocked non-localhost API URL: %s (set DICTATE_ALLOW_REMOTE_API=1 to override)", self.api_url)
        return "http://localhost:8005/v1/chat/completions"

    @property
    def ptt_pynput_key(self) -> "Key":
        from pynput import keyboard
        key_map = {
            "ctrl_l": keyboard.Key.ctrl_l,
            "ctrl_r": keyboard.Key.ctrl_r,
            "cmd_r": keyboard.Key.cmd_r,
            "alt_l": keyboard.Key.alt_l,
            "alt_r": keyboard.Key.alt_r,
        }
        return key_map.get(self.ptt_key, keyboard.Key.ctrl_l)

    @property
    def command_pynput_key(self) -> "Key | None":
        if self.command_key == "none":
            return None
        from pynput import keyboard
        key_map = {
            "ctrl_l": keyboard.Key.ctrl_l,
            "ctrl_r": keyboard.Key.ctrl_r,
            "cmd_r": keyboard.Key.cmd_r,
            "alt_l": keyboard.Key.alt_l,
            "alt_r": keyboard.Key.alt_r,
        }
        return key_map.get(self.command_key)

    @staticmethod
    def load_dictionary() -> list[str]:
        if not DICTIONARY_FILE.exists():
            return []
        try:
            data = json.loads(DICTIONARY_FILE.read_text())
            if isinstance(data, list):
                return [str(w) for w in data]
            if isinstance(data, dict) and "words" in data:
                return [str(w) for w in data["words"]]
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load dictionary")
        return []

    @staticmethod
    def save_dictionary(words: list[str]) -> None:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        DICTIONARY_FILE.write_text(json.dumps({"words": words}, indent=2))
        os.chmod(DICTIONARY_FILE, 0o600)
