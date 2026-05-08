"""Speech-to-text transcription and text cleanup."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import selectors
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

# Suppress huggingface/tqdm progress bars (must be set before imports)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from scipy.io.wavfile import write as wav_write

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np

    from dictate.config import LLMConfig, WhisperConfig

logger = logging.getLogger(__name__)

API_TIMEOUT_SECONDS = 15
LOCAL_LLM_TIMEOUT_SECONDS = 10
MLX_UNAVAILABLE_HINT = (
    "Check macOS/Metal compatibility, disable LLM cleanup for raw ANE dictation, "
    "or point the API backend at a localhost LLM server."
)


def _private_tmp_dir() -> str:
    """Return a private temp directory for audio files (owner-only access)."""
    from pathlib import Path

    d = Path.home() / "Library" / "Application Support" / "Dictate" / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return str(d)


@contextlib.contextmanager
def _temp_wav_context(audio: "NDArray[np.int16]", sample_rate: int):
    """Context manager for creating and cleaning up temporary WAV files."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="dictate_", dir=_private_tmp_dir())
    os.close(fd)
    try:
        try:
            wav_write(path, sample_rate, audio)
        except OSError as e:
            # Clean up the temp file if write failed
            try:
                os.remove(path)
            except OSError:
                pass
            raise RuntimeError(f"Failed to save temporary WAV file (disk full?): {e}") from e
        yield path
    finally:
        try:
            os.remove(path)
        except OSError as e:
            logger.warning("Failed to remove temp file %s: %s", path, e)


class WhisperTranscriber:
    def __init__(self, config: "WhisperConfig") -> None:
        self._config = config
        self._model_loaded = False

    def load_model(self) -> None:
        if self._model_loaded:
            return

        print(f"   Whisper: {self._config.model}...", end=" ", flush=True)

        from dictate.mlx_check import is_mlx_available
        if not is_mlx_available():
            raise RuntimeError(
                "MLX cannot initialize Metal GPU on this system. "
                f"Whisper STT requires MLX. {MLX_UNAVAILABLE_HINT}"
            )

        try:
            import mlx_whisper
        except ImportError:
            raise ImportError(
                "mlx-whisper is required for the Whisper engine. "
                "Install it with: pip install mlx-whisper"
            )
        import numpy as np

        silent_audio = np.zeros(16000, dtype=np.int16)

        with _temp_wav_context(silent_audio, 16000) as wav_path:
            mlx_whisper.transcribe(
                wav_path,
                path_or_hf_repo=self._config.model,
                language=self._config.language,
            )
            self._model_loaded = True
            print("✓")

    def transcribe(
        self,
        audio: "NDArray[np.int16]",
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        with _temp_wav_context(audio, sample_rate) as wav_path:
            import mlx_whisper

            if not self._model_loaded:
                logger.info("Lazy-loading Whisper model: %s", self._config.model)
                # Flag set after first successful transcribe() call below

            transcribe_language = language if language is not None else self._config.language

            result = mlx_whisper.transcribe(
                wav_path,
                path_or_hf_repo=self._config.model,
                language=transcribe_language,
            )
            self._model_loaded = True
            text = result.get("text", "")
            return str(text) if isinstance(text, str) else ""


def _dedup_transcription(text: str) -> str:
    """Remove repeated phrases from transcription output.

    TDT models (like Parakeet) can sometimes produce the same phrase twice.
    Detects if the second half of the text repeats the first half.
    """
    words = text.split()
    n = len(words)
    if n < 4:
        return text

    # Check if the text is a repeated phrase (exact duplicate)
    half = n // 2
    first_half = " ".join(words[:half])
    second_half = " ".join(words[half : half * 2])
    if first_half.lower() == second_half.lower():
        logger.info("Deduped repeated transcription: %d words → %d", n, half)
        return " ".join(words[:half])

    # Check for off-by-one repetitions (odd word count)
    if n >= 5:
        for split_at in (half, half + 1):
            if split_at >= n:
                continue
            a = " ".join(words[:split_at]).lower()
            b = " ".join(words[split_at:]).lower()
            if a == b:
                logger.info("Deduped repeated transcription: %d words → %d", n, split_at)
                return " ".join(words[:split_at])

    return text


class ParakeetTranscriber:
    """Speech-to-text using NVIDIA Parakeet TDT via MLX — much faster than Whisper."""

    def __init__(self, config: "WhisperConfig") -> None:
        self._config = config
        self._model = None

    def load_model(self) -> None:
        if self._model is not None:
            return

        from dictate.mlx_check import is_mlx_available
        if not is_mlx_available():
            raise RuntimeError(
                "MLX cannot initialize Metal GPU on this system. "
                f"Parakeet STT requires MLX. {MLX_UNAVAILABLE_HINT}"
            )

        try:
            from parakeet_mlx import from_pretrained
        except ImportError:
            raise ImportError(
                "parakeet-mlx is required for the Parakeet engine. "
                "Install it with: pip install parakeet-mlx"
            )

        model_name = self._config.model
        print(f"   Parakeet: {model_name}...", end=" ", flush=True)
        self._model = from_pretrained(model_name)
        print("✓")

    def transcribe(
        self,
        audio: "NDArray[np.int16]",
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        if self._model is None:
            self.load_model()

        with _temp_wav_context(audio, sample_rate) as path:
            result = self._model.transcribe(path)
            text = getattr(result, "text", "")
            text = str(text).strip() if isinstance(text, str) else ""
            return _dedup_transcription(text)


class Qwen3ASRTranscriber:
    """Speech-to-text using Qwen3-ASR via mlx-audio — 52 languages, faster than Whisper."""

    def __init__(self, config: "WhisperConfig") -> None:
        self._config = config
        self._model = None

    @staticmethod
    def is_available() -> bool:
        """Check if mlx-audio is installed with STT support."""
        try:
            from mlx_audio.stt.utils import load  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self) -> None:
        if self._model is not None:
            return

        from dictate.mlx_check import is_mlx_available
        if not is_mlx_available():
            raise RuntimeError(
                "MLX cannot initialize Metal GPU on this system. "
                f"Qwen3-ASR requires MLX. {MLX_UNAVAILABLE_HINT}"
            )

        try:
            from mlx_audio.stt.utils import load
        except ImportError:
            raise ImportError(
                "mlx-audio is required for the Qwen3-ASR engine. "
                "Install it with: pip install mlx-audio"
            )

        model_name = self._config.model
        print(f"   Qwen3-ASR: {model_name}...", end=" ", flush=True)
        self._model = load(model_name)
        print("✓")

    def transcribe(
        self,
        audio: "NDArray[np.int16]",
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        if self._model is None:
            self.load_model()

        with _temp_wav_context(audio, sample_rate) as path:
            kwargs: dict = {}
            if language:
                # Map ISO codes to full names for Qwen3-ASR
                lang_map = {
                    "en": "English", "zh": "Chinese", "ja": "Japanese",
                    "ko": "Korean", "es": "Spanish", "fr": "French",
                    "de": "German", "it": "Italian", "pt": "Portuguese",
                    "nl": "Dutch", "ru": "Russian", "pl": "Polish",
                }
                kwargs["language"] = lang_map.get(language, language)
            result = self._model.generate(path, **kwargs)
            text = getattr(result, "text", "")
            text = str(text).strip() if isinstance(text, str) else ""
            return _dedup_transcription(text)


class ANETranscriber:
    """Speech-to-text using Apple Neural Engine via the dictate-stt Swift CLI."""

    _BINARY_NAME = "dictate-stt"

    def __init__(self, config: "WhisperConfig", binary_path: str | None = None) -> None:
        self._config = config
        self._binary = binary_path or self._find_binary()
        self._model_loaded = False
        self._server: subprocess.Popen[str] | None = None

    @staticmethod
    def _find_binary() -> str | None:
        """Locate the dictate-stt binary in PATH or known build locations."""
        import shutil

        found = shutil.which("dictate-stt")
        if found:
            return found

        # Check relative to this file: app bundle and dev build
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base, "MacOS", "dictate-stt"),
            os.path.join(base, "swift-stt", ".build", "release", "dictate-stt"),
            os.path.join(base, "..", "swift-stt", ".build", "release", "dictate-stt"),
        ]
        for path in candidates:
            path = os.path.realpath(path)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    @staticmethod
    def is_available() -> bool:
        """Check if ANE transcription is available (binary exists + macOS 14+)."""
        import platform

        if platform.system() != "Darwin":
            return False

        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            try:
                if int(mac_ver.split(".")[0]) < 14:
                    return False
            except (ValueError, IndexError):
                return False

        return bool(ANETranscriber._find_binary())

    @staticmethod
    def _readline_with_timeout(stream, timeout: float) -> str:
        selector = selectors.DefaultSelector()
        try:
            selector.register(stream, selectors.EVENT_READ)
            events = selector.select(timeout)
            if not events:
                raise subprocess.TimeoutExpired("dictate-stt serve", timeout)
            return stream.readline()
        finally:
            selector.close()

    def _stop_server(self) -> None:
        proc = self._server
        self._server = None
        if not proc:
            return
        with contextlib.suppress(Exception):
            if proc.stdin:
                proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()

    def load_model(self) -> None:
        """Start the persistent Swift helper and load CoreML models once."""
        if self._model_loaded:
            return

        if not self._binary:
            raise RuntimeError(
                "dictate-stt binary not found. Build it with: "
                "cd swift-stt && swift build -c release"
            )

        print(f"   ANE STT: {self._binary}...", end=" ", flush=True)
        try:
            self._server = subprocess.Popen(
                [self._binary, "serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            if not self._server.stdout:
                raise RuntimeError("dictate-stt serve did not expose stdout")

            line = self._readline_with_timeout(self._server.stdout, 120)
            if not line:
                stderr = self._server.stderr.read() if self._server.stderr else ""
                raise RuntimeError(f"dictate-stt serve exited early: {stderr.strip()}")

            ready = json.loads(line)
            if not ready.get("ready"):
                raise RuntimeError(f"dictate-stt serve did not become ready: {line.strip()}")
        except FileNotFoundError:
            raise RuntimeError(f"dictate-stt not found at: {self._binary}")
        except subprocess.TimeoutExpired:
            self._stop_server()
            raise RuntimeError("dictate-stt model load timed out")
        except json.JSONDecodeError as e:
            self._stop_server()
            raise RuntimeError(f"dictate-stt serve returned invalid JSON: {e}") from e
        except Exception:
            self._stop_server()
            raise

        self._model_loaded = True
        print("\u2713")

    def transcribe(
        self,
        audio: "NDArray[np.int16]",
        sample_rate: int,
        language: str | None = None,  # unused — FluidAudio handles language internally
    ) -> str:
        if not self._binary:
            raise RuntimeError("dictate-stt binary not found")

        if not self._model_loaded or not self._server or self._server.poll() is not None:
            self._model_loaded = False
            self.load_model()

        # ANE requires >= 1 second of audio. Pad short clips with silence.
        min_samples = sample_rate  # 1 second
        if len(audio) < min_samples:
            import numpy as np
            audio = np.pad(audio, (0, min_samples - len(audio)), mode="constant")

        with _temp_wav_context(audio, sample_rate) as wav_path:
            try:
                if not self._server or not self._server.stdin or not self._server.stdout:
                    raise RuntimeError("dictate-stt serve is not running")
                self._server.stdin.write(json.dumps({"path": wav_path}) + "\n")
                self._server.stdin.flush()
                line = self._readline_with_timeout(self._server.stdout, 30)
            except subprocess.TimeoutExpired:
                logger.error("ANE transcription timed out after 30s")
                return ""
            except (BrokenPipeError, OSError, RuntimeError) as e:
                logger.error("ANE helper failed: %s", e)
                self._stop_server()
                self._model_loaded = False
                return ""

            try:
                result = json.loads(line)
                if "error" in result:
                    logger.error("ANE transcription failed: %s", result["error"])
                    return ""
                text = result.get("text", "")
                logger.info("ANE transcribed in %dms", result.get("duration_ms", 0))
                return _dedup_transcription(str(text).strip())
            except json.JSONDecodeError as e:
                logger.error("Failed to parse ANE output: %s", e)
                return line.strip()

    def __del__(self) -> None:
        self._stop_server()


# ── Shared postprocessing ────────────────────────────────────────


def _postprocess(text: str) -> str:
    """Clean up LLM output: strip special tokens, preambles, quotes."""
    special_tokens = [
        "<|end|>",
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "</s>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    # Strip <think>...</think> blocks from reasoning models (Qwen3, DeepSeek R1)
    # Handle both closed and incomplete think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also strip incomplete think blocks (when max_tokens cuts off generation)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    text_lower = text.lower()

    preambles = [
        "Sure, here's the corrected text:",
        "Sure, here is the corrected text:",
        "Sure, here's the text:",
        "Sure, here is the text:",
        "Sure, here you go:",
        "Sure!",
        "Sure:",
        "Sure,",
        "Here's the corrected text:",
        "Here is the corrected text:",
        "Here's the formatted text:",
        "Here is the formatted text:",
        "Here's the text:",
        "Here is the text:",
        "Here you go:",
        "Here it is:",
        "Corrected text:",
        "Corrected:",
        "Fixed text:",
        "Fixed:",
        "Formatted text:",
        "Formatted:",
        "The corrected text is:",
        "The corrected text:",
        "The text:",
        "I've corrected the text:",
        "I have corrected the text:",
        "I fixed the text:",
        "Of course!",
        "Of course:",
        "Of course,",
        "Certainly!",
        "Certainly:",
        "Certainly,",
        "Output:",
        "Result:",
        "Answer:",
    ]

    for preamble in preambles:
        if text_lower.startswith(preamble.lower()):
            text = text[len(preamble) :].strip()
            text_lower = text.lower()

    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    text = text.lstrip("\n")

    lines = text.split("\n")
    if not lines:
        return text

    first_line = lines[0].strip()
    if len(lines) > 1 and lines[1].strip() == first_line:
        logger.warning("Detected repetition in LLM output, truncating")
        return first_line

    return text


# ── Text cleaners ────────────────────────────────────────────────


class TextCleaner:
    """Cleans up transcription text using a local MLX model."""

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._last_cleanup_failed = False

    def load_model(self) -> None:
        if self._model is not None:
            return

        from dictate.mlx_check import is_mlx_available
        if not is_mlx_available():
            raise RuntimeError(
                "MLX cannot initialize Metal GPU on this system. "
                f"Local LLM cleanup requires MLX. {MLX_UNAVAILABLE_HINT}"
            )

        from mlx_lm import load

        print(f"   LLM: {self._config.model}...", end=" ", flush=True)
        self._model, self._tokenizer = load(self._config.model)
        print("✓")

    def cleanup(self, text: str, output_language: str | None = None) -> str:
        if not self._config.enabled:
            return text

        if self._model is None or self._tokenizer is None:
            self.load_model()

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        system_prompt = self._config.get_system_prompt(output_language)
        # Qwen3/Qwen3.5 reasoning models burn max_tokens on <think> blocks before
        # producing output. /no_think disables this for simple cleanup tasks.
        model_lower = self._config.model.lower()
        user_content = f"/no_think\n{text}" if ("qwen3" in model_lower or "qwen3.5" in model_lower) else text
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_words = len(text.split())
        max_tokens = min(self._config.max_tokens, max(50, input_words * 3))
        sampler = make_sampler(temp=self._config.temperature)

        result_box: list[str] = []

        def _run_generate() -> None:
            result_box.append(
                generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                )
            )

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
        thread.join(timeout=LOCAL_LLM_TIMEOUT_SECONDS)

        if not result_box:
            logger.warning(
                "Local LLM timed out after %ds, returning raw text", LOCAL_LLM_TIMEOUT_SECONDS
            )
            self._last_cleanup_failed = True
            return text

        result = result_box[0]
        logger.debug("LLM raw result: %r", result[:100] if len(result) > 100 else result)
        return _postprocess(result.strip())


class APITextCleaner:
    """Cleans up transcription text via an OpenAI-compatible API server."""

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._last_cleanup_failed = False

    def load_model(self) -> None:
        """No model to load — verify server is reachable."""
        url = self._config.api_url.replace("/chat/completions", "").rstrip("/")
        try:
            req = urllib.request.Request(f"{url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                pass
            print(f"   API: {self._config.api_url} ✓")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            logger.info("API server not reachable at %s: %s", url, e)
            print(f"   API: {self._config.api_url} (will retry on first use)")
        except Exception as e:
            logger.warning("API server check failed at %s: %s", url, e)
            print(f"   API: {self._config.api_url} (error: {e})")

    def cleanup(self, text: str, output_language: str | None = None) -> str:
        if not self._config.enabled:
            return text

        system_prompt = self._config.get_system_prompt(output_language)
        input_words = len(text.split())
        max_tokens = min(self._config.max_tokens, max(50, input_words * 3))

        payload = json.dumps(
            {
                "model": "default",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "max_tokens": max_tokens,
                "temperature": self._config.temperature,
            }
        ).encode()

        req = urllib.request.Request(
            self._config.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
                result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"].strip()
            return _postprocess(content)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            logger.warning("API %s: %s, retrying...", type(e).__name__, e)
            import time

            time.sleep(0.5)
            try:
                with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
                    result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"].strip()
                return _postprocess(content)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e2:
                logger.error("API retry failed: %s, returning raw text", e2)
                self._last_cleanup_failed = True
                return text
            except (json.JSONDecodeError, KeyError, IndexError) as e2:
                logger.error("API returned unexpected response on retry: %s", e2)
                self._last_cleanup_failed = True
                return text
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error("API returned unexpected response: %s — check server config", e)
            self._last_cleanup_failed = True
            return text
        except Exception as e:
            logger.exception("API cleanup failed (%s: %s), returning raw text", type(e).__name__, e)
            self._last_cleanup_failed = True
            return text


# ── Smart skip heuristic ──────────────────────────────────────────

_FILLER_STARTS = (
    "um ",
    "uh ",
    "er ",
    "ah ",
    "like ",
    "you know",
    "basically ",
    "i mean ",
    "so ",
    "well ",
)


def _looks_clean(text: str) -> bool:
    """Check if a short transcription is clean enough to skip LLM cleanup.

    Only triggers for short utterances (<=8 words) that already have
    proper capitalization and punctuation. Longer text always goes
    through the LLM because compound sentences need punctuation fixes.
    """
    words = text.split()
    if not words or len(words) > 8:
        return False

    first_char = text[0]
    if not (first_char.isupper() or first_char.isdigit() or first_char in '"('):
        return False

    lower = text.lower()
    for filler in _FILLER_STARTS:
        if lower.startswith(filler):
            return False

    # 4+ words need ending punctuation to look "clean"
    if len(words) >= 4 and text[-1] not in ".!?,;:":
        return False

    return True


# ── Pipeline ─────────────────────────────────────────────────────

SMART_ROUTING_THRESHOLD = 15  # words — short messages use fast local model


DEDUP_WINDOW_SECONDS = 15.0


class TranscriptionPipeline:
    def __init__(
        self,
        whisper_config: "WhisperConfig",
        llm_config: "LLMConfig",
    ) -> None:
        from dictate.config import LLMBackend, STTEngine

        if whisper_config.engine == STTEngine.QWEN3_ASR:
            if Qwen3ASRTranscriber.is_available():
                self._whisper: WhisperTranscriber | ParakeetTranscriber | ANETranscriber | Qwen3ASRTranscriber = (
                    Qwen3ASRTranscriber(whisper_config)
                )
            else:
                logger.warning("mlx-audio not installed, falling back to Parakeet MLX")
                self._whisper = ParakeetTranscriber(whisper_config)
        elif whisper_config.engine == STTEngine.ANE:
            if ANETranscriber.is_available():
                self._whisper = ANETranscriber(whisper_config)
            else:
                logger.warning("ANE binary not found, falling back to Parakeet MLX")
                self._whisper = ParakeetTranscriber(whisper_config)
        elif whisper_config.engine == STTEngine.PARAKEET:
            self._whisper = ParakeetTranscriber(whisper_config)
        else:
            self._whisper = WhisperTranscriber(whisper_config)
        if llm_config.backend == LLMBackend.API:
            self._cleaner: TextCleaner | APITextCleaner = APITextCleaner(llm_config)
        else:
            self._cleaner = TextCleaner(llm_config)
        self._fast_cleaner: TextCleaner | None = self._create_fast_cleaner(llm_config)
        self._llm_config = llm_config
        self._sample_rate = 16_000
        self._last_output: str = ""
        self._last_output_time: float = 0.0
        self.last_cleanup_failed: bool = False

    @staticmethod
    def _create_fast_cleaner(llm_config: "LLMConfig") -> "TextCleaner | None":
        """Create a fast local cleaner for smart routing (API mode only)."""
        from dictate.config import LLMBackend, LLMConfig, LLMModel, is_model_cached

        if llm_config.backend != LLMBackend.API:
            return None

        # Pick the fastest cached local model (prefer instruction models over reasoning)
        for model in [
            LLMModel.QWEN25_1_5B,
            LLMModel.QWEN35_2B,
            LLMModel.QWEN_3B,
            LLMModel.QWEN3_0_6B,
            LLMModel.QWEN3_1_7B,
        ]:
            if is_model_cached(model.hf_repo):
                fast_config = LLMConfig(
                    enabled=llm_config.enabled,
                    model_choice=model,
                    max_tokens=llm_config.max_tokens,
                    temperature=llm_config.temperature,
                    output_language=llm_config.output_language,
                    writing_style=llm_config.writing_style,
                    dictionary=llm_config.dictionary,
                )
                logger.info("Smart routing: %s for short, API for long", model.value)
                return TextCleaner(fast_config)

        return None

    def set_sample_rate(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate

    def _is_duplicate(self, text: str) -> bool:
        """Check if text matches the last output within the dedup window."""
        import time as _time

        now = _time.time()
        if (
            self._last_output
            and (now - self._last_output_time) < DEDUP_WINDOW_SECONDS
            and text.lower().strip() == self._last_output.lower().strip()
        ):
            logger.info("Skipped duplicate output (%.1fs ago)", now - self._last_output_time)
            return True
        self._last_output = text
        self._last_output_time = now
        return False

    def preload_models(self, on_progress=None) -> None:
        """Preload all models with detailed progress reporting."""
        from dictate.config import is_model_cached
        from dictate.model_download import download_model

        # ANE uses its own model management (CoreML via Swift binary)
        is_ane = isinstance(self._whisper, ANETranscriber)
        engine_name = (
            "ANE" if is_ane
            else "Whisper" if isinstance(self._whisper, WhisperTranscriber)
            else "Qwen3-ASR" if isinstance(self._whisper, Qwen3ASRTranscriber)
            else "Parakeet"
        )

        if not is_ane:
            # Download STT model if needed with progress
            whisper_cached = is_model_cached(self._whisper._config.model)
            if not whisper_cached:
                if on_progress:
                    on_progress(f"Downloading {engine_name} model...")

                def whisper_progress(percent: float) -> None:
                    if on_progress:
                        on_progress(f"Downloading {engine_name} ({int(percent)}%)...")

                try:
                    download_model(self._whisper._config.model, progress_callback=whisper_progress)
                except Exception:
                    logger.exception("Failed to download STT model")
                    raise

        if on_progress:
            label = "Checking ANE binary..." if is_ane else f"Loading {engine_name}..."
            on_progress(label)
        self._whisper.load_model()

        # Load fast cleaner if configured
        if self._fast_cleaner:
            if on_progress:
                on_progress("Loading fast local model...")
            self._fast_cleaner.load_model()

        # Get LLM model info
        llm_model = getattr(self._cleaner, "_config", None)
        llm_repo = llm_model.model if llm_model else ""

        # Download LLM if needed with progress
        if llm_repo and not isinstance(self._cleaner, APITextCleaner):
            llm_cached = is_model_cached(llm_repo)
            if not llm_cached:
                if on_progress:
                    on_progress("Downloading LLM model...")

                def llm_progress(percent: float) -> None:
                    if on_progress:
                        on_progress(f"Downloading LLM ({int(percent)}%)...")

                try:
                    download_model(llm_repo, progress_callback=llm_progress)
                except Exception:
                    logger.exception("Failed to download LLM model")
                    raise

        # Load the main cleaner
        if isinstance(self._cleaner, APITextCleaner):
            if on_progress:
                on_progress("Connecting to API server...")
        else:
            if on_progress:
                on_progress("Loading LLM...")
        self._cleaner.load_model()

    def _pick_cleaner(self, word_count: int) -> "TextCleaner | APITextCleaner":
        """Route short messages to the fast local model, long ones to API."""
        if self._fast_cleaner and word_count <= SMART_ROUTING_THRESHOLD:
            return self._fast_cleaner
        return self._cleaner

    def process(
        self,
        audio: "NDArray[np.int16]",
        input_language: str | None = None,
        output_language: str | None = None,
    ) -> str | None:
        import time

        duration_s = len(audio) / self._sample_rate
        logger.info("Processing %.1fs of audio...", duration_s)

        t0 = time.time()
        raw_text = self._whisper.transcribe(
            audio, self._sample_rate, language=input_language
        ).strip()
        t1 = time.time()

        if not raw_text:
            logger.info("No speech detected")
            return None

        word_count = len(raw_text.split())
        logger.info("Transcribed in %.1fs (%d words)", t1 - t0, word_count)
        logger.debug("Transcription text: %s...", raw_text[:80] if len(raw_text) > 80 else raw_text)

        # Smart skip: if LLM is enabled but text already looks clean,
        # skip the expensive LLM round-trip. Translation mode always
        # runs through LLM since it needs to translate.
        needs_translation = output_language is not None or (
            self._llm_config.output_language is not None
        )
        if (
            self._llm_config.enabled
            and not needs_translation
            and self._llm_config.writing_style == "clean"
            and _looks_clean(raw_text)
        ):
            logger.info("Skipped LLM (clean transcription, %d words)", word_count)
            if self._is_duplicate(raw_text):
                return None
            return raw_text

        cleaner = self._pick_cleaner(word_count)
        route = "local" if cleaner is self._fast_cleaner else "API"

        t2 = time.time()
        cleaned_text = cleaner.cleanup(raw_text, output_language=output_language).strip()
        t3 = time.time()

        # Surface cleanup failures to UI
        failed = getattr(cleaner, "_last_cleanup_failed", False)
        if failed:
            cleaner._last_cleanup_failed = False
        self.last_cleanup_failed = failed

        if not cleaned_text:
            logger.info("Cleanup returned empty, using raw transcription")
            cleaned_text = raw_text

        if cleaned_text != raw_text:
            logger.info(
                "Cleaned via %s in %.0fms (%d words)",
                route,
                (t3 - t2) * 1000,
                len(cleaned_text.split()),
            )
            logger.debug(
                "Cleaned text: %s...", cleaned_text[:80] if len(cleaned_text) > 80 else cleaned_text
            )
        else:
            logger.info("No changes needed via %s (%.0fms)", route, (t3 - t2) * 1000)

        if self._is_duplicate(cleaned_text):
            return None

        return cleaned_text
