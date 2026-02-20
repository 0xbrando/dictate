"""Speech-to-text transcription and text cleanup."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
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


@contextlib.contextmanager
def _temp_wav_context(audio: "NDArray[np.int16]", sample_rate: int):
    """Context manager for creating and cleaning up temporary WAV files."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="dictate_")
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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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

    def load_model(self) -> None:
        if self._model is not None:
            return

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_words = len(text.split())
        max_tokens = min(self._config.max_tokens, max(50, input_words * 3))
        sampler = make_sampler(temp=self._config.temperature)

        result = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
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

        if whisper_config.engine == STTEngine.PARAKEET:
            self._whisper: WhisperTranscriber | ParakeetTranscriber = ParakeetTranscriber(
                whisper_config
            )
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

        # Pick the fastest cached local model (Qwen3 first - newer architecture, ~20% faster)
        for model in [
            LLMModel.QWEN3_0_6B,
            LLMModel.QWEN3_1_5B,
            LLMModel.QWEN_1_5B,
            LLMModel.QWEN,
            LLMModel.QWEN_7B,
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

        # Download Whisper if needed with progress
        whisper_cached = is_model_cached(self._whisper._config.model)
        if not whisper_cached:
            if on_progress:
                on_progress("Downloading Whisper model...")

            def whisper_progress(percent: float) -> None:
                if on_progress:
                    on_progress(f"Downloading Whisper ({int(percent)}%)...")

            try:
                download_model(self._whisper._config.model, progress_callback=whisper_progress)
            except Exception:
                logger.exception("Failed to download Whisper model")
                raise

        if on_progress:
            on_progress("Loading Whisper...")
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

        # Raw mode: bypass LLM cleanup entirely, return transcription as-is
        if self._llm_config.writing_style == "raw":
            logger.info("Raw mode — skipping LLM cleanup")
            if self._is_duplicate(raw_text):
                return None
            return raw_text

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
            logger.info("Cleanup failed")
            return None

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
