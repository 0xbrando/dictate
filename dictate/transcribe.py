"""Speech-to-text transcription and text cleanup."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.request
from typing import TYPE_CHECKING

# Suppress huggingface/tqdm progress bars (must be set before imports)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import mlx_whisper
from scipy.io.wavfile import write as wav_write

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np

    from dictate.config import LLMConfig, WhisperConfig

logger = logging.getLogger(__name__)

API_TIMEOUT_SECONDS = 15


class WhisperTranscriber:
    def __init__(self, config: "WhisperConfig") -> None:
        self._config = config
        self._model_loaded = False

    def load_model(self) -> None:
        if self._model_loaded:
            return

        print(f"   Whisper: {self._config.model}...", end=" ", flush=True)

        import numpy as np
        silent_audio = np.zeros(16000, dtype=np.int16)
        wav_path = self._save_temp_wav(silent_audio, 16000)

        try:
            mlx_whisper.transcribe(
                wav_path,
                path_or_hf_repo=self._config.model,
                language=self._config.language,
            )
            self._model_loaded = True
            print("✓")
        finally:
            self._cleanup_temp_file(wav_path)

    def transcribe(
        self,
        audio: "NDArray[np.int16]",
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        wav_path = self._save_temp_wav(audio, sample_rate)
        try:
            if not self._model_loaded:
                logger.info("Loading Whisper model: %s", self._config.model)
                self._model_loaded = True

            transcribe_language = language if language is not None else self._config.language

            result = mlx_whisper.transcribe(
                wav_path,
                path_or_hf_repo=self._config.model,
                language=transcribe_language,
            )
            text = result.get("text", "")
            return str(text) if isinstance(text, str) else ""
        finally:
            self._cleanup_temp_file(wav_path)

    def _save_temp_wav(
        self,
        audio: "NDArray[np.int16]",
        sample_rate: int,
    ) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="dictate_")
        os.close(fd)
        wav_write(path, sample_rate, audio)
        return path

    def _cleanup_temp_file(self, path: str) -> None:
        try:
            os.remove(path)
        except OSError as e:
            logger.warning("Failed to remove temp file %s: %s", path, e)


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
            text = text[len(preamble):].strip()
            text_lower = text.lower()

    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    text = text.lstrip('\n')

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
            messages, tokenize=False, add_generation_prompt=True,
        )

        input_words = len(text.split())
        max_tokens = min(self._config.max_tokens, max(50, input_words * 3))
        sampler = make_sampler(temp=self._config.temperature)

        result = generate(
            self._model, self._tokenizer,
            prompt=prompt, max_tokens=max_tokens, sampler=sampler,
        )
        logger.debug("LLM raw result: %r", result)
        return _postprocess(result.strip())


class APITextCleaner:
    """Cleans up transcription text via an OpenAI-compatible API server."""

    def __init__(self, config: "LLMConfig") -> None:
        self._config = config

    def load_model(self) -> None:
        """No model to load — verify server is reachable."""
        url = self._config.api_url.replace("/chat/completions", "").rstrip("/")
        try:
            req = urllib.request.Request(f"{url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                pass
            print(f"   API: {self._config.api_url} ✓")
        except Exception:
            print(f"   API: {self._config.api_url} (will retry on first use)")

    def cleanup(self, text: str, output_language: str | None = None) -> str:
        if not self._config.enabled:
            return text

        system_prompt = self._config.get_system_prompt(output_language)
        input_words = len(text.split())
        max_tokens = min(self._config.max_tokens, max(50, input_words * 3))

        payload = json.dumps({
            "model": "default",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": max_tokens,
            "temperature": self._config.temperature,
        }).encode()

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
        except Exception:
            logger.exception("API cleanup failed, returning raw text")
            return text


# ── Pipeline ─────────────────────────────────────────────────────


class TranscriptionPipeline:
    def __init__(
        self,
        whisper_config: "WhisperConfig",
        llm_config: "LLMConfig",
    ) -> None:
        from dictate.config import LLMBackend

        self._whisper = WhisperTranscriber(whisper_config)
        if llm_config.backend == LLMBackend.API:
            self._cleaner: TextCleaner | APITextCleaner = APITextCleaner(llm_config)
        else:
            self._cleaner = TextCleaner(llm_config)
        self._sample_rate = 16_000

    def set_sample_rate(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate

    def preload_models(self) -> None:
        self._whisper.load_model()
        self._cleaner.load_model()

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

        logger.info("Transcribed in %.1fs (%d words)", t1 - t0, len(raw_text.split()))
        logger.debug("Transcription text: %s", raw_text)

        t2 = time.time()
        cleaned_text = self._cleaner.cleanup(raw_text, output_language=output_language).strip()
        t3 = time.time()

        if not cleaned_text:
            logger.info("Cleanup failed")
            return None

        if cleaned_text != raw_text:
            logger.info("Cleaned in %.0fms (%d words)", (t3 - t2) * 1000, len(cleaned_text.split()))
            logger.debug("Cleaned text: %s", cleaned_text)
        else:
            logger.info("No changes needed (%.0fms)", (t3 - t2) * 1000)

        return cleaned_text
