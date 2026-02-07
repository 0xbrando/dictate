"""Speech-to-text transcription and text cleanup."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

# Suppress huggingface/tqdm progress bars (must be set before imports)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import mlx_whisper
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from scipy.io.wavfile import write as wav_write

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np
    from mlx.nn import Module

    from dictate.config import LLMConfig, WhisperConfig

logger = logging.getLogger(__name__)


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


class TextCleaner:
    def __init__(self, config: "LLMConfig") -> None:
        self._config = config
        self._model: "Module | None" = None
        self._tokenizer = None

    def load_model(self) -> None:
        if self._model is not None:
            return

        print(f"   Qwen: {self._config.model}...", end=" ", flush=True)
        self._model, self._tokenizer = load(self._config.model)
        print("✓")

    def cleanup(self, text: str, output_language: str | None = None) -> str:
        if not self._config.enabled:
            return text

        if self._model is None or self._tokenizer is None:
            self.load_model()

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
        
        logger.debug("LLM raw result: %r", result)

        return self._postprocess(result.strip())

    def _postprocess(self, text: str) -> str:
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
            # "Sure" variants
            "Sure, here's the corrected text:",
            "Sure, here is the corrected text:",
            "Sure, here's the text:",
            "Sure, here is the text:",
            "Sure, here you go:",
            "Sure!",
            "Sure:",
            "Sure,",
            # "Here" variants
            "Here's the corrected text:",
            "Here is the corrected text:",
            "Here's the formatted text:",
            "Here is the formatted text:",
            "Here's the text:",
            "Here is the text:",
            "Here you go:",
            "Here it is:",
            # "Corrected/Fixed" variants
            "Corrected text:",
            "Corrected:",
            "Fixed text:",
            "Fixed:",
            "Formatted text:",
            "Formatted:",
            # "The" variants
            "The corrected text is:",
            "The corrected text:",
            "The text:",
            # "I" variants
            "I've corrected the text:",
            "I have corrected the text:",
            "I fixed the text:",
            # "Of course" variants
            "Of course!",
            "Of course:",
            "Of course,",
            # "Certainly" variants
            "Certainly!",
            "Certainly:",
            "Certainly,",
            # Other
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


class TranscriptionPipeline:
    def __init__(
        self,
        whisper_config: "WhisperConfig",
        llm_config: "LLMConfig",
    ) -> None:
        self._whisper = WhisperTranscriber(whisper_config)
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
        duration_s = len(audio) / self._sample_rate
        print(f"⏳ Processing {duration_s:.1f}s of audio...")

        import time
        t0 = time.time()
        raw_text = self._whisper.transcribe(
            audio, self._sample_rate, language=input_language
        ).strip()
        t1 = time.time()

        if not raw_text:
            print("   ⚠️ No speech detected")
            return None

        print(f"   📝 Heard: \"{raw_text}\" ({t1-t0:.1f}s)")

        t2 = time.time()
        cleaned_text = self._cleaner.cleanup(raw_text, output_language=output_language).strip()
        t3 = time.time()

        if not cleaned_text:
            print("   ⚠️ Cleanup failed")
            return None

        if cleaned_text != raw_text:
            print(f"   ✨ Fixed: \"{cleaned_text}\" ({t3-t2:.1f}s)")
        else:
            print(f"   ✓ No changes needed ({t3-t2:.1f}s)")

        return cleaned_text
