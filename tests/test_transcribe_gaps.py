"""Tests targeting uncovered paths in transcribe.py — dedup edge cases,
download progress callbacks, duplicate detection, and LLM output postprocessing."""

import os
import time as time_mod
from unittest.mock import MagicMock, patch

import pytest


class TestDedupTranscription:
    """_dedup_transcription — word-level deduplication edge cases."""

    def test_exact_repetition_even(self):
        """6-word exact repetition detected."""
        from dictate.transcribe import _dedup_transcription
        text = "hello world test hello world test"
        result = _dedup_transcription(text)
        assert result == "hello world test"

    def test_case_insensitive_dedup(self):
        """Dedup is case-insensitive, preserves original case."""
        from dictate.transcribe import _dedup_transcription
        text = "Hello World hello world"
        result = _dedup_transcription(text)
        assert result == "Hello World"

    def test_short_text_no_dedup(self):
        """< 4 words skips dedup."""
        from dictate.transcribe import _dedup_transcription
        assert _dedup_transcription("one two three") == "one two three"
        assert _dedup_transcription("one two") == "one two"
        assert _dedup_transcription("one") == "one"

    def test_no_repetition(self):
        """Non-repeated text passes through."""
        from dictate.transcribe import _dedup_transcription
        text = "this is a unique sentence with no repeats"
        assert _dedup_transcription(text) == text

    def test_off_by_one_odd_count(self):
        """Off-by-one check for n >= 5 with matching split."""
        from dictate.transcribe import _dedup_transcription
        # 8 words. half=4. Check at half (4) and half+1 (5).
        # At split=4: "a b c d" vs "a b c d" — match!
        text = "a b c d a b c d"
        result = _dedup_transcription(text)
        assert result == "a b c d"

    def test_off_by_one_split_at_ge_n(self):
        """split_at >= n triggers continue (skip that split)."""
        from dictate.transcribe import _dedup_transcription
        # 5 words, half=2, half+1=3. split=2: "x y" vs "z x y" (no).
        # split=3: "x y z" vs "x y" (no). No dedup.
        text = "x y z x y"
        result = _dedup_transcription(text)
        # "x y z" != "x y", "x y" != "z x y" → no dedup
        assert result == text


class TestPostprocess:
    """_postprocess — LLM output cleaning."""

    def test_repeated_first_line(self):
        """Detects and truncates repeated first line."""
        from dictate.transcribe import _postprocess
        text = "Hello world\nHello world\nSomething else"
        result = _postprocess(text)
        assert result == "Hello world"

    def test_no_repetition_multiline(self):
        """Non-repeated multiline passes through."""
        from dictate.transcribe import _postprocess
        text = "Hello world\nGoodbye world"
        result = _postprocess(text)
        assert "Hello world" in result
        assert "Goodbye world" in result

    def test_single_line(self):
        """Single line passes through."""
        from dictate.transcribe import _postprocess
        assert _postprocess("Just one line") == "Just one line"

    def test_empty_string(self):
        """Empty string returns empty."""
        from dictate.transcribe import _postprocess
        assert _postprocess("") == ""

    def test_strips_single_quotes(self):
        """Single quotes around text are stripped."""
        from dictate.transcribe import _postprocess
        assert _postprocess("'Hello world'") == "Hello world"

    def test_strips_double_quotes(self):
        """Double quotes around text are stripped."""
        from dictate.transcribe import _postprocess
        assert _postprocess('"Hello world"') == "Hello world"

    def test_leading_newlines_stripped(self):
        """Leading newlines are stripped."""
        from dictate.transcribe import _postprocess
        assert _postprocess("\n\nHello world") == "Hello world"

    def test_strips_preamble(self):
        """Common LLM preambles are stripped."""
        from dictate.transcribe import _postprocess
        assert _postprocess("Sure, here's the corrected text: Hello world") == "Hello world"

    def test_strips_think_tags(self):
        """<think> blocks are stripped."""
        from dictate.transcribe import _postprocess
        text = "<think>reasoning here</think>Hello world"
        assert _postprocess(text) == "Hello world"

    def test_strips_special_tokens(self):
        """Special tokens like <|end|> are removed."""
        from dictate.transcribe import _postprocess
        assert _postprocess("Hello world<|end|>") == "Hello world"
        assert _postprocess("Hello<|endoftext|>") == "Hello"


class TestPipelineLoadModels:
    """TranscriptionPipeline.load_models — download and progress paths."""

    def _make_pipeline(self, whisper_model="mlx-community/whisper-small",
                       llm_model=None, has_fast_cleaner=False):
        from dictate.transcribe import TranscriptionPipeline

        mock_whisper = MagicMock()
        mock_whisper._config = MagicMock()
        mock_whisper._config.model = whisper_model

        mock_cleaner = MagicMock()
        if llm_model:
            mock_cleaner._config = MagicMock()
            mock_cleaner._config.model = llm_model
        else:
            mock_cleaner._config = None

        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline._whisper = mock_whisper
        pipeline._cleaner = mock_cleaner
        pipeline._fast_cleaner = MagicMock() if has_fast_cleaner else None
        pipeline._is_loaded = False
        pipeline._loading = False
        return pipeline

    def test_whisper_cached_loads_directly(self):
        """When Whisper cached, loads without downloading."""
        pipeline = self._make_pipeline()
        progress = []

        with patch("dictate.config.is_model_cached", return_value=True):
            pipeline.preload_models(on_progress=lambda m: progress.append(m))

        pipeline._whisper.load_model.assert_called_once()
        assert any("Loading Whisper" in p for p in progress)

    def test_whisper_download_with_progress(self):
        """Downloads Whisper with progress callbacks."""
        pipeline = self._make_pipeline()
        progress = []

        def mock_download(model, progress_callback=None):
            if progress_callback:
                progress_callback(50.0)

        with patch("dictate.config.is_model_cached", return_value=False):
            with patch("dictate.model_download.download_model", side_effect=mock_download):
                pipeline.preload_models(on_progress=lambda m: progress.append(m))

        assert any("Downloading Whisper" in p for p in progress)
        assert any("50%" in p for p in progress)

    def test_whisper_download_failure_raises(self):
        """RuntimeError when Whisper download fails."""
        pipeline = self._make_pipeline()

        with patch("dictate.config.is_model_cached", return_value=False):
            with patch("dictate.model_download.download_model", side_effect=RuntimeError("fail")):
                with pytest.raises(RuntimeError):
                    pipeline.preload_models()

    def test_fast_cleaner_loaded(self):
        """Fast cleaner model loaded when configured."""
        pipeline = self._make_pipeline(has_fast_cleaner=True)
        progress = []

        with patch("dictate.config.is_model_cached", return_value=True):
            pipeline.preload_models(on_progress=lambda m: progress.append(m))

        pipeline._fast_cleaner.load_model.assert_called_once()
        assert any("fast local" in p for p in progress)

    def test_llm_download_with_progress(self):
        """Downloads LLM model when not cached."""
        pipeline = self._make_pipeline(llm_model="mlx-community/some-llm")
        progress = []

        def mock_download(model, progress_callback=None):
            if progress_callback:
                progress_callback(75.0)

        call_count = [0]
        def mock_is_cached(model):
            call_count[0] += 1
            return call_count[0] == 1  # First call (Whisper) = True, second (LLM) = False

        with patch("dictate.config.is_model_cached", side_effect=mock_is_cached):
            with patch("dictate.model_download.download_model", side_effect=mock_download):
                pipeline.preload_models(on_progress=lambda m: progress.append(m))

        assert any("Downloading LLM" in p for p in progress)

    def test_llm_download_failure_raises(self):
        """RuntimeError when LLM download fails."""
        pipeline = self._make_pipeline(llm_model="mlx-community/some-llm")

        call_count = [0]
        def mock_is_cached(model):
            call_count[0] += 1
            return call_count[0] == 1  # Whisper cached, LLM not

        with patch("dictate.config.is_model_cached", side_effect=mock_is_cached):
            with patch("dictate.model_download.download_model", side_effect=RuntimeError("LLM fail")):
                with pytest.raises(RuntimeError):
                    pipeline.preload_models()


class TestPipelineDuplicateDetection:
    """TranscriptionPipeline._is_duplicate — time-windowed duplicate detection."""

    def test_duplicate_within_window(self):
        """Same text within dedup window is duplicate."""
        from dictate.transcribe import TranscriptionPipeline
        import time

        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline._last_output = "hello world"
        pipeline._last_output_time = time.time()  # uses time.time(), not monotonic

        assert pipeline._is_duplicate("hello world") is True

    def test_different_text_not_duplicate(self):
        """Different text is never a duplicate."""
        from dictate.transcribe import TranscriptionPipeline
        import time

        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline._last_output = "hello world"
        pipeline._last_output_time = time.time()

        assert pipeline._is_duplicate("goodbye world") is False

    def test_first_output_not_duplicate(self):
        """First output (empty last) is not duplicate."""
        from dictate.transcribe import TranscriptionPipeline

        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline._last_output = ""
        pipeline._last_output_time = 0.0

        assert pipeline._is_duplicate("hello") is False

    def test_duplicate_outside_window(self):
        """Same text outside dedup window is not duplicate."""
        from dictate.transcribe import TranscriptionPipeline

        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline._last_output = "hello world"
        pipeline._last_output_time = 0.0  # Way in the past (epoch 0)

        assert pipeline._is_duplicate("hello world") is False

    def test_duplicate_case_insensitive(self):
        """Duplicate detection is case-insensitive."""
        from dictate.transcribe import TranscriptionPipeline
        import time

        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline._last_output = "Hello World"
        pipeline._last_output_time = time.time()

        assert pipeline._is_duplicate("hello world") is True


class TestTempWavContext:
    """_temp_wav_context — temp file creation and cleanup."""

    def test_creates_and_cleans_wav(self):
        """Creates temp WAV and cleans up after."""
        from dictate.transcribe import _temp_wav_context
        import numpy as np

        audio = np.zeros(1600, dtype=np.int16)
        with _temp_wav_context(audio, 16000) as path:
            assert os.path.exists(path)
            assert path.endswith(".wav")
        # Cleaned up after context
        assert not os.path.exists(path)

    def test_disk_full_raises_runtime_error(self):
        """OSError during write raises RuntimeError."""
        from dictate.transcribe import _temp_wav_context
        import numpy as np

        audio = np.zeros(1600, dtype=np.int16)
        with patch("dictate.transcribe.wav_write", side_effect=OSError("No space")):
            with pytest.raises(RuntimeError, match="disk full"):
                with _temp_wav_context(audio, 16000) as path:
                    pass


class TestParakeetTranscriber:
    """ParakeetTranscriber edge cases."""

    def test_load_model_already_loaded_noop(self):
        """load_model is a no-op when model already loaded."""
        from dictate.transcribe import ParakeetTranscriber

        t = ParakeetTranscriber.__new__(ParakeetTranscriber)
        t._model = MagicMock()
        t._config = MagicMock()

        # Should return immediately without importing anything
        t.load_model()

    def test_load_model_import_error(self):
        """Raises ImportError when parakeet_mlx not installed."""
        from dictate.transcribe import ParakeetTranscriber

        t = ParakeetTranscriber.__new__(ParakeetTranscriber)
        t._model = None
        t._config = MagicMock()

        with patch.dict("sys.modules", {"parakeet_mlx": None}):
            with pytest.raises(ImportError, match="parakeet"):
                t.load_model()
