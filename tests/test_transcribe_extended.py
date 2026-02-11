"""Extended tests for dictate.transcribe — classes, pipeline, and error paths.

Tests WhisperTranscriber, ParakeetTranscriber, TextCleaner, APITextCleaner,
TranscriptionPipeline with mocked ML backends.
"""

from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dictate.config import (
    LLMBackend,
    LLMConfig,
    LLMModel,
    STTEngine,
    WhisperConfig,
)
from dictate.transcribe import (
    DEDUP_WINDOW_SECONDS,
    SMART_ROUTING_THRESHOLD,
    APITextCleaner,
    ParakeetTranscriber,
    TextCleaner,
    TranscriptionPipeline,
    WhisperTranscriber,
    _looks_clean,
    _postprocess,
    _temp_wav_context,
)


# ── Helpers ───────────────────────────────────────────────────


def _make_http_response(body: dict) -> MagicMock:
    """Create a mock HTTP response context manager."""
    data = json.dumps(body).encode()
    resp = MagicMock()
    resp.read.return_value = data
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _mock_mlx_whisper():
    """Create a mock mlx_whisper module."""
    m = MagicMock()
    m.transcribe.return_value = {"text": ""}
    return m


def _mock_mlx_lm():
    """Create mock mlx_lm and mlx_lm.sample_utils modules."""
    mlx_lm = MagicMock()
    mlx_lm.load.return_value = (MagicMock(), MagicMock())
    mlx_lm.generate.return_value = "result"
    sample_utils = MagicMock()
    sample_utils.make_sampler.return_value = MagicMock()
    mlx_lm.sample_utils = sample_utils
    return mlx_lm, sample_utils


# ── _temp_wav_context ─────────────────────────────────────────


class TestTempWavContext:
    """Test the temporary WAV file context manager."""

    def test_creates_and_cleans_up_wav_file(self):
        audio = np.zeros(1600, dtype=np.int16)
        with _temp_wav_context(audio, 16000) as path:
            assert os.path.exists(path)
            assert path.endswith(".wav")
            assert "dictate_" in os.path.basename(path)
        assert not os.path.exists(path)

    def test_wav_file_is_valid(self):
        from scipy.io.wavfile import read as wav_read

        audio = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)
        with _temp_wav_context(audio, 16000) as path:
            sr, data = wav_read(path)
            assert sr == 16000
            np.testing.assert_array_equal(data, audio)

    def test_cleanup_on_exception(self):
        audio = np.zeros(100, dtype=np.int16)
        path_ref = None
        with pytest.raises(ValueError):
            with _temp_wav_context(audio, 16000) as path:
                path_ref = path
                raise ValueError("test")
        assert path_ref is not None
        assert not os.path.exists(path_ref)

    @patch("dictate.transcribe.wav_write", side_effect=OSError("disk full"))
    def test_write_failure_raises_runtime_error(self, mock_write):
        audio = np.zeros(100, dtype=np.int16)
        with pytest.raises(RuntimeError, match="disk full"):
            with _temp_wav_context(audio, 16000):
                pass


# ── WhisperTranscriber ────────────────────────────────────────


class TestWhisperTranscriber:
    """Tests for WhisperTranscriber with mocked mlx_whisper."""

    @pytest.fixture
    def config(self):
        return WhisperConfig(model="test-whisper", language="en", engine=STTEngine.WHISPER)

    def test_init(self, config):
        t = WhisperTranscriber(config)
        assert t._config is config
        assert t._model_loaded is False

    def test_load_model(self, config, capsys):
        mock_mw = _mock_mlx_whisper()
        with patch.dict(sys.modules, {"mlx_whisper": mock_mw}):
            t = WhisperTranscriber(config)
            t.load_model()
            assert t._model_loaded is True
            mock_mw.transcribe.assert_called_once()
            captured = capsys.readouterr()
            assert "Whisper" in captured.out
            assert "✓" in captured.out

    def test_load_model_idempotent(self, config):
        mock_mw = _mock_mlx_whisper()
        with patch.dict(sys.modules, {"mlx_whisper": mock_mw}):
            t = WhisperTranscriber(config)
            t.load_model()
            t.load_model()
            assert mock_mw.transcribe.call_count == 1

    def test_transcribe(self, config):
        mock_mw = _mock_mlx_whisper()
        mock_mw.transcribe.return_value = {"text": "Hello world."}
        with patch.dict(sys.modules, {"mlx_whisper": mock_mw}):
            t = WhisperTranscriber(config)
            audio = np.zeros(16000, dtype=np.int16)
            result = t.transcribe(audio, 16000)
            assert result == "Hello world."
            assert t._model_loaded is True

    def test_transcribe_with_language_override(self, config):
        mock_mw = _mock_mlx_whisper()
        mock_mw.transcribe.return_value = {"text": "Bonjour."}
        with patch.dict(sys.modules, {"mlx_whisper": mock_mw}):
            t = WhisperTranscriber(config)
            audio = np.zeros(16000, dtype=np.int16)
            result = t.transcribe(audio, 16000, language="fr")
            assert result == "Bonjour."
            call_kw = mock_mw.transcribe.call_args[1]
            assert call_kw["language"] == "fr"

    def test_transcribe_empty_text(self, config):
        mock_mw = _mock_mlx_whisper()
        mock_mw.transcribe.return_value = {"text": ""}
        with patch.dict(sys.modules, {"mlx_whisper": mock_mw}):
            t = WhisperTranscriber(config)
            result = t.transcribe(np.zeros(16000, dtype=np.int16), 16000)
            assert result == ""

    def test_transcribe_missing_text_key(self, config):
        mock_mw = _mock_mlx_whisper()
        mock_mw.transcribe.return_value = {}
        with patch.dict(sys.modules, {"mlx_whisper": mock_mw}):
            t = WhisperTranscriber(config)
            result = t.transcribe(np.zeros(16000, dtype=np.int16), 16000)
            assert result == ""

    def test_transcribe_none_language_uses_config(self, config):
        mock_mw = _mock_mlx_whisper()
        mock_mw.transcribe.return_value = {"text": "test"}
        with patch.dict(sys.modules, {"mlx_whisper": mock_mw}):
            t = WhisperTranscriber(config)
            t.transcribe(np.zeros(16000, dtype=np.int16), 16000, language=None)
            call_kw = mock_mw.transcribe.call_args[1]
            assert call_kw["language"] == "en"


# ── ParakeetTranscriber ───────────────────────────────────────


class TestParakeetTranscriber:
    """Tests for ParakeetTranscriber."""

    @pytest.fixture
    def config(self):
        return WhisperConfig(model="test-parakeet", language=None, engine=STTEngine.PARAKEET)

    def test_init(self, config):
        t = ParakeetTranscriber(config)
        assert t._config is config
        assert t._model is None

    def test_load_model(self, config, capsys):
        mock_parakeet = MagicMock()
        mock_model = MagicMock()
        mock_parakeet.from_pretrained.return_value = mock_model
        with patch.dict(sys.modules, {"parakeet_mlx": mock_parakeet}):
            t = ParakeetTranscriber(config)
            t.load_model()
            assert t._model is mock_model
            captured = capsys.readouterr()
            assert "Parakeet" in captured.out
            assert "✓" in captured.out

    def test_load_model_import_error(self, config):
        with patch.dict(sys.modules, {"parakeet_mlx": None}):
            t = ParakeetTranscriber(config)
            with pytest.raises(ImportError, match="parakeet-mlx is required"):
                t.load_model()

    def test_transcribe_deduplicates(self, config):
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "hello world hello world"
        mock_model.transcribe.return_value = mock_result

        t = ParakeetTranscriber(config)
        t._model = mock_model
        result = t.transcribe(np.zeros(16000, dtype=np.int16), 16000)
        assert result == "hello world"

    def test_transcribe_strips_whitespace(self, config):
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "  Hello.  "
        mock_model.transcribe.return_value = mock_result

        t = ParakeetTranscriber(config)
        t._model = mock_model
        result = t.transcribe(np.zeros(16000, dtype=np.int16), 16000)
        assert result == "Hello."

    def test_transcribe_empty(self, config):
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = ""
        mock_model.transcribe.return_value = mock_result

        t = ParakeetTranscriber(config)
        t._model = mock_model
        result = t.transcribe(np.zeros(16000, dtype=np.int16), 16000)
        assert result == ""

    def test_transcribe_auto_loads_model(self, config):
        mock_parakeet = MagicMock()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Loaded."
        mock_model.transcribe.return_value = mock_result
        mock_parakeet.from_pretrained.return_value = mock_model

        with patch.dict(sys.modules, {"parakeet_mlx": mock_parakeet}):
            t = ParakeetTranscriber(config)
            assert t._model is None
            result = t.transcribe(np.zeros(16000, dtype=np.int16), 16000)
            assert t._model is not None
            assert result == "Loaded."


# ── _postprocess extended ─────────────────────────────────────


class TestPostprocessExtended:
    """Additional _postprocess edge cases."""

    def test_strips_think_tags(self):
        assert _postprocess("<think>reasoning</think>Hello.") == "Hello."

    def test_strips_multiline_think_tags(self):
        assert _postprocess("<think>\nline1\nline2\n</think>Result.") == "Result."

    def test_all_preambles(self):
        preambles = [
            "Sure, here's the corrected text:", "Sure, here is the corrected text:",
            "Here's the corrected text:", "Here is the corrected text:",
            "Corrected text:", "Corrected:", "Fixed text:", "Fixed:",
            "Formatted text:", "Formatted:", "The corrected text is:",
            "The corrected text:", "The text:", "I've corrected the text:",
            "I have corrected the text:", "I fixed the text:", "Of course!",
            "Of course:", "Certainly!", "Certainly:", "Output:", "Result:", "Answer:",
        ]
        for p in preambles:
            result = _postprocess(f"{p} Hello.")
            assert result == "Hello.", f"Failed for preamble: {p!r}"


# ── TextCleaner ───────────────────────────────────────────────


class TestTextCleaner:
    """Tests for TextCleaner with mocked mlx_lm."""

    @pytest.fixture
    def config(self):
        return LLMConfig(enabled=True, backend=LLMBackend.LOCAL, model_choice=LLMModel.QWEN,
                         max_tokens=300, temperature=0.0)

    def test_init(self, config):
        c = TextCleaner(config)
        assert c._model is None
        assert c._tokenizer is None

    def test_load_model(self, config, capsys):
        mock_mlx, mock_su = _mock_mlx_lm()
        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "mlx_lm.sample_utils": mock_su}):
            c = TextCleaner(config)
            c.load_model()
            mock_mlx.load.assert_called_once_with(config.model)
            assert c._model is not None
            assert c._tokenizer is not None
            captured = capsys.readouterr()
            assert "LLM" in captured.out

    def test_load_model_idempotent(self, config):
        mock_mlx, mock_su = _mock_mlx_lm()
        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "mlx_lm.sample_utils": mock_su}):
            c = TextCleaner(config)
            c.load_model()
            c.load_model()
            assert mock_mlx.load.call_count == 1

    def test_cleanup(self, config):
        mock_mlx, mock_su = _mock_mlx_lm()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted"
        mock_mlx.load.return_value = (MagicMock(), mock_tokenizer)
        mock_mlx.generate.return_value = "Hello world."

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "mlx_lm.sample_utils": mock_su}):
            c = TextCleaner(config)
            result = c.cleanup("hello world")
            assert result == "Hello world."
            mock_mlx.generate.assert_called_once()

    def test_cleanup_auto_loads(self, config):
        mock_mlx, mock_su = _mock_mlx_lm()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_mlx.load.return_value = (MagicMock(), mock_tokenizer)
        mock_mlx.generate.return_value = "Clean."

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "mlx_lm.sample_utils": mock_su}):
            c = TextCleaner(config)
            assert c._model is None
            c.cleanup("dirty")
            mock_mlx.load.assert_called_once()

    def test_cleanup_disabled_returns_input(self, config):
        config.enabled = False
        c = TextCleaner(config)
        assert c.cleanup("hello") == "hello"

    def test_cleanup_with_output_language(self, config):
        mock_mlx, mock_su = _mock_mlx_lm()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_mlx.load.return_value = (MagicMock(), mock_tokenizer)
        mock_mlx.generate.return_value = "Bonjour."

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "mlx_lm.sample_utils": mock_su}):
            c = TextCleaner(config)
            result = c.cleanup("hello", output_language="fr")
            assert result == "Bonjour."
            msgs = mock_tokenizer.apply_chat_template.call_args[0][0]
            assert any("French" in m["content"] for m in msgs if m["role"] == "system")

    def test_cleanup_max_tokens_capped(self, config):
        mock_mlx, mock_su = _mock_mlx_lm()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_mlx.load.return_value = (MagicMock(), mock_tokenizer)
        mock_mlx.generate.return_value = "result"

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "mlx_lm.sample_utils": mock_su}):
            c = TextCleaner(config)
            c.cleanup("one two three")
            # 3 words * 3 = 9, max(50, 9) = 50, min(300, 50) = 50
            call_kw = mock_mlx.generate.call_args[1]
            assert call_kw["max_tokens"] == 50

    def test_cleanup_postprocesses(self, config):
        mock_mlx, mock_su = _mock_mlx_lm()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_mlx.load.return_value = (MagicMock(), mock_tokenizer)
        mock_mlx.generate.return_value = "Sure! Hello."

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx, "mlx_lm.sample_utils": mock_su}):
            c = TextCleaner(config)
            result = c.cleanup("hello")
            assert result == "Hello."


# ── APITextCleaner ────────────────────────────────────────────


class TestAPITextCleaner:
    """Tests for APITextCleaner with mocked urllib."""

    @pytest.fixture
    def config(self):
        return LLMConfig(enabled=True, backend=LLMBackend.API,
                         api_url="http://localhost:8005/v1/chat/completions",
                         max_tokens=300, temperature=0.0)

    def test_init(self, config):
        c = APITextCleaner(config)
        assert c._last_cleanup_failed is False

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_load_model_success(self, mock_urlopen, config, capsys):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        c = APITextCleaner(config)
        c.load_model()
        captured = capsys.readouterr()
        assert "API" in captured.out and "✓" in captured.out

    @patch("dictate.transcribe.urllib.request.urlopen", side_effect=TimeoutError("timeout"))
    def test_load_model_timeout(self, mock_urlopen, config, capsys):
        c = APITextCleaner(config)
        c.load_model()
        captured = capsys.readouterr()
        assert "will retry" in captured.out

    @patch("dictate.transcribe.urllib.request.urlopen", side_effect=Exception("unknown"))
    def test_load_model_unknown_error(self, mock_urlopen, config, capsys):
        c = APITextCleaner(config)
        c.load_model()
        captured = capsys.readouterr()
        assert "error" in captured.out

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_success(self, mock_urlopen, config):
        mock_urlopen.return_value = _make_http_response(
            {"choices": [{"message": {"content": "Hello."}}]}
        )
        c = APITextCleaner(config)
        assert c.cleanup("hello") == "Hello."
        assert c._last_cleanup_failed is False

    def test_cleanup_disabled(self, config):
        config.enabled = False
        c = APITextCleaner(config)
        assert c.cleanup("hello") == "hello"

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_with_language(self, mock_urlopen, config):
        mock_urlopen.return_value = _make_http_response(
            {"choices": [{"message": {"content": "Bonjour."}}]}
        )
        c = APITextCleaner(config)
        result = c.cleanup("hello", output_language="fr")
        assert result == "Bonjour."
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data)
        assert "French" in payload["messages"][0]["content"]

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_retry_on_network_error(self, mock_urlopen, config):
        import urllib.error
        success = _make_http_response({"choices": [{"message": {"content": "OK."}}]})
        mock_urlopen.side_effect = [urllib.error.URLError("refused"), success]

        c = APITextCleaner(config)
        assert c.cleanup("test") == "OK."
        assert mock_urlopen.call_count == 2

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_both_fail_returns_raw(self, mock_urlopen, config):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("down")

        c = APITextCleaner(config)
        assert c.cleanup("raw text") == "raw text"
        assert c._last_cleanup_failed is True

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_json_decode_error(self, mock_urlopen, config):
        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        c = APITextCleaner(config)
        assert c.cleanup("raw") == "raw"
        assert c._last_cleanup_failed is True

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_missing_choices_key(self, mock_urlopen, config):
        mock_urlopen.return_value = _make_http_response({"result": "wrong"})
        c = APITextCleaner(config)
        assert c.cleanup("raw") == "raw"
        assert c._last_cleanup_failed is True

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_unexpected_exception(self, mock_urlopen, config):
        mock_urlopen.side_effect = RuntimeError("boom")
        c = APITextCleaner(config)
        assert c.cleanup("raw") == "raw"
        assert c._last_cleanup_failed is True

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_retry_json_error(self, mock_urlopen, config):
        import urllib.error
        bad_resp = MagicMock()
        bad_resp.read.return_value = b"bad"
        bad_resp.__enter__ = MagicMock(return_value=bad_resp)
        bad_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [urllib.error.URLError("err"), bad_resp]
        c = APITextCleaner(config)
        assert c.cleanup("raw") == "raw"
        assert c._last_cleanup_failed is True

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_postprocesses(self, mock_urlopen, config):
        mock_urlopen.return_value = _make_http_response(
            {"choices": [{"message": {"content": "Sure! Hello."}}]}
        )
        c = APITextCleaner(config)
        assert c.cleanup("hello") == "Hello."

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_timeout_error_retries(self, mock_urlopen, config):
        success = _make_http_response({"choices": [{"message": {"content": "OK."}}]})
        mock_urlopen.side_effect = [TimeoutError("timed out"), success]
        c = APITextCleaner(config)
        assert c.cleanup("test") == "OK."

    @patch("dictate.transcribe.urllib.request.urlopen")
    def test_cleanup_connection_error_retries(self, mock_urlopen, config):
        success = _make_http_response({"choices": [{"message": {"content": "OK."}}]})
        mock_urlopen.side_effect = [ConnectionError("refused"), success]
        c = APITextCleaner(config)
        assert c.cleanup("test") == "OK."


# ── _looks_clean extended ─────────────────────────────────────


class TestLooksCleanExtended:
    def test_parenthesis_start(self):
        assert _looks_clean("(Note) ok.") is True

    def test_fillers(self):
        for filler in ["I mean", "So", "Well", "Er", "Ah", "Basically"]:
            assert _looks_clean(f"{filler} yes.") is False

    def test_semicolon_ending(self):
        assert _looks_clean("Four words end here;") is True

    def test_colon_ending(self):
        assert _looks_clean("Four words end here:") is True


# ── TranscriptionPipeline ────────────────────────────────────


class TestTranscriptionPipeline:
    @pytest.fixture
    def whisper_config(self):
        return WhisperConfig(model="test", language="en", engine=STTEngine.WHISPER)

    @pytest.fixture
    def parakeet_config(self):
        return WhisperConfig(model="test", language=None, engine=STTEngine.PARAKEET)

    @pytest.fixture
    def llm_config(self):
        return LLMConfig(enabled=True, backend=LLMBackend.LOCAL, model_choice=LLMModel.QWEN)

    @pytest.fixture
    def api_llm_config(self):
        return LLMConfig(enabled=True, backend=LLMBackend.API,
                         api_url="http://localhost:8005/v1/chat/completions")

    def test_init_whisper(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        assert isinstance(pipe._whisper, WhisperTranscriber)
        assert isinstance(pipe._cleaner, TextCleaner)

    def test_init_parakeet(self, parakeet_config, llm_config):
        pipe = TranscriptionPipeline(parakeet_config, llm_config)
        assert isinstance(pipe._whisper, ParakeetTranscriber)

    def test_init_api(self, whisper_config, api_llm_config):
        with patch.object(TranscriptionPipeline, "_create_fast_cleaner", return_value=None):
            pipe = TranscriptionPipeline(whisper_config, api_llm_config)
            assert isinstance(pipe._cleaner, APITextCleaner)

    def test_set_sample_rate(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe.set_sample_rate(44100)
        assert pipe._sample_rate == 44100

    def test_is_duplicate_within_window(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        assert pipe._is_duplicate("hello") is False
        assert pipe._is_duplicate("hello") is True

    def test_is_duplicate_case_insensitive(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._is_duplicate("Hello World")
        assert pipe._is_duplicate("hello world") is True

    def test_is_duplicate_different_text(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._is_duplicate("hello")
        assert pipe._is_duplicate("goodbye") is False

    def test_is_duplicate_outside_window(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._is_duplicate("hello")
        pipe._last_output_time = time.time() - DEDUP_WINDOW_SECONDS - 1
        assert pipe._is_duplicate("hello") is False

    def test_pick_cleaner_short_with_fast(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        mock_fast = MagicMock()
        pipe._fast_cleaner = mock_fast
        assert pipe._pick_cleaner(5) is mock_fast

    def test_pick_cleaner_long_uses_main(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._fast_cleaner = MagicMock()
        assert pipe._pick_cleaner(SMART_ROUTING_THRESHOLD + 1) is pipe._cleaner

    def test_pick_cleaner_no_fast(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._fast_cleaner = None
        assert pipe._pick_cleaner(3) is pipe._cleaner

    def test_process_no_speech(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = ""
        assert pipe.process(np.zeros(16000, dtype=np.int16)) is None

    def test_process_with_cleanup(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "hello world this is a longer test sentence"
        pipe._cleaner = MagicMock()
        pipe._cleaner.cleanup.return_value = "Hello world. This is a longer test sentence."
        pipe._cleaner._last_cleanup_failed = False
        pipe._fast_cleaner = None

        result = pipe.process(np.zeros(16000, dtype=np.int16))
        assert result == "Hello world. This is a longer test sentence."
        pipe._cleaner.cleanup.assert_called_once()

    def test_process_clean_text_skips_llm(self, whisper_config, llm_config):
        llm_config.writing_style = "clean"
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "Hello."
        pipe._cleaner = MagicMock()
        pipe._fast_cleaner = None

        result = pipe.process(np.zeros(16000, dtype=np.int16))
        assert result == "Hello."
        pipe._cleaner.cleanup.assert_not_called()

    def test_process_translation_never_skips(self, whisper_config, llm_config):
        llm_config.writing_style = "clean"
        llm_config.output_language = "fr"
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "Hello."
        pipe._cleaner = MagicMock()
        pipe._cleaner.cleanup.return_value = "Bonjour."
        pipe._cleaner._last_cleanup_failed = False
        pipe._fast_cleaner = None

        result = pipe.process(np.zeros(16000, dtype=np.int16))
        assert result == "Bonjour."
        pipe._cleaner.cleanup.assert_called_once()

    def test_process_duplicate_returns_none(self, whisper_config, llm_config):
        llm_config.writing_style = "clean"
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "Hello."
        pipe._fast_cleaner = None

        assert pipe.process(np.zeros(16000, dtype=np.int16)) == "Hello."
        assert pipe.process(np.zeros(16000, dtype=np.int16)) is None

    def test_process_empty_cleanup(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "some words that need cleanup here"
        pipe._cleaner = MagicMock()
        pipe._cleaner.cleanup.return_value = ""
        pipe._cleaner._last_cleanup_failed = False
        pipe._fast_cleaner = None

        assert pipe.process(np.zeros(16000, dtype=np.int16)) is None

    def test_process_surfaces_cleanup_failure(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "hello world this text needs cleanup right now"
        pipe._cleaner = MagicMock()
        pipe._cleaner.cleanup.return_value = "hello world this text needs cleanup right now"
        pipe._cleaner._last_cleanup_failed = True
        pipe._fast_cleaner = None

        pipe.process(np.zeros(16000, dtype=np.int16))
        assert pipe.last_cleanup_failed is True
        assert pipe._cleaner._last_cleanup_failed is False

    def test_process_smart_routing(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "hey there how are you"
        mock_fast = MagicMock()
        mock_fast.cleanup.return_value = "Hey there, how are you?"
        mock_fast._last_cleanup_failed = False
        pipe._fast_cleaner = mock_fast

        result = pipe.process(np.zeros(16000, dtype=np.int16))
        assert result == "Hey there, how are you?"
        mock_fast.cleanup.assert_called_once()

    def test_process_non_clean_style_uses_llm(self, whisper_config, llm_config):
        llm_config.writing_style = "formal"
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "Hello."
        pipe._cleaner = MagicMock()
        pipe._cleaner.cleanup.return_value = "Hello."
        pipe._cleaner._last_cleanup_failed = False
        pipe._fast_cleaner = None

        pipe.process(np.zeros(16000, dtype=np.int16))
        pipe._cleaner.cleanup.assert_called_once()

    def test_process_with_language_override(self, whisper_config, llm_config):
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._whisper.transcribe.return_value = "bonjour monde cette phrase est longue pour test"
        pipe._cleaner = MagicMock()
        pipe._cleaner.cleanup.return_value = "Bonjour monde."
        pipe._cleaner._last_cleanup_failed = False
        pipe._fast_cleaner = None

        pipe.process(np.zeros(16000, dtype=np.int16), input_language="fr", output_language="en")
        pipe._whisper.transcribe.assert_called_once()
        call_args = pipe._whisper.transcribe.call_args
        assert call_args[1].get("language") or call_args[0][2] == "fr"


# ── _create_fast_cleaner ─────────────────────────────────────


class TestCreateFastCleaner:
    def test_returns_none_for_local(self):
        config = LLMConfig(backend=LLMBackend.LOCAL)
        assert TranscriptionPipeline._create_fast_cleaner(config) is None

    @patch("dictate.config.is_model_cached", return_value=False)
    def test_returns_none_when_no_cached(self, mock_cached):
        config = LLMConfig(backend=LLMBackend.API)
        assert TranscriptionPipeline._create_fast_cleaner(config) is None

    @patch("dictate.config.is_model_cached")
    def test_picks_first_cached(self, mock_cached):
        def check(repo):
            return "Qwen2.5-3B" in repo
        mock_cached.side_effect = check

        config = LLMConfig(backend=LLMBackend.API)
        result = TranscriptionPipeline._create_fast_cleaner(config)
        assert result is not None
        assert isinstance(result, TextCleaner)

    @patch("dictate.config.is_model_cached", return_value=True)
    def test_prefers_smallest(self, mock_cached):
        config = LLMConfig(backend=LLMBackend.API)
        result = TranscriptionPipeline._create_fast_cleaner(config)
        assert result is not None
        assert "1.5B" in result._config.model


# ── preload_models ────────────────────────────────────────────


class TestPreloadModels:
    @pytest.fixture
    def pipeline(self):
        whisper_config = WhisperConfig(model="test", engine=STTEngine.WHISPER)
        llm_config = LLMConfig(enabled=True, backend=LLMBackend.LOCAL)
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._cleaner = MagicMock()
        pipe._cleaner._config = MagicMock()
        pipe._cleaner._config.model = "test-llm-model"
        pipe._fast_cleaner = None
        return pipe

    @patch("dictate.config.is_model_cached", return_value=True)
    def test_preload_cached(self, mock_cached, pipeline):
        progress = []
        pipeline.preload_models(on_progress=progress.append)
        pipeline._whisper.load_model.assert_called_once()
        pipeline._cleaner.load_model.assert_called_once()
        assert any("Whisper" in c for c in progress)

    @patch("dictate.model_download.download_model")
    @patch("dictate.config.is_model_cached", return_value=False)
    def test_preload_downloads(self, mock_cached, mock_download, pipeline):
        progress = []
        pipeline.preload_models(on_progress=progress.append)
        assert mock_download.call_count >= 1
        assert any("Downloading" in c for c in progress)

    @patch("dictate.model_download.download_model", side_effect=Exception("fail"))
    @patch("dictate.config.is_model_cached", return_value=False)
    def test_preload_download_failure_raises(self, mock_cached, mock_download, pipeline):
        with pytest.raises(Exception, match="fail"):
            pipeline.preload_models()

    @patch("dictate.config.is_model_cached", return_value=True)
    def test_preload_with_fast_cleaner(self, mock_cached):
        whisper_config = WhisperConfig(model="test", engine=STTEngine.WHISPER)
        llm_config = LLMConfig(enabled=True, backend=LLMBackend.LOCAL)
        pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()
        pipe._cleaner = MagicMock()
        pipe._cleaner._config = MagicMock()
        pipe._cleaner._config.model = "test"
        pipe._fast_cleaner = MagicMock()

        pipe.preload_models()
        pipe._fast_cleaner.load_model.assert_called_once()

    @patch("dictate.config.is_model_cached", return_value=True)
    def test_preload_api_cleaner(self, mock_cached):
        whisper_config = WhisperConfig(model="test", engine=STTEngine.WHISPER)
        llm_config = LLMConfig(enabled=True, backend=LLMBackend.API)
        with patch.object(TranscriptionPipeline, "_create_fast_cleaner", return_value=None):
            pipe = TranscriptionPipeline(whisper_config, llm_config)
        pipe._whisper = MagicMock()

        progress = []
        pipe.preload_models(on_progress=progress.append)
        assert any("API" in c or "Connecting" in c for c in progress)

    @patch("dictate.config.is_model_cached", return_value=True)
    def test_preload_no_callback(self, mock_cached, pipeline):
        pipeline.preload_models(on_progress=None)
        pipeline._whisper.load_model.assert_called_once()
