"""Regression coverage for fresh-install and persistent-helper failures."""
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dictate.config import (
    QWEN3_ASR_MODEL,
    WHISPER_MODEL,
    LLMConfig,
    LLMModel,
    STTEngine,
    WhisperConfig,
)
from dictate.transcribe import (
    ANETranscriber,
    TextCleaner,
    TranscriptionPipeline,
    WhisperTranscriber,
)


@pytest.mark.parametrize('enabled,style', [(False, 'clean'), (True, 'raw')])
def test_raw_preload_never_downloads_or_loads_cleanup(enabled, style):
    with patch('dictate.transcribe.WhisperTranscriber') as stt:
        pipe = TranscriptionPipeline(WhisperConfig(engine=STTEngine.WHISPER), LLMConfig(enabled=enabled, writing_style=style))
    pipe._cleaner = MagicMock()
    with patch('dictate.config.is_model_cached', return_value=True), patch('dictate.model_download.download_model') as download:
        pipe.preload_models()
    stt.return_value.load_model.assert_called_once()
    pipe._cleaner.load_model.assert_not_called()
    download.assert_not_called()


def test_missing_qwen_runtime_falls_back_with_compatible_model_and_language():
    config = WhisperConfig(engine=STTEngine.QWEN3_ASR, model=QWEN3_ASR_MODEL, language='ja')
    with patch('dictate.transcribe.Qwen3ASRTranscriber.is_available', return_value=False):
        pipe = TranscriptionPipeline(config, LLMConfig(enabled=False))
    assert isinstance(pipe._whisper, WhisperTranscriber)
    assert pipe._whisper._config.model == WHISPER_MODEL
    assert pipe._whisper._config.language == 'ja'
    assert config.model == QWEN3_ASR_MODEL


def test_ane_timeout_discards_server_and_stale_response():
    transcriber = ANETranscriber(WhisperConfig(), binary_path='/fake/helper')
    proc = MagicMock()
    proc.poll.return_value = None
    transcriber._server = proc
    transcriber._model_loaded = True
    with patch.object(transcriber, '_readline_with_timeout', side_effect=subprocess.TimeoutExpired('serve', 30)):
        assert transcriber.transcribe(np.zeros(16000, dtype=np.int16), 16000) == ''
    assert transcriber._server is None
    assert not transcriber._model_loaded
    proc.terminate.assert_called_once()


def test_qwen_cleanup_disables_thinking_in_template():
    cleaner = TextCleaner(LLMConfig(model_choice=LLMModel.QWEN35_2B))
    cleaner._model = object()
    cleaner._tokenizer = MagicMock()
    llm = MagicMock()
    llm.generate.return_value = 'Hello world.'
    with patch.dict(sys.modules, {'mlx_lm': llm, 'mlx_lm.sample_utils': MagicMock()}):
        assert cleaner.cleanup('hello world') == 'Hello world.'
    assert cleaner._tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'] is False
    assert cleaner._tokenizer.apply_chat_template.call_args.args[0][1]['content'] == 'hello world'


def test_template_failure_does_not_lock_out_future_cleanup():
    cleaner = TextCleaner(LLMConfig())
    cleaner._model = object()
    cleaner._tokenizer = MagicMock()
    cleaner._tokenizer.apply_chat_template.side_effect = ValueError('bad template')
    with patch.dict(sys.modules, {'mlx_lm': MagicMock(), 'mlx_lm.sample_utils': MagicMock()}):
        with pytest.raises(ValueError):
            cleaner.cleanup('hello world')
    assert not cleaner._generation_lock.locked()


def test_qwen_availability_does_not_import_native_runtime():
    from dictate.transcribe import Qwen3ASRTranscriber
    with patch('importlib.util.find_spec', return_value=object()) as find:
        assert Qwen3ASRTranscriber.is_available()
    find.assert_called_once_with('mlx_audio')


def test_download_progress_supports_huggingface_xet_protocol():
    from dictate.model_download import download_model
    progress = []
    def hf_snapshot(**kwargs):
        bar = kwargs['tqdm_class'](total=100)
        assert bar.total == 100
        bar.total += 50
        bar.update(75)
        assert bar.n == 75
        bar.refresh()
        bar.close()
        assert 100.0 not in progress
    with patch('dictate.config.is_model_cached', return_value=False), patch('dictate.model_download.snapshot_download', side_effect=hf_snapshot):
        download_model('mlx-community/test-model', progress_callback=progress.append)
    assert progress[-1] == 100.0
