"""Regressions observed during dictation from the Finder-launched Mac app."""

import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from dictate.config import WhisperConfig
from dictate.output import PastePermissionError, TyperOutput
from dictate.transcribe import ANETranscriber


def test_helper_reads_utf8_even_when_default_pipe_encoding_is_ascii(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'{\"ready\":true}\\n')\n"
        "sys.stdout.buffer.flush()\n"
        "for line in sys.stdin:\n"
        "    sys.stdout.buffer.write('{\"text\":\"Привет 世界 café\"}\\n'.encode('utf-8'))\n"
        "    sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    real_popen = subprocess.Popen

    def start_helper(args, **kwargs):
        return real_popen([sys.executable, str(helper)], **kwargs)

    transcriber = ANETranscriber(WhisperConfig(), binary_path=str(helper))
    try:
        with (
            patch("subprocess._text_encoding", return_value="ascii"),
            patch("dictate.transcribe.subprocess.Popen", side_effect=start_helper),
        ):
            assert transcriber.transcribe(np.zeros(16000, dtype=np.int16), 16000) == "Привет 世界 café"
    finally:
        transcriber._stop_server()


def test_blocked_paste_preserves_clipboard_and_can_retry_without_extra_space():
    with (
        patch("dictate.output.KeyboardController") as controller,
        patch("dictate.output.pyperclip.copy") as copy,
        patch("dictate.output.time.sleep"),
        patch("dictate.output._accessibility_trusted", return_value=False) as trusted,
    ):
        output = TyperOutput()
        with pytest.raises(PastePermissionError, match="Accessibility"):
            output.output("Hello world")
        copy.assert_called_once_with("Hello world")
        controller.return_value.press.assert_not_called()
        controller.return_value.type.assert_not_called()
        trusted.return_value = True
        output.output("Hello world")
        assert copy.call_args.args == ("Hello world",)
        assert controller.return_value.press.call_count == 2


def test_bad_helper_encoding_discards_outstanding_response():
    from unittest.mock import MagicMock

    transcriber = ANETranscriber(WhisperConfig(), binary_path="/fake/helper")
    proc = MagicMock()
    proc.poll.return_value = None
    transcriber._server = proc
    transcriber._model_loaded = True
    with patch.object(
        transcriber, "_readline_with_timeout",
        side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
    ):
        assert transcriber.transcribe(np.zeros(16000, dtype=np.int16), 16000) == ""
    assert transcriber._server is None
    assert not transcriber._model_loaded
    proc.terminate.assert_called_once()
