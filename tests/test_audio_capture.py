"""Tests for dictate.audio — AudioCapture class with mocked sounddevice.

Covers start/stop, stream management, audio callback, VAD processing,
chunk finalization, error handling, and properties.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from dictate.audio import (
    AUDIO_CLIP_MAX,
    AUDIO_CLIP_MIN,
    DEFAULT_SAMPLE_RATE,
    FADE_DURATION_SECONDS,
    FIRST_CHANNEL_INDEX,
    INT16_MAX,
    MIN_CHUNK_DURATION_SECONDS,
    RMS_EPSILON,
    AudioCapture,
    AudioDevice,
    VADState,
)
from dictate.config import AudioConfig, VADConfig


@pytest.fixture
def audio_config():
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        block_ms=30,
        device_id=None,
    )


@pytest.fixture
def vad_config():
    return VADConfig(
        rms_threshold=0.012,
        silence_timeout_s=2.0,
        pre_roll_s=0.25,
    )


@pytest.fixture
def on_chunk():
    return MagicMock()


@pytest.fixture
def capture(audio_config, vad_config, on_chunk):
    return AudioCapture(audio_config, vad_config, on_chunk)


class TestAudioCaptureInit:
    """Tests for AudioCapture initialization."""

    def test_init_sets_configs(self, audio_config, vad_config, on_chunk):
        cap = AudioCapture(audio_config, vad_config, on_chunk)
        assert cap._audio_config is audio_config
        assert cap._vad_config is vad_config
        assert cap._on_chunk_ready is on_chunk

    def test_init_not_recording(self, capture):
        assert capture.is_recording is False
        assert capture._stream is None
        assert capture._recording is False

    def test_init_rms_zero(self, capture):
        assert capture.current_rms == 0.0

    def test_init_pre_roll_maxlen(self, capture, vad_config, audio_config):
        expected = int(vad_config.pre_roll_s * audio_config.sample_rate)
        assert capture._vad.pre_roll.maxlen == expected


class TestAudioCaptureProperties:
    """Tests for AudioCapture properties."""

    def test_is_recording_false_initially(self, capture):
        assert capture.is_recording is False

    def test_is_recording_true_when_set(self, capture):
        capture._recording = True
        assert capture.is_recording is True

    def test_current_rms(self, capture):
        capture._current_rms = 0.5
        assert capture.current_rms == 0.5

    def test_recording_duration_zero_when_not_recording(self, capture):
        assert capture.recording_duration == 0.0

    def test_recording_duration_positive_when_recording(self, capture):
        capture._recording = True
        capture._recording_started_at = time.time() - 1.5
        duration = capture.recording_duration
        assert 1.0 <= duration <= 2.0


class TestAudioCaptureStartStop:
    """Tests for start/stop lifecycle."""

    @patch.object(AudioCapture, "_start_stream")
    def test_start_sets_recording(self, mock_start, capture):
        capture.start()
        assert capture._recording is True
        mock_start.assert_called_once()

    @patch.object(AudioCapture, "_start_stream")
    def test_start_resets_vad(self, mock_start, capture):
        capture._vad.in_speech = True
        capture._vad.last_speech_time = 99.0
        capture._vad.current_chunk.append(np.zeros(100, dtype=np.float32))

        capture.start()

        assert capture._vad.in_speech is False
        assert capture._vad.last_speech_time == 0.0
        assert len(capture._vad.current_chunk) == 0

    @patch.object(AudioCapture, "_start_stream")
    def test_start_idempotent(self, mock_start, capture):
        """Starting twice should only start stream once."""
        capture.start()
        capture.start()
        assert mock_start.call_count == 1

    @patch.object(AudioCapture, "_finalize_chunk")
    @patch.object(AudioCapture, "_stop_stream")
    @patch.object(AudioCapture, "_start_stream")
    def test_stop_returns_duration(self, mock_start, mock_stop, mock_finalize, capture):
        capture.start()
        time.sleep(0.05)
        duration = capture.stop()

        assert duration > 0
        assert capture._recording is False
        mock_stop.assert_called_once()
        mock_finalize.assert_called_once_with(force=True)

    @patch.object(AudioCapture, "_start_stream")
    def test_stop_when_not_recording_returns_zero(self, mock_start, capture):
        duration = capture.stop()
        assert duration == 0.0

    @patch.object(AudioCapture, "_finalize_chunk")
    @patch.object(AudioCapture, "_stop_stream")
    @patch.object(AudioCapture, "_start_stream")
    def test_stop_sets_recording_false_before_stream_stop(
        self, mock_start, mock_stop, mock_finalize, capture
    ):
        """_recording should be False before _stop_stream to prevent race."""
        recording_during_stop = []

        def check_recording():
            recording_during_stop.append(capture._recording)

        mock_stop.side_effect = check_recording

        capture.start()
        capture.stop()

        # _recording should be False when _stop_stream is called
        assert recording_during_stop == [False]


class TestAudioCaptureStream:
    """Tests for stream management."""

    @patch("dictate.audio.sd")
    def test_start_stream_creates_input_stream(self, mock_sd, capture):
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        capture._start_stream()

        mock_sd.InputStream.assert_called_once_with(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=capture._audio_config.block_size,
            device=None,
            callback=capture._audio_callback,
        )
        mock_stream.start.assert_called_once()
        assert capture._stream is mock_stream

    @patch("dictate.audio.sd")
    def test_start_stream_port_audio_invalid_device(self, mock_sd, capture):
        """PortAudioError with 'invalid device' should raise RuntimeError."""
        import sounddevice as sd

        mock_sd.InputStream.side_effect = sd.PortAudioError("Invalid device id")
        # Make the except clause find the real PortAudioError
        mock_sd.PortAudioError = sd.PortAudioError

        capture._recording = True  # Pre-set so _start_stream's except can clear it
        with pytest.raises(RuntimeError, match="Audio device not available"):
            capture._start_stream()

        assert capture._recording is False
        assert capture._stream is None

    @patch("dictate.audio.sd")
    def test_start_stream_port_audio_other_error(self, mock_sd, capture):
        """Non-device PortAudioError should re-raise."""
        import sounddevice as sd

        mock_sd.InputStream.side_effect = sd.PortAudioError("buffer underrun")
        mock_sd.PortAudioError = sd.PortAudioError

        capture._recording = True
        with pytest.raises(sd.PortAudioError):
            capture._start_stream()

        assert capture._recording is False

    @patch("dictate.audio.sd")
    def test_start_stream_unexpected_error(self, mock_sd, capture):
        """Non-PortAudioError should propagate but still clean up."""
        import sounddevice as sd

        # Need PortAudioError to be a real exception class so the except clause works
        mock_sd.PortAudioError = sd.PortAudioError
        mock_sd.InputStream.side_effect = RuntimeError("unexpected")

        capture._recording = True
        with pytest.raises(RuntimeError):
            capture._start_stream()

        assert capture._recording is False
        assert capture._stream is None

    def test_stop_stream_no_stream(self, capture):
        """Stopping when no stream should not raise."""
        capture._stream = None
        capture._stop_stream()  # Should not raise

    def test_stop_stream_closes_and_stops(self, capture):
        mock_stream = MagicMock()
        capture._stream = mock_stream

        capture._stop_stream()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert capture._stream is None

    def test_stop_stream_error_still_cleans_up(self, capture):
        """If stream.stop() raises, should still set stream to None."""
        mock_stream = MagicMock()
        mock_stream.stop.side_effect = Exception("error")
        capture._stream = mock_stream

        capture._stop_stream()

        assert capture._stream is None


class TestAudioCallback:
    """Tests for the audio callback and VAD processing."""

    def test_callback_extracts_first_channel(self, capture):
        """Callback should take first channel of multi-channel input."""
        capture._recording = True
        indata = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        with patch.object(capture, "_process_audio_block") as mock_process:
            capture._audio_callback(indata, 2, {}, MagicMock(input_overflow=False, input_underflow=False, priming_output=False))
            call_args = mock_process.call_args[0][0]
            np.testing.assert_array_almost_equal(call_args, [0.1, 0.3])

    def test_process_block_not_recording(self, capture):
        """Should return early if not recording."""
        capture._recording = False
        audio = np.array([0.5, 0.5], dtype=np.float32)
        capture._process_audio_block(audio)
        # No crash, no state change
        assert capture._vad.in_speech is False

    def test_process_block_updates_rms(self, capture):
        capture._recording = True
        # Loud audio — RMS should be high
        audio = np.ones(100, dtype=np.float32) * 0.5
        capture._process_audio_block(audio)
        assert capture._current_rms > 0

    def test_process_block_speech_detection(self, capture, vad_config):
        """Loud audio should trigger speech detection."""
        capture._recording = True
        # Audio well above threshold
        audio = np.ones(100, dtype=np.float32) * 0.5
        capture._process_audio_block(audio)

        assert capture._vad.in_speech is True
        assert len(capture._vad.current_chunk) > 0

    def test_process_block_silence_after_speech(self, capture, vad_config):
        """Silence after speech should trigger chunk finalization after timeout."""
        capture._recording = True

        # First: loud audio → speech starts
        loud = np.ones(100, dtype=np.float32) * 0.5
        capture._process_audio_block(loud)
        assert capture._vad.in_speech is True

        # Simulate time passing beyond silence timeout
        capture._vad.last_speech_time = time.time() - vad_config.silence_timeout_s - 0.1

        # Then: quiet audio → should finalize
        with patch.object(capture, "_finalize_chunk") as mock_finalize:
            quiet = np.ones(100, dtype=np.float32) * 0.001  # Below threshold
            capture._process_audio_block(quiet)
            mock_finalize.assert_called_once_with(force=False)

    def test_process_block_quiet_no_speech(self, capture):
        """Very quiet audio should not trigger speech."""
        capture._recording = True
        quiet = np.ones(100, dtype=np.float32) * 0.001
        capture._process_audio_block(quiet)

        assert capture._vad.in_speech is False

    def test_process_block_pre_roll_accumulates(self, capture):
        """Pre-roll buffer should accumulate samples."""
        capture._recording = True
        audio = np.array([0.001] * 50, dtype=np.float32)
        capture._process_audio_block(audio)
        assert len(capture._vad.pre_roll) == 50

    def test_process_block_speech_includes_pre_roll(self, capture):
        """When speech starts, pre-roll should be included in chunk."""
        capture._recording = True

        # Feed some quiet pre-roll
        quiet = np.array([0.001] * 50, dtype=np.float32)
        capture._process_audio_block(quiet)
        pre_roll_len = len(capture._vad.pre_roll)
        assert pre_roll_len == 50

        # Now loud audio triggers speech — pre_roll should be in chunk
        loud = np.ones(100, dtype=np.float32) * 0.5
        capture._process_audio_block(loud)

        assert capture._vad.in_speech is True
        assert len(capture._vad.current_chunk) == 2  # pre_roll + audio


class TestFinalizeChunk:
    """Tests for chunk finalization and callback."""

    def test_finalize_empty_chunk(self, capture, on_chunk):
        """Empty chunk should not trigger callback."""
        capture._finalize_chunk(force=False)
        on_chunk.assert_not_called()

    def test_finalize_short_chunk_skipped(self, capture, on_chunk, audio_config):
        """Short chunks should be skipped when not forced."""
        # 10 samples at 16kHz = 0.000625s — way under MIN_CHUNK_DURATION_SECONDS
        capture._vad.current_chunk = [np.zeros(10, dtype=np.float32)]
        capture._finalize_chunk(force=False)
        on_chunk.assert_not_called()

    def test_finalize_short_chunk_forced(self, capture, on_chunk):
        """Short chunks should still process when forced (stop recording)."""
        capture._vad.current_chunk = [np.zeros(10, dtype=np.float32)]
        capture._finalize_chunk(force=True)
        on_chunk.assert_called_once()

    def test_finalize_long_chunk_calls_callback(self, capture, on_chunk, audio_config):
        """Chunks above min duration should trigger callback."""
        # Enough samples for MIN_CHUNK_DURATION_SECONDS + margin
        n = int(audio_config.sample_rate * (MIN_CHUNK_DURATION_SECONDS + 0.1))
        capture._vad.current_chunk = [np.zeros(n, dtype=np.float32)]
        capture._finalize_chunk(force=False)
        on_chunk.assert_called_once()

    def test_finalize_produces_int16(self, capture, on_chunk):
        """Callback should receive int16 audio."""
        n = int(16000 * 0.5)  # 0.5s
        capture._vad.current_chunk = [np.ones(n, dtype=np.float32) * 0.5]
        capture._finalize_chunk(force=True)

        chunk = on_chunk.call_args[0][0]
        assert chunk.dtype == np.int16

    def test_finalize_clips_audio(self, capture, on_chunk):
        """Audio should be clipped to [-1, 1] before int16 conversion."""
        n = int(16000 * 0.5)
        # Values outside [-1, 1]
        loud = np.ones(n, dtype=np.float32) * 2.0
        capture._vad.current_chunk = [loud]
        capture._finalize_chunk(force=True)

        chunk = on_chunk.call_args[0][0]
        # Clipped to 1.0 * 32767 = 32767
        assert chunk.max() <= 32767
        assert chunk.min() >= -32768

    def test_finalize_clears_current_chunk(self, capture, on_chunk):
        """After finalization, current_chunk should be empty."""
        n = int(16000 * 0.5)
        capture._vad.current_chunk = [np.zeros(n, dtype=np.float32)]
        capture._finalize_chunk(force=True)
        assert len(capture._vad.current_chunk) == 0

    def test_finalize_concatenates_multiple_chunks(self, capture, on_chunk):
        """Multiple chunks should be concatenated."""
        n1 = int(16000 * 0.3)
        n2 = int(16000 * 0.3)
        capture._vad.current_chunk = [
            np.ones(n1, dtype=np.float32) * 0.1,
            np.ones(n2, dtype=np.float32) * 0.2,
        ]
        capture._finalize_chunk(force=True)

        chunk = on_chunk.call_args[0][0]
        assert len(chunk) == n1 + n2


class TestAudioCallbackFlags:
    """Tests for handling sounddevice callback status flags."""

    def test_callback_with_input_overflow(self, capture):
        """Input overflow should log warning but not crash."""
        capture._recording = True
        indata = np.zeros((100, 1), dtype=np.float32)
        status = MagicMock()
        status.input_overflow = True
        status.input_underflow = False
        status.priming_output = False
        status.__bool__ = lambda self: True

        # Should not raise
        capture._audio_callback(indata, 100, {}, status)

    def test_callback_with_input_underflow(self, capture):
        """Input underflow should log warning but not crash."""
        capture._recording = True
        indata = np.zeros((100, 1), dtype=np.float32)
        status = MagicMock()
        status.input_overflow = False
        status.input_underflow = True
        status.priming_output = False
        status.__bool__ = lambda self: True

        capture._audio_callback(indata, 100, {}, status)

    def test_callback_with_priming_output(self, capture):
        """Priming output should log debug but not crash."""
        capture._recording = True
        indata = np.zeros((100, 1), dtype=np.float32)
        status = MagicMock()
        status.input_overflow = False
        status.input_underflow = False
        status.priming_output = True
        status.__bool__ = lambda self: True

        capture._audio_callback(indata, 100, {}, status)
