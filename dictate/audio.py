from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dictate.config import AudioConfig, ToneConfig, VADConfig

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_PRE_ROLL_SAMPLES = 4000
FADE_DURATION_SECONDS = 0.008
MIN_CHUNK_DURATION_SECONDS = 0.20
INT16_MAX = 32767.0
RMS_EPSILON = 1e-12
AUDIO_CLIP_MIN = -1.0
AUDIO_CLIP_MAX = 1.0
FIRST_CHANNEL_INDEX = 0


@dataclass
class AudioDevice:
    index: int
    name: str
    is_default: bool = False

    def __str__(self) -> str:
        marker = " (DEFAULT)" if self.is_default else ""
        return f"[{self.index}] {self.name}{marker}"


def list_input_devices() -> list[AudioDevice]:
    devices = sd.query_devices()
    default_input = sd.default.device[FIRST_CHANNEL_INDEX]
    
    input_devices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:  # type: ignore[index]
            input_devices.append(
                AudioDevice(
                    index=i,
                    name=dev["name"],  # type: ignore[index]
                    is_default=(i == default_input),
                )
            )
    return input_devices


def get_device_name(device_id: int | None) -> str:
    if device_id is not None:
        info = sd.query_devices(device_id)
        return info["name"]  # type: ignore[index,return-value]

    default_id = sd.default.device[FIRST_CHANNEL_INDEX]
    if default_id < 0:
        return "(no input device found)"
    info = sd.query_devices(default_id)
    return info["name"]  # type: ignore[index,return-value]


TONE_SAMPLE_RATE = 44_100


def play_tone(
    config: "ToneConfig",
    frequency_hz: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    if not config.enabled:
        return

    sr = TONE_SAMPLE_RATE
    style = getattr(config, "style", "simple")
    vol = config.volume

    synthesizers = {
        "soft_pop": _synth_soft_pop,
        "chime": _synth_chime,
        "warm": _synth_warm,
        "click": _synth_click,
        "marimba": _synth_marimba,
    }

    synth = synthesizers.get(style)
    if synth is not None:
        tone = synth(frequency_hz, vol, sr)
    else:
        tone = _synth_simple(frequency_hz, config.duration_s, vol, sr)

    sd.play(tone.astype(np.float32), sr, blocking=False)


def _synth_simple(freq: int, duration_s: float, vol: float, sr: int) -> "NDArray":
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    tone = np.sin(2.0 * np.pi * freq * t) * vol
    fade = max(1, int(FADE_DURATION_SECONDS * sr))
    if fade * 2 < n:
        w = np.ones(n, dtype=np.float32)
        w[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
        w[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
        tone *= w
    return tone


def _synth_soft_pop(freq: int, vol: float, sr: int) -> "NDArray":
    n = int(sr * 0.06)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 60)
    tone = np.sin(2.0 * np.pi * freq * t) * env * vol
    fade_in = min(int(sr * 0.002), n)
    tone[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    return tone


def _synth_chime(freq: int, vol: float, sr: int) -> "NDArray":
    n = int(sr * 0.10)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 25)
    tone = (
        np.sin(2.0 * np.pi * freq * t) * 0.7
        + np.sin(2.0 * np.pi * freq * 2 * t) * 0.2
        + np.sin(2.0 * np.pi * freq * 3 * t) * 0.1
    ) * env * vol * 0.8
    fade_in = min(int(sr * 0.003), n)
    tone[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    return tone


def _synth_warm(freq: int, vol: float, sr: int) -> "NDArray":
    n = int(sr * 0.08)
    t = np.arange(n, dtype=np.float32) / sr
    attack = min(int(sr * 0.004), n)
    env = np.exp(-t * 30)
    env[:attack] *= np.linspace(0, 1, attack, dtype=np.float32)
    tone = (
        np.sin(2.0 * np.pi * freq * t) * 0.55
        + np.sin(2.0 * np.pi * freq * 2.0 * t) * 0.25
        + np.sin(2.0 * np.pi * freq * 3.0 * t) * 0.12
        + np.sin(2.0 * np.pi * freq * 4.0 * t) * 0.08
    ) * env * vol * 0.8
    return tone


def _synth_click(freq: int, vol: float, sr: int) -> "NDArray":
    n = int(sr * 0.015)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 250)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1, 1, n).astype(np.float32)
    tone = (
        np.sin(2.0 * np.pi * freq * t) * 0.6 + noise * 0.4
    ) * env * vol * 0.5
    return tone


def _synth_marimba(freq: int, vol: float, sr: int) -> "NDArray":
    n = int(sr * 0.10)
    t = np.arange(n, dtype=np.float32) / sr
    env = np.exp(-t * 25)
    attack = min(int(sr * 0.002), n)
    env[:attack] *= np.linspace(0, 1, attack, dtype=np.float32)
    tone = (
        np.sin(2.0 * np.pi * freq * t) * 0.6
        + np.sin(2.0 * np.pi * freq * 3.98 * t) * 0.15
        + np.sin(2.0 * np.pi * freq * 2.01 * t) * 0.2
        + np.sin(2.0 * np.pi * freq * 9.1 * t) * 0.05
    ) * env * vol * 0.8
    return tone


@dataclass
class VADState:
    in_speech: bool = False
    last_speech_time: float = 0.0
    pre_roll: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_PRE_ROLL_SAMPLES))
    current_chunk: list["NDArray[np.float32]"] = field(default_factory=list)

    def reset(self, pre_roll_samples: int) -> None:
        self.in_speech = False
        self.last_speech_time = 0.0
        self.pre_roll = deque(maxlen=pre_roll_samples)
        self.current_chunk = []


class AudioCapture:
    def __init__(
        self,
        audio_config: "AudioConfig",
        vad_config: "VADConfig",
        on_chunk_ready: Callable[["NDArray[np.int16]"], None],
    ) -> None:
        self._audio_config = audio_config
        self._vad_config = vad_config
        self._on_chunk_ready = on_chunk_ready

        self._stream: sd.InputStream | None = None
        self._recording = False
        self._recording_started_at = 0.0
        self._current_rms: float = 0.0
        self._lock = threading.Lock()

        pre_roll_samples = int(vad_config.pre_roll_s * audio_config.sample_rate)
        self._vad = VADState(pre_roll=deque(maxlen=pre_roll_samples))

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def current_rms(self) -> float:
        return self._current_rms

    @property
    def recording_duration(self) -> float:
        if not self._recording:
            return 0.0
        return time.time() - self._recording_started_at

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._recording_started_at = time.time()
            pre_roll_samples = int(
                self._vad_config.pre_roll_s * self._audio_config.sample_rate
            )
            self._vad.reset(pre_roll_samples)

        self._start_stream()

    def stop(self) -> float:
        """Stop recording and return the duration.
        
        Note: We set _recording = False before stopping the stream to prevent
        the callback from processing stale data during the race with the callback thread.
        """
        with self._lock:
            if not self._recording:
                return 0.0
            # Set recording to False BEFORE stopping stream to prevent
            # callback from processing stale data during the race
            self._recording = False
            duration = time.time() - self._recording_started_at

        self._stop_stream()
        self._finalize_chunk(force=True)
        return duration

    def _start_stream(self) -> None:
        try:
            self._stream = sd.InputStream(
                samplerate=self._audio_config.sample_rate,
                channels=self._audio_config.channels,
                dtype="float32",
                blocksize=self._audio_config.block_size,
                device=self._audio_config.device_id,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            error_msg = str(e).lower()
            if "invalid device" in error_msg or "device not found" in error_msg or "bad device" in error_msg:
                logger.error("Audio device not found or invalid: %s", e)
                self._stream = None
                # Set recording to False since we can't start
                with self._lock:
                    self._recording = False
                raise RuntimeError(f"Audio device not available: {e}") from e
            else:
                logger.error("Failed to start audio stream: %s", e)
                self._stream = None
                with self._lock:
                    self._recording = False
                raise
        except Exception as e:
            logger.exception("Unexpected error starting audio stream")
            self._stream = None
            with self._lock:
                self._recording = False
            raise

    def _stop_stream(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None

    def _audio_callback(
        self,
        indata: "NDArray[np.float32]",
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            # Handle specific callback flags
            if status.input_overflow:
                logger.warning("Audio input overflow - samples dropped due to slow processing")
            if status.input_underflow:
                logger.warning("Audio input underflow - device may be disconnecting")
            if status.priming_output:
                logger.debug("Audio output priming (expected during startup)")
            # Log the full status for debugging
            if not (status.input_overflow or status.input_underflow):
                logger.warning("Audio callback status: %s", status)

        audio = indata[:, FIRST_CHANNEL_INDEX].astype(np.float32, copy=True)
        self._process_audio_block(audio)

    def _process_audio_block(self, audio: "NDArray[np.float32]") -> None:
        now = time.time()

        with self._lock:
            if not self._recording:
                return

            self._vad.pre_roll.extend(audio.tolist())

            rms = float(np.sqrt(np.mean(audio * audio) + RMS_EPSILON))
            self._current_rms = rms
            is_speech = rms >= self._vad_config.rms_threshold

            if is_speech:
                self._vad.last_speech_time = now
                if not self._vad.in_speech:
                    self._vad.in_speech = True
                    logger.info("Speech detected")
                    if self._vad.pre_roll:
                        pre_audio = np.array(self._vad.pre_roll, dtype=np.float32)
                        self._vad.current_chunk.append(pre_audio)
                self._vad.current_chunk.append(audio)
            else:
                if self._vad.in_speech:
                    self._vad.current_chunk.append(audio)
                    silence_duration = now - self._vad.last_speech_time
                    if silence_duration >= self._vad_config.silence_timeout_s:
                        self._finalize_chunk(force=False)
                        self._vad.in_speech = False

    def _finalize_chunk(self, force: bool) -> None:
        if not self._vad.current_chunk:
            return

        chunk = np.concatenate(self._vad.current_chunk).astype(np.float32)
        chunk = np.clip(chunk, AUDIO_CLIP_MIN, AUDIO_CLIP_MAX)
        chunk_i16 = (chunk * INT16_MAX).astype(np.int16)

        duration_s = len(chunk_i16) / self._audio_config.sample_rate
        self._vad.current_chunk = []

        if duration_s < MIN_CHUNK_DURATION_SECONDS and not force:
            logger.debug("Skipping short chunk (%.2fs)", duration_s)
            return

        self._on_chunk_ready(chunk_i16)
