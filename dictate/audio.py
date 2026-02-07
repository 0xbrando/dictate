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


def play_tone(
    config: "ToneConfig",
    frequency_hz: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    if not config.enabled:
        return

    n_samples = int(sample_rate * config.duration_s)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    tone = np.sin(2.0 * np.pi * frequency_hz * t) * config.volume

    fade_samples = max(1, int(FADE_DURATION_SECONDS * sample_rate))
    if fade_samples * 2 < n_samples:
        window = np.ones(n_samples, dtype=np.float32)
        window[:fade_samples] = np.linspace(0, 1, fade_samples, dtype=np.float32)
        window[-fade_samples:] = np.linspace(1, 0, fade_samples, dtype=np.float32)
        tone *= window

    sd.play(tone.astype(np.float32), sample_rate, blocking=False)


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
        with self._lock:
            if not self._recording:
                return 0.0
            self._recording = False
            duration = time.time() - self._recording_started_at

        self._stop_stream()
        self._finalize_chunk(force=True)
        return duration

    def _start_stream(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._audio_config.sample_rate,
            channels=self._audio_config.channels,
            dtype="float32",
            blocksize=self._audio_config.block_size,
            device=self._audio_config.device_id,
            callback=self._audio_callback,
        )
        self._stream.start()

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
