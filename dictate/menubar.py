"""Menu bar app for push-to-talk dictation."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import pyperclip
import rumps

from dictate.audio import AudioCapture, list_input_devices, play_tone
from dictate.config import Config, OutputMode
from collections import deque

from dictate.icons import cleanup_temp_files, generate_reactive_icon, get_icon_path
from dictate.output import TextAggregator, create_output_handler
from dictate.presets import (
    INPUT_LANGUAGES,
    OUTPUT_LANGUAGES,
    QUALITY_PRESETS,
    SOUND_PRESETS,
    Preferences,
)
from dictate.transcribe import TranscriptionPipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

KEYBOARD_RELEASE_DELAY_SECONDS = 0.05
SHUTDOWN_TIMEOUT_SECONDS = 2.0
MAX_NOTIFICATION_LENGTH = 120
MAX_RECENT_ITEMS = 10
RECENT_MENU_TRUNCATE = 50
UI_POLL_INTERVAL_SECONDS = 0.1

# Reactive waveform: maps RMS to bar heights
BAR_WEIGHTS = [0.65, 0.85, 1.0, 0.80, 0.60]
RMS_REFERENCE = 0.12
MIN_BAR_H = 5
MAX_BAR_H = 30


class DictateMenuBarApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("", quit_button=None)
        self.template = True
        self.icon = get_icon_path("idle")

        self._prefs = Preferences.load()
        self._config = Config.from_env()
        self._apply_prefs()

        self._work_queue: queue.Queue[NDArray[np.int16]] = queue.Queue()
        self._ui_queue: queue.Queue[tuple[str, ...]] = queue.Queue()
        self._stop_event = threading.Event()

        self._audio: AudioCapture | None = None
        self._pipeline: TranscriptionPipeline | None = None
        self._output = create_output_handler(OutputMode.TYPE)
        self._aggregator = TextAggregator()
        self._worker: threading.Thread | None = None

        self._recording_locked = False
        self._ptt_held = False
        self._is_recording = False
        self._rms_history: deque[float] = deque([0.0] * 5, maxlen=5)

        self._recent: list[str] = []
        self._status_item = rumps.MenuItem("Status: Loading...")
        self._build_menu()

    # ── UI queue (thread-safe main-thread updates) ─────────────────

    @rumps.timer(UI_POLL_INTERVAL_SECONDS)
    def _poll_ui(self, _timer: rumps.Timer) -> None:
        """Drain the UI queue on the main thread."""
        while True:
            try:
                msg = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == "status":
                self._status_item.title = f"Status: {msg[1]}"
            elif kind == "icon":
                self.icon = get_icon_path(msg[1])
            elif kind == "notify":
                text = str(msg[1])[:MAX_NOTIFICATION_LENGTH]
                rumps.notification("Dictate", "", text)
            elif kind == "recent":
                self._recent.insert(0, str(msg[1]))
                self._recent = self._recent[:MAX_RECENT_ITEMS]
                self._build_menu()

        # Reactive waveform: bars follow actual voice level
        if self._is_recording and self._audio:
            rms = self._audio.current_rms
            self._rms_history.append(rms)
            heights = []
            for i, r in enumerate(self._rms_history):
                level = min(1.0, (r / RMS_REFERENCE) ** 0.6) if r > 0 else 0.0
                h = int(MIN_BAR_H + (MAX_BAR_H - MIN_BAR_H) * level * BAR_WEIGHTS[i])
                heights.append(max(MIN_BAR_H, h))
            self.icon = generate_reactive_icon(heights)

    def _post_ui(self, *msg: str) -> None:
        self._ui_queue.put(msg)

    # ── Menu construction ──────────────────────────────────────────

    def _build_menu(self) -> None:
        self.menu.clear()
        self.menu = [
            self._status_item,
            None,
            self._build_mic_menu(),
            self._build_quality_menu(),
            self._build_sound_menu(),
            None,
            self._build_input_lang_menu(),
            self._build_output_lang_menu(),
            self._build_llm_toggle(),
            None,
            self._build_recent_menu(),
            None,
            rumps.MenuItem("Quit Dictate", callback=self._on_quit, key="q"),
        ]

    def _build_mic_menu(self) -> rumps.MenuItem:
        mic_menu = rumps.MenuItem("Microphone")
        devices = list_input_devices()
        for dev in devices:
            is_selected = dev.index == self._prefs.device_id or (
                self._prefs.device_id is None and dev.is_default
            )
            title = f"[{dev.index}] {dev.name}"
            item = rumps.MenuItem(title, callback=self._on_mic_select)
            item.state = is_selected
            item._device_index = dev.index  # type: ignore[attr-defined]
            mic_menu.add(item)
        return mic_menu

    def _build_quality_menu(self) -> rumps.MenuItem:
        quality_menu = rumps.MenuItem("Quality")
        for i, preset in enumerate(QUALITY_PRESETS):
            item = rumps.MenuItem(preset.label, callback=self._on_quality_select)
            item.state = i == self._prefs.quality_preset
            item._preset_index = i  # type: ignore[attr-defined]
            quality_menu.add(item)
        return quality_menu

    def _build_input_lang_menu(self) -> rumps.MenuItem:
        lang_menu = rumps.MenuItem("Input Language")
        for code, label in INPUT_LANGUAGES:
            item = rumps.MenuItem(label, callback=self._on_input_lang_select)
            item.state = code == self._prefs.input_language
            item._lang_code = code  # type: ignore[attr-defined]
            lang_menu.add(item)
        return lang_menu

    def _build_output_lang_menu(self) -> rumps.MenuItem:
        lang_menu = rumps.MenuItem("Output Language")
        for code, label in OUTPUT_LANGUAGES:
            item = rumps.MenuItem(label, callback=self._on_output_lang_select)
            item.state = code == self._prefs.output_language
            item._lang_code = code  # type: ignore[attr-defined]
            lang_menu.add(item)
        return lang_menu

    def _build_sound_menu(self) -> rumps.MenuItem:
        sound_menu = rumps.MenuItem("Sounds")
        for i, preset in enumerate(SOUND_PRESETS):
            item = rumps.MenuItem(preset.label, callback=self._on_sound_select)
            item.state = i == self._prefs.sound_preset
            item._sound_index = i  # type: ignore[attr-defined]
            sound_menu.add(item)
        return sound_menu

    def _build_llm_toggle(self) -> rumps.MenuItem:
        item = rumps.MenuItem("LLM Cleanup", callback=self._on_llm_toggle)
        item.state = self._prefs.llm_cleanup
        return item

    def _build_recent_menu(self) -> rumps.MenuItem:
        recent_menu = rumps.MenuItem("Recent")
        if not self._recent:
            item = rumps.MenuItem("No recent items")
            item.set_callback(None)
            recent_menu.add(item)
        else:
            for i, text in enumerate(self._recent):
                truncated = text[:RECENT_MENU_TRUNCATE]
                if len(text) > RECENT_MENU_TRUNCATE:
                    truncated += "..."
                item = rumps.MenuItem(truncated, callback=self._on_recent_select)
                item._full_text = text  # type: ignore[attr-defined]
                recent_menu.add(item)
            recent_menu.add(None)  # separator
            recent_menu.add(
                rumps.MenuItem("Clear Recent", callback=self._on_clear_recent)
            )
        return recent_menu

    # ── Menu callbacks ─────────────────────────────────────────────

    def _on_mic_select(self, sender: rumps.MenuItem) -> None:
        idx = sender._device_index  # type: ignore[attr-defined]
        self._prefs.device_id = idx
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()

    def _on_quality_select(self, sender: rumps.MenuItem) -> None:
        idx = sender._preset_index  # type: ignore[attr-defined]
        if idx == self._prefs.quality_preset:
            return
        self._prefs.quality_preset = idx
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()
        self._reload_pipeline()

    def _on_input_lang_select(self, sender: rumps.MenuItem) -> None:
        code = sender._lang_code  # type: ignore[attr-defined]
        self._prefs.input_language = code
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()

    def _on_output_lang_select(self, sender: rumps.MenuItem) -> None:
        code = sender._lang_code  # type: ignore[attr-defined]
        self._prefs.output_language = code
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()

    def _on_sound_select(self, sender: rumps.MenuItem) -> None:
        idx = sender._sound_index  # type: ignore[attr-defined]
        self._prefs.sound_preset = idx
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()
        # Play a preview of the start tone
        sound = SOUND_PRESETS[idx]
        if sound.start_hz > 0:
            play_tone(self._config.tones, sound.start_hz, self._config.audio.sample_rate)

    def _on_llm_toggle(self, sender: rumps.MenuItem) -> None:
        self._prefs.llm_cleanup = not self._prefs.llm_cleanup
        sender.state = self._prefs.llm_cleanup
        self._prefs.save()
        self._apply_prefs()

    def _on_recent_select(self, sender: rumps.MenuItem) -> None:
        text = sender._full_text  # type: ignore[attr-defined]
        pyperclip.copy(text)
        rumps.notification("Dictate", "", "Copied to clipboard")

    def _on_clear_recent(self, _sender: rumps.MenuItem) -> None:
        self._recent.clear()
        self._build_menu()

    def _on_quit(self, _sender: rumps.MenuItem) -> None:
        self.shutdown()
        rumps.quit_application()

    # ── Preferences → Config ───────────────────────────────────────

    def _apply_prefs(self) -> None:
        self._config.audio.device_id = self._prefs.device_id
        self._config.whisper.language = self._prefs.whisper_language
        self._config.llm.output_language = self._prefs.llm_output_language
        self._config.llm.model_choice = self._prefs.llm_model
        self._config.llm.enabled = self._prefs.llm_cleanup
        sound = self._prefs.sound
        if sound.start_hz == 0:
            self._config.tones.enabled = False
        else:
            self._config.tones.enabled = True
            self._config.tones.start_hz = sound.start_hz
            self._config.tones.stop_hz = sound.stop_hz

    # ── Pipeline management ────────────────────────────────────────

    def _reload_pipeline(self) -> None:
        self._post_ui("icon", "idle")
        self._post_ui("status", "Loading models...")

        def _load() -> None:
            try:
                self._pipeline = TranscriptionPipeline(
                    whisper_config=self._config.whisper,
                    llm_config=self._config.llm,
                )
                self._pipeline.set_sample_rate(self._config.audio.sample_rate)
                self._pipeline.preload_models()
                self._post_ui("status", "Idle")
            except Exception:
                logger.exception("Failed to reload pipeline")
                self._post_ui("status", "Model load failed")

        threading.Thread(target=_load, daemon=True).start()

    # ── Lifecycle ──────────────────────────────────────────────────

    def start_app(self) -> None:
        init_thread = threading.Thread(target=self._init_pipeline, daemon=True)
        init_thread.start()
        self._start_keyboard_listener()
        self.run()

    def _init_pipeline(self) -> None:
        try:
            self._pipeline = TranscriptionPipeline(
                whisper_config=self._config.whisper,
                llm_config=self._config.llm,
            )
            self._pipeline.set_sample_rate(self._config.audio.sample_rate)
            self._pipeline.preload_models()
            logger.info("Pipeline ready")
            self._post_ui("status", "Idle")
        except Exception:
            logger.exception("Failed to initialize pipeline")
            self._post_ui("status", "Model load failed")

        self._audio = AudioCapture(
            audio_config=self._config.audio,
            vad_config=self._config.vad,
            on_chunk_ready=self._on_chunk_ready,
        )
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── Keyboard listener (pynput — own CFRunLoop thread) ──────────

    def _start_keyboard_listener(self) -> None:
        from pynput import keyboard

        def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if key == self._config.keybinds.ptt_key:
                self._ptt_held = True
                if self._recording_locked:
                    self._recording_locked = False
                    time.sleep(KEYBOARD_RELEASE_DELAY_SECONDS)
                    self._stop_recording()
                else:
                    self._start_recording()
                return

            is_space = key == keyboard.Key.space or (
                isinstance(key, keyboard.KeyCode)
                and getattr(key, "char", None) in (" ", "\xa0")
            )
            if is_space and self._ptt_held:
                if self._audio and self._audio.is_recording and not self._recording_locked:
                    self._recording_locked = True
                    logger.info("Recording locked")

        def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if key == self._config.keybinds.ptt_key:
                self._ptt_held = False
                if not self._recording_locked:
                    time.sleep(KEYBOARD_RELEASE_DELAY_SECONDS)
                    self._stop_recording()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    # ── Recording ──────────────────────────────────────────────────

    def _start_recording(self) -> None:
        if self._audio is None or self._audio.is_recording:
            return
        play_tone(
            self._config.tones,
            self._config.tones.start_hz,
            self._config.audio.sample_rate,
        )
        self._audio.start()
        self._is_recording = True
        self._rms_history = deque([0.0] * 5, maxlen=5)
        self._post_ui("status", "Recording...")

    def _stop_recording(self) -> None:
        if self._audio is None or not self._audio.is_recording:
            return
        play_tone(
            self._config.tones,
            self._config.tones.stop_hz,
            self._config.audio.sample_rate,
        )
        duration = self._audio.stop()
        self._is_recording = False
        self._post_ui("icon", "idle")

        if duration < self._config.min_hold_to_process_s:
            self._post_ui("status", "Idle")
            return

        self._post_ui("status", "Processing...")

    # ── Audio + worker ─────────────────────────────────────────────

    def _on_chunk_ready(self, audio: NDArray[np.int16]) -> None:
        self._work_queue.put(audio)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio = self._work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop_event.is_set():
                break
            if audio.size == 0:
                continue
            self._process_chunk(audio)

    def _process_chunk(self, audio: NDArray[np.int16]) -> None:
        if self._pipeline is None:
            return
        try:
            text = self._pipeline.process(audio)
            if text:
                self._emit_output(text)
        except Exception:
            logger.exception("Processing error")
        finally:
            self._post_ui("status", "Idle")

    def _emit_output(self, text: str) -> None:
        self._aggregator.append(text)
        try:
            self._output.output(text)
            logger.info("Output: %s", text)
            self._post_ui("notify", text)
            self._post_ui("recent", text)
        except Exception:
            logger.exception("Output error")

    # ── Shutdown ───────────────────────────────────────────────────

    def shutdown(self) -> None:
        logger.info("Shutting down...")
        if self._audio and self._audio.is_recording:
            self._audio.stop()
        self._stop_event.set()
        try:
            self._work_queue.put_nowait(np.zeros((0,), dtype=np.int16))
        except queue.Full:
            pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        if hasattr(self, "_listener"):
            self._listener.stop()
        self._cleanup_icon_temp_files()

    @staticmethod
    def _cleanup_icon_temp_files() -> None:
        try:
            cleanup_temp_files()
        except Exception:
            logger.debug("Icon temp file cleanup failed", exc_info=True)
