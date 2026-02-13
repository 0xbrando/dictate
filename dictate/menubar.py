"""Menu bar app for push-to-talk dictation."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import rumps

from dictate.audio import AudioCapture, list_input_devices, play_tone
from dictate.config import (
    Config,
    OutputMode,
    WHISPER_MODEL,
    delete_cached_model,
    get_cached_model_disk_size,
    get_model_size_str,
    is_model_cached,
)
from dictate.model_download import download_model, is_download_in_progress
from collections import deque
from pathlib import Path

from dictate.icons import cleanup_temp_files, generate_reactive_icon, get_icon_path
from dictate.output import TextAggregator, create_output_handler
from dictate.presets import (
    INPUT_LANGUAGES,
    OUTPUT_LANGUAGES,
    PTT_KEYS,
    QUALITY_PRESETS,
    SOUND_PRESETS,
    STT_PRESETS,
    WRITING_STYLES,
    Preferences,
)
from dictate.transcribe import TranscriptionPipeline
from dictate import __version__ as DICTATE_VERSION

try:
    from packaging.version import Version as parse_version
except ImportError:
    # Fallback for simple version comparison
    def parse_version(v):
        class SimpleVersion:
            def __init__(self, version_str):
                self.parts = [int(x) for x in version_str.split('.') if x.isdigit()]
            def __gt__(self, other):
                return self.parts > other.parts
            def __eq__(self, other):
                return self.parts == other.parts
            def __ge__(self, other):
                return self.parts >= other.parts
        return SimpleVersion(v)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

KEYBOARD_RELEASE_DELAY_SECONDS = 0.05
SHUTDOWN_TIMEOUT_SECONDS = 2.0
MAX_NOTIFICATION_LENGTH = 120
MAX_RECENT_ITEMS = 10
RECENT_MENU_TRUNCATE = 50
UI_POLL_INTERVAL_SECONDS = 0.1
DEVICE_POLL_INTERVAL_SECONDS = 3.0

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
        self._pipeline_lock = threading.Lock()
        self._output = create_output_handler(OutputMode.TYPE)
        self._aggregator = TextAggregator()
        self._worker: threading.Thread | None = None

        self._recording_locked = False
        self._ptt_held = False
        self._is_recording = False
        self._paused = False
        self._rms_history: deque[float] = deque([0.0] * 5, maxlen=5)

        self._recent: list[str] = []
        self._active_downloads: dict[str, threading.Thread] = {}
        self._download_progress: dict[str, float] = {}
        self._reload_in_progress = False
        self._last_device_check: float = 0.0
        self._known_device_ids: set[int] = {d.index for d in list_input_devices()}
        whisper_cached = is_model_cached(WHISPER_MODEL)
        llm_cached = is_model_cached(self._prefs.llm_model.hf_repo)
        if whisper_cached and llm_cached:
            init_status = "◐ Loading models..."
        else:
            init_status = "◐ Downloading models (first launch)..."
        self._status_item = rumps.MenuItem(init_status, callback=lambda _: None)
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
                status_text = msg[1]
                if status_text in ("Ready", "Ready (cleanup skipped)"):
                    self._status_item.title = f"● {status_text}"
                elif status_text == "Paused":
                    self._status_item.title = f"○ {status_text}"
                elif "Recording" in status_text or "Processing" in status_text:
                    self._status_item.title = f"● {status_text}"
                elif "error" in status_text.lower() or "failed" in status_text.lower() or "No microphone" in status_text:
                    self._status_item.title = f"○ {status_text}"
                else:
                    self._status_item.title = f"◐ {status_text}"
            elif kind == "icon":
                self.icon = get_icon_path(msg[1])
            elif kind == "notify":
                text = str(msg[1])[:MAX_NOTIFICATION_LENGTH]
                rumps.notification("Dictate", "", text)
            elif kind == "rebuild_menu":
                self._build_menu()
            elif kind == "recent":
                self._recent.insert(0, str(msg[1]))
                self._recent = self._recent[:MAX_RECENT_ITEMS]
                self._build_menu()

        # Detect audio device hot-plug/unplug
        now = time.time()
        if now - self._last_device_check >= DEVICE_POLL_INTERVAL_SECONDS:
            self._last_device_check = now
            self._check_device_changes()

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

    def _check_device_changes(self) -> None:
        """Detect audio device hot-plug/unplug and rebuild menu."""
        try:
            current_ids = {d.index for d in list_input_devices()}
        except Exception:
            logger.debug("Device enumeration failed, skipping check", exc_info=True)
            return
        if current_ids == self._known_device_ids:
            return
        added = current_ids - self._known_device_ids
        removed = self._known_device_ids - current_ids
        self._known_device_ids = current_ids
        if added:
            logger.info("Audio device(s) connected: %s", added)
        if removed:
            logger.info("Audio device(s) disconnected: %s", removed)
        self._build_menu()
        # Create AudioCapture if it doesn't exist yet and devices are now available
        if self._audio is None and current_ids and self._pipeline is not None:
            logger.info("Creating AudioCapture after device hot-plug")
            self._audio = AudioCapture(
                audio_config=self._config.audio,
                vad_config=self._config.vad,
                on_chunk_ready=self._on_chunk_ready,
            )

    # ── Menu construction ──────────────────────────────────────────

    def _build_menu(self) -> None:
        self.menu.clear()
        pause_label = "Resume Dictation" if self._paused else "Pause Dictation"

        # Build Advanced submenu for power-user features
        advanced_menu = rumps.MenuItem("Advanced...")
        advanced_menu.add(self._build_llm_toggle())
        advanced_menu.add(self._build_login_toggle())
        advanced_menu.add(None)
        advanced_menu.add(self._build_stt_menu())
        advanced_menu.add(self._build_ptt_key_menu())
        advanced_menu.add(self._build_sound_menu())
        advanced_menu.add(None)
        advanced_menu.add(self._build_input_lang_menu())
        advanced_menu.add(self._build_output_lang_menu())
        advanced_menu.add(None)
        advanced_menu.add(self._build_endpoint_menu())
        advanced_menu.add(self._build_dictionary_menu())
        advanced_menu.add(self._build_manage_models_menu())

        self.menu = [
            self._status_item,
            None,
            self._build_writing_style_menu(),
            self._build_quality_menu(),
            self._build_mic_menu(),
            None,
            self._build_recent_menu(),
            None,
            advanced_menu,
            None,
            rumps.MenuItem(pause_label, callback=self._on_pause_toggle),
            rumps.MenuItem("Quit Dictate", callback=self._on_quit, key="q"),
        ]

    def _build_mic_menu(self) -> rumps.MenuItem:
        mic_menu = rumps.MenuItem("Input Device")
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

    def _build_ptt_key_menu(self) -> rumps.MenuItem:
        ptt_menu = rumps.MenuItem("Push-to-Talk Key")
        for key_id, label in PTT_KEYS:
            item = rumps.MenuItem(label, callback=self._on_ptt_key_select)
            item.state = key_id == self._prefs.ptt_key
            item._key_id = key_id  # type: ignore[attr-defined]
            ptt_menu.add(item)
        return ptt_menu

    def _build_quality_menu(self) -> rumps.MenuItem:
        from dictate.config import LLMBackend
        quality_menu = rumps.MenuItem("Quality")
        for i, preset in enumerate(QUALITY_PRESETS):
            is_api = preset.backend == LLMBackend.API
            # For API backend, show the discovered model name
            if is_api:
                display_label = self._get_api_preset_label()
            else:
                # Check download status for local models
                hf_repo = preset.llm_model.hf_repo
                cached = is_model_cached(hf_repo)
                downloading = is_download_in_progress(hf_repo)
                progress = self._download_progress.get(hf_repo, 0)
                
                if cached:
                    # Model is downloaded - show checkmark
                    display_label = f"{preset.label} ✓"
                elif downloading:
                    # Model is downloading - show progress
                    display_label = f"{preset.label} ⏳ {int(progress)}%"
                else:
                    # Model not downloaded - show size and download indicator
                    size = get_model_size_str(hf_repo)
                    display_label = f"{preset.label} ↓ {size}"
                    
            item = rumps.MenuItem(display_label, callback=self._on_quality_select)
            item.state = i == self._prefs.quality_preset
            item._preset_index = i  # type: ignore[attr-defined]
            quality_menu.add(item)
        return quality_menu

    def _get_api_preset_label(self) -> str:
        """Get the label for the API preset showing discovered model."""
        display = self._prefs.discovered_model_display
        if display and "No local model" not in display:
            return f"Local: {display}"
        return "Local Server (configure endpoint)"

    def _build_endpoint_menu(self) -> rumps.MenuItem:
        """Build menu for LLM endpoint configuration."""
        endpoint_menu = rumps.MenuItem("LLM Endpoint")

        # Show current endpoint
        current = rumps.MenuItem(f"Current: {self._prefs.llm_endpoint}")
        current.set_callback(None)
        endpoint_menu.add(current)

        # Show discovered model status
        display = self._prefs.discovered_model_display
        if display:
            status = rumps.MenuItem(f"Model: {display}")
            status.set_callback(None)
            endpoint_menu.add(status)

        endpoint_menu.add(None)  # separator

        # Preset endpoints
        presets = [
            ("Ollama (11434)", "localhost:11434"),
            ("LM Studio (1234)", "localhost:1234"),
            ("vLLM (8000)", "localhost:8000"),
        ]
        for label, endpoint in presets:
            item = rumps.MenuItem(label, callback=self._on_endpoint_preset_select)
            item.state = self._prefs.llm_endpoint == endpoint
            item._endpoint = endpoint  # type: ignore[attr-defined]
            endpoint_menu.add(item)

        endpoint_menu.add(None)  # separator
        endpoint_menu.add(rumps.MenuItem("Set Custom...", callback=self._on_endpoint_custom))
        return endpoint_menu

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

    def _build_writing_style_menu(self) -> rumps.MenuItem:
        style_menu = rumps.MenuItem("Writing Style")
        for key, label, description in WRITING_STYLES:
            title = f"{label} - {description}" if description else label
            item = rumps.MenuItem(title, callback=self._on_writing_style_select)
            item.state = key == self._prefs.writing_style
            item._style_key = key  # type: ignore[attr-defined]
            style_menu.add(item)
        return style_menu

    def _build_login_toggle(self) -> rumps.MenuItem:
        item = rumps.MenuItem("Launch at Login", callback=self._on_login_toggle)
        item.state = self._is_launch_at_login()
        return item

    def _build_llm_toggle(self) -> rumps.MenuItem:
        item = rumps.MenuItem("LLM Cleanup", callback=self._on_llm_toggle)
        item.state = self._prefs.llm_cleanup
        return item

    def _build_stt_menu(self) -> rumps.MenuItem:
        from dictate.config import STTEngine

        stt_menu = rumps.MenuItem("STT Engine")
        for i, preset in enumerate(STT_PRESETS):
            # Only show Parakeet if the package is installed
            if preset.engine == STTEngine.PARAKEET:
                try:
                    import parakeet_mlx  # noqa: F401
                except ImportError:
                    continue
            desc = f" — {preset.description}" if preset.description else ""
            item = rumps.MenuItem(f"{preset.label}{desc}", callback=self._on_stt_select)
            item.state = i == self._prefs.stt_preset
            item._stt_index = i  # type: ignore[attr-defined]
            stt_menu.add(item)
        return stt_menu

    def _build_dictionary_menu(self) -> rumps.MenuItem:
        dict_menu = rumps.MenuItem("Personal Dictionary")
        words = Preferences.load_dictionary()
        dict_menu.add(rumps.MenuItem("Add Word…", callback=self._on_dict_add))
        if words:
            dict_menu.add(None)  # separator
            for word in words:
                item = rumps.MenuItem(f"✕  {word}", callback=self._on_dict_remove)
                item._dict_word = word  # type: ignore[attr-defined]
                dict_menu.add(item)
            dict_menu.add(None)
            dict_menu.add(
                rumps.MenuItem("Clear All", callback=self._on_dict_clear)
            )
        else:
            dict_menu.add(None)
            no_words = rumps.MenuItem("No words yet")
            no_words.set_callback(None)
            dict_menu.add(no_words)
        return dict_menu

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

    def _build_manage_models_menu(self) -> rumps.MenuItem:
        """Build the Manage Models submenu for viewing and deleting cached models."""
        from dictate.config import LLMBackend

        manage_menu = rumps.MenuItem("Manage Models")

        # List all QUALITY_PRESETS models (skip API backend)
        for i, preset in enumerate(QUALITY_PRESETS):
            if preset.backend == LLMBackend.API:
                continue

            hf_repo = preset.llm_model.hf_repo
            cached = is_model_cached(hf_repo)

            if cached:
                size = get_cached_model_disk_size(hf_repo)
                label = f"{preset.label} ({size})"
                item = rumps.MenuItem(label, callback=self._on_delete_model)
                item._preset_index = i  # type: ignore[attr-defined]
                item._preset_label = preset.label  # type: ignore[attr-defined]
                item._hf_repo = hf_repo  # type: ignore[attr-defined]
                item._size = size  # type: ignore[attr-defined]
            else:
                label = f"{preset.label} — Not downloaded"
                item = rumps.MenuItem(label)
                item.set_callback(None)

            manage_menu.add(item)

        manage_menu.add(None)  # separator

        # Show cache location
        cache_path = str(Path.home() / ".cache" / "huggingface" / "hub")
        cache_info = rumps.MenuItem(f"Cache: {cache_path}")
        cache_info.set_callback(None)
        manage_menu.add(cache_info)

        # Open in Finder
        open_finder = rumps.MenuItem("Open in Finder", callback=self._on_open_cache_folder)
        manage_menu.add(open_finder)

        return manage_menu

    def _on_delete_model(self, sender: rumps.MenuItem) -> None:
        """Handle deletion of a cached model with confirmation."""
        preset_label = getattr(sender, "_preset_label", "Unknown")
        hf_repo = getattr(sender, "_hf_repo", "")
        size = getattr(sender, "_size", "Unknown")

        if not hf_repo:
            return

        # Show confirmation dialog
        result = rumps.alert(
            title="Delete Model",
            message=f"Delete {preset_label}?\n\nThis will free {size}.",
            ok="Delete",
            cancel="Cancel",
        )

        if result == 1:  # Delete button clicked
            if delete_cached_model(hf_repo):
                self._post_ui("notify", f"Deleted {preset_label}")

                # If the deleted model was the active preset, switch to first available cached model
                current_preset_idx = self._prefs.quality_preset
                current_preset = QUALITY_PRESETS[current_preset_idx]

                if current_preset.backend != LLMBackend.API and current_preset.llm_model.hf_repo == hf_repo:
                    # Find first available cached model
                    for i, preset in enumerate(QUALITY_PRESETS):
                        if preset.backend == LLMBackend.API:
                            # Switch to API backend
                            self._prefs.quality_preset = i
                            self._prefs.save()
                            self._apply_prefs()
                            self._reload_pipeline()
                            break
                        elif is_model_cached(preset.llm_model.hf_repo):
                            # Switch to this cached model
                            self._prefs.quality_preset = i
                            self._prefs.save()
                            self._apply_prefs()
                            self._reload_pipeline()
                            break

                self._post_ui("rebuild_menu")
            else:
                rumps.alert("Error", f"Failed to delete {preset_label}")

    def _on_open_cache_folder(self, _sender: rumps.MenuItem) -> None:
        """Open the HuggingFace cache folder in Finder."""
        import subprocess

        cache_path = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_path.exists():
            subprocess.run(["open", str(cache_path)])
        else:
            rumps.alert("Cache Not Found", f"Cache directory does not exist:\n{cache_path}")

    # ── Menu callbacks ─────────────────────────────────────────────

    def _on_pause_toggle(self, _sender: rumps.MenuItem) -> None:
        self._paused = not self._paused
        if self._paused:
            if self._audio and self._audio.is_recording:
                self._audio.stop()
                self._is_recording = False
            self._post_ui("status", "Paused")
            self._post_ui("icon", "idle")
            logger.info("Dictation paused")
        else:
            self._post_ui("status", "Ready")
            logger.info("Dictation resumed")
        self._build_menu()

    def _on_ptt_key_select(self, sender: rumps.MenuItem) -> None:
        key_id = sender._key_id  # type: ignore[attr-defined]
        self._prefs.ptt_key = key_id
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()

    def _on_mic_select(self, sender: rumps.MenuItem) -> None:
        idx = sender._device_index  # type: ignore[attr-defined]
        self._prefs.device_id = idx
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()

    def _on_quality_select(self, sender: rumps.MenuItem) -> None:
        from dictate.config import LLMBackend
        
        idx = sender._preset_index  # type: ignore[attr-defined]
        if idx == self._prefs.quality_preset:
            return
            
        preset = QUALITY_PRESETS[idx]
        
        # For API backend, just switch immediately
        if preset.backend == LLMBackend.API:
            self._prefs.quality_preset = idx
            self._prefs.save()
            self._apply_prefs()
            self._build_menu()
            self._reload_pipeline()
            return
            
        # For local models, check if cached
        hf_repo = preset.llm_model.hf_repo
        if is_model_cached(hf_repo):
            # Model is cached, switch immediately
            self._prefs.quality_preset = idx
            self._prefs.save()
            self._apply_prefs()
            self._build_menu()
            self._reload_pipeline()
        elif is_download_in_progress(hf_repo):
            # Download already in progress, just show message
            self._post_ui("notify", f"Download already in progress for {preset.label}")
        else:
            # Start download
            self._start_model_download(idx, hf_repo)
            
    def _start_model_download(self, preset_index: int, hf_repo: str) -> None:
        """Start downloading a model in a background thread."""
        preset = QUALITY_PRESETS[preset_index]
        
        # Initialize progress tracking
        self._download_progress[hf_repo] = 0.0
        self._post_ui("status", f"Downloading {preset.label}...")
        self._post_ui("rebuild_menu")  # Update menu to show downloading state
        
        def progress_callback(percent: float) -> None:
            """Called periodically with download progress (0-100)."""
            self._download_progress[hf_repo] = percent
            self._post_ui("status", f"Downloading {preset.label} ({int(percent)}%)...")
            # Rebuild menu periodically to show progress (throttle to avoid UI spam)
            if int(percent) % 10 == 0 or percent >= 99:
                self._post_ui("rebuild_menu")
        
        def download_complete(success: bool, error: Exception | None = None) -> None:
            """Called when download completes or fails."""
            if hf_repo in self._active_downloads:
                del self._active_downloads[hf_repo]
            
            if success:
                self._download_progress[hf_repo] = 100.0
                self._post_ui("notify", f"Downloaded {preset.label}")
                # Auto-switch to this preset
                self._prefs.quality_preset = preset_index
                self._prefs.save()
                self._apply_prefs()
                self._post_ui("status", f"Loading {preset.label}...")
                self._post_ui("rebuild_menu")
                # Reload pipeline with new model
                self._reload_pipeline()
            else:
                # Download failed
                error_msg = str(error) if error else "Unknown error"
                self._post_ui("notify", f"Download failed: {error_msg}")
                if hf_repo in self._download_progress:
                    del self._download_progress[hf_repo]
                self._post_ui("status", "Download failed")
                self._post_ui("rebuild_menu")
        
        def do_download() -> None:
            """Background thread function to download the model."""
            try:
                download_model(hf_repo, progress_callback=progress_callback)
                download_complete(True)
            except Exception as e:
                logger.exception("Download failed for %s", hf_repo)
                download_complete(False, e)
        
        # Start download in background thread
        thread = threading.Thread(target=do_download, daemon=True)
        self._active_downloads[hf_repo] = thread
        thread.start()

    def _on_endpoint_preset_select(self, sender: rumps.MenuItem) -> None:
        """Handle selection of a preset endpoint."""
        endpoint = sender._endpoint  # type: ignore[attr-defined]
        if endpoint == self._prefs.llm_endpoint:
            return
        self._prefs.update_endpoint(endpoint)
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()
        self._reload_pipeline()

    def _on_endpoint_custom(self, _sender: rumps.MenuItem) -> None:
        """Handle custom endpoint input."""
        window = rumps.Window(
            message="Enter LLM endpoint (host:port):",
            title="LLM Endpoint",
            default_text=self._prefs.llm_endpoint,
            ok="Set",
            cancel="Cancel",
        )
        response = window.run()
        if response.clicked and response.text.strip():
            new_endpoint = response.text.strip()
            # Remove protocol prefix if user included it
            if new_endpoint.startswith("http://"):
                new_endpoint = new_endpoint[7:]
            elif new_endpoint.startswith("https://"):
                new_endpoint = new_endpoint[8:]
            # Remove path if included
            new_endpoint = new_endpoint.split("/")[0]
            # Validate host:port format
            import re
            if not re.match(r'^[a-zA-Z0-9._-]+(:\d{1,5})?$', new_endpoint):
                logger.warning("Invalid endpoint format: %s", new_endpoint)
                rumps.alert("Invalid endpoint", "Enter a valid host:port (e.g. localhost:8005)")
                return
            if new_endpoint and new_endpoint != self._prefs.llm_endpoint:
                self._prefs.update_endpoint(new_endpoint)
                self._prefs.save()
                self._apply_prefs()
                self._build_menu()
                self._reload_pipeline()
                logger.info("Updated LLM endpoint to: %s", new_endpoint)

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

    def _on_writing_style_select(self, sender: rumps.MenuItem) -> None:
        key = sender._style_key  # type: ignore[attr-defined]
        self._prefs.writing_style = key
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()

    def _on_login_toggle(self, _sender: rumps.MenuItem) -> None:
        enabled = not self._is_launch_at_login()
        self._set_launch_at_login(enabled)
        self._build_menu()

    @staticmethod
    def _launch_agent_path() -> Path:
        return Path.home() / "Library" / "LaunchAgents" / "com.dictate.app.plist"

    def _is_launch_at_login(self) -> bool:
        return self._launch_agent_path().exists()

    def _set_launch_at_login(self, enabled: bool) -> None:
        import plistlib
        import sys

        plist_path = self._launch_agent_path()
        if enabled:
            python = sys.executable
            project_dir = str(Path(__file__).resolve().parent.parent)
            plist_data = {
                "Label": "com.dictate.app",
                "ProgramArguments": [python, "-m", "dictate"],
                "WorkingDirectory": project_dir,
                "RunAtLoad": True,
            }
            plist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(plist_path, "wb") as f:
                plistlib.dump(plist_data, f)
            logger.info("Launch at Login enabled: %s", plist_path)
        else:
            if plist_path.exists():
                plist_path.unlink()
                logger.info("Launch at Login disabled")

    def _on_llm_toggle(self, sender: rumps.MenuItem) -> None:
        self._prefs.llm_cleanup = not self._prefs.llm_cleanup
        sender.state = self._prefs.llm_cleanup
        self._prefs.save()
        self._apply_prefs()

    def _on_stt_select(self, sender: rumps.MenuItem) -> None:
        idx = sender._stt_index  # type: ignore[attr-defined]
        if idx == self._prefs.stt_preset:
            return
        self._prefs.stt_preset = idx
        self._prefs.save()
        self._apply_prefs()
        self._build_menu()
        self._reload_pipeline()

    def _on_dict_add(self, _sender: rumps.MenuItem) -> None:
        window = rumps.Window(
            message="Enter a word or phrase to always spell correctly:",
            title="Add to Personal Dictionary",
            default_text="",
            ok="Add",
            cancel="Cancel",
        )
        response = window.run()
        if response.clicked and response.text.strip():
            words = Preferences.load_dictionary()
            new_word = response.text.strip()
            if new_word not in words:
                words.append(new_word)
                Preferences.save_dictionary(words)
                self._config.llm.dictionary = words
                self._build_menu()
                logger.info("Added '%s' to dictionary", new_word)

    def _on_dict_remove(self, sender: rumps.MenuItem) -> None:
        word = sender._dict_word  # type: ignore[attr-defined]
        words = Preferences.load_dictionary()
        if word in words:
            words.remove(word)
            Preferences.save_dictionary(words)
            self._config.llm.dictionary = words or None
            self._build_menu()
            logger.info("Removed '%s' from dictionary", word)

    def _on_dict_clear(self, _sender: rumps.MenuItem) -> None:
        Preferences.save_dictionary([])
        self._config.llm.dictionary = None
        self._build_menu()
        logger.info("Cleared personal dictionary")

    def _on_recent_select(self, sender: rumps.MenuItem) -> None:
        text = sender._full_text  # type: ignore[attr-defined]
        self._output.output(text)

    def _on_clear_recent(self, _sender: rumps.MenuItem) -> None:
        self._recent.clear()
        self._build_menu()

    def _on_quit(self, _sender: rumps.MenuItem) -> None:
        self.shutdown()
        rumps.quit_application()
        # Force exit — rumps sometimes leaves the process alive
        import os
        os._exit(0)

    # ── Preferences → Config ───────────────────────────────────────

    def _apply_prefs(self) -> None:
        self._config.audio.device_id = self._prefs.device_id
        self._config.whisper.language = self._prefs.whisper_language
        self._config.whisper.engine = self._prefs.stt_engine
        self._config.whisper.model = self._prefs.stt_model
        self._config.llm.output_language = self._prefs.llm_output_language
        self._config.llm.model_choice = self._prefs.llm_model
        self._config.llm.backend = self._prefs.backend
        self._config.llm.api_url = self._prefs.validated_api_url
        self._config.llm.enabled = self._prefs.llm_cleanup
        self._config.llm.writing_style = self._prefs.writing_style
        self._config.llm.dictionary = Preferences.load_dictionary() or None
        self._config.keybinds.ptt_key = self._prefs.ptt_pynput_key
        sound = self._prefs.sound
        if sound.start_hz == 0:
            self._config.tones.enabled = False
        else:
            self._config.tones.enabled = True
            self._config.tones.start_hz = sound.start_hz
            self._config.tones.stop_hz = sound.stop_hz
            self._config.tones.style = sound.style

    # ── Pipeline management ────────────────────────────────────────

    def _reload_pipeline(self) -> None:
        if self._reload_in_progress:
            logger.info("Pipeline reload already in progress, skipping")
            return
        self._reload_in_progress = True
        self._post_ui("icon", "idle")
        self._post_ui("status", "Loading models...")

        def _load() -> None:
            try:
                new_pipeline = TranscriptionPipeline(
                    whisper_config=self._config.whisper,
                    llm_config=self._config.llm,
                )
                new_pipeline.set_sample_rate(self._config.audio.sample_rate)
                new_pipeline.preload_models(
                    on_progress=lambda msg: self._post_ui("status", msg)
                )
                with self._pipeline_lock:
                    self._pipeline = new_pipeline
                self._post_ui("status", "Ready")
                self._post_ui("rebuild_menu")
            except Exception:
                logger.exception("Failed to reload pipeline")
                self._post_ui("status", "Model load failed")
            finally:
                self._reload_in_progress = False

        threading.Thread(target=_load, daemon=True).start()

    # ── Lifecycle ──────────────────────────────────────────────────

    def start_app(self) -> None:
        init_thread = threading.Thread(target=self._init_pipeline, daemon=True)
        init_thread.start()
        self._start_keyboard_listener()
        # Start update check in background after 10-second delay
        update_check_thread = threading.Thread(target=self._check_for_update, daemon=True)
        update_check_thread.start()
        self.run()

    def _init_pipeline(self) -> None:
        try:
            new_pipeline = TranscriptionPipeline(
                whisper_config=self._config.whisper,
                llm_config=self._config.llm,
            )
            new_pipeline.set_sample_rate(self._config.audio.sample_rate)
            new_pipeline.preload_models(
                on_progress=lambda msg: self._post_ui("status", msg)
            )
            with self._pipeline_lock:
                self._pipeline = new_pipeline
            logger.info("Pipeline ready")
            self._post_ui("status", "Ready")
            self._post_ui("rebuild_menu")
        except ImportError as e:
            logger.error("Missing dependency: %s", e)
            self._post_ui("status", f"Missing package: {e.name or e}")
        except Exception:
            logger.exception("Failed to initialize pipeline")
            self._post_ui("status", "Model load failed")

        if list_input_devices():
            self._audio = AudioCapture(
                audio_config=self._config.audio,
                vad_config=self._config.vad,
                on_chunk_ready=self._on_chunk_ready,
            )
        else:
            logger.warning("No input devices found — waiting for hot-plug")
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── Keyboard listener (pynput — own CFRunLoop thread) ──────────

    def _start_keyboard_listener(self) -> None:
        from pynput import keyboard

        def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
            if self._paused:
                return
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
        if self._audio is None:
            self._post_ui("status", "No microphone detected")
            return
        if self._audio.is_recording:
            return
        try:
            play_tone(
                self._config.tones,
                self._config.tones.start_hz,
                self._config.audio.sample_rate,
            )
            self._audio.start()
        except Exception:
            logger.exception("Failed to start recording")
            self._post_ui("status", "Mic error — check connection")
            return
        self._is_recording = True
        self._rms_history = deque([0.0] * 5, maxlen=5)
        self._post_ui("status", "Recording (Space to lock)")

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
            self._post_ui("status", "Ready")
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
        with self._pipeline_lock:
            pipeline = self._pipeline
        if pipeline is None:
            return
        try:
            text = pipeline.process(audio)
            if text:
                self._emit_output(text)
                if pipeline.last_cleanup_failed:
                    self._post_ui("status", "Ready (cleanup skipped)")
                else:
                    self._post_ui("status", "Ready")
            else:
                self._post_ui("status", "Ready")
        except RuntimeError as e:
            logger.error("Processing error: %s", e)
            self._post_ui("status", f"Error: {e}")
        except Exception:
            logger.exception("Processing error")
            self._post_ui("status", "Processing error — try again")

    def _emit_output(self, text: str) -> None:
        self._aggregator.append(text)
        # Add to recent first so text isn't lost if output fails
        self._post_ui("recent", text)
        try:
            self._output.output(text)
            self._post_ui("notify", text)
        except Exception:
            logger.exception("Output error — text saved to Recent")
            self._post_ui("status", "Output error — check Recent")

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

    def _check_for_update(self) -> None:
        """Check PyPI for updates in background thread. Silently fails if no internet."""
        # Wait 10 seconds so it doesn't slow startup
        time.sleep(10)
        
        try:
            from urllib.request import urlopen, Request
            from urllib.error import URLError, HTTPError
            
            req = Request(
                "https://pypi.org/pypi/dictate-mlx/json",
                headers={"User-Agent": f"dictate-mlx/{DICTATE_VERSION}"}
            )
            
            MAX_RESPONSE_BYTES = 1_048_576  # 1 MB cap
            with urlopen(req, timeout=10) as response:
                raw = response.read(MAX_RESPONSE_BYTES)
                if len(raw) >= MAX_RESPONSE_BYTES:
                    return  # Response suspiciously large
                data = json.loads(raw.decode("utf-8"))

            import re as _re
            latest_version = data.get("info", {}).get("version", "")
            if not latest_version or not _re.match(r'^\d+\.\d+\.\d+$', latest_version):
                return  # Missing or invalid version format

            current = parse_version(DICTATE_VERSION)
            latest = parse_version(latest_version)
            
            if latest > current:
                # Update available - notify and update status
                notification_text = f"v{DICTATE_VERSION} → v{latest_version}. Run: pip install --upgrade dictate-mlx"
                rumps.notification("Dictate update available!", "", notification_text)
                self._post_ui("status", f"● Update available (v{latest_version})")
                logger.info(f"Update available: {DICTATE_VERSION} → {latest_version}")
        except (URLError, HTTPError, json.JSONDecodeError, ValueError):
            # Silently fail if no internet or other errors
            logger.debug("Update check failed (no internet or error)", exc_info=True)
        except Exception:
            # Catch-all to never crash
            logger.debug("Update check unexpected error", exc_info=True)
