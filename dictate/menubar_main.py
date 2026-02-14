"""Entry point for the menu bar app: python -m dictate.menubar_main"""

import platform
import subprocess
import sys

if platform.system() != "Darwin":
    print("Dictate requires macOS with Apple Silicon. See https://github.com/0xbrando/dictate", file=sys.stderr)
    sys.exit(1)

import fcntl
import logging
import os
import signal
import time
from pathlib import Path

# Disable HuggingFace telemetry — all inference is local
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

# Load .env file if it exists (before importing config)
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

LOCK_FILE = Path.home() / "Library" / "Application Support" / "Dictate" / "dictate.lock"
LOG_FILE = Path.home() / "Library" / "Logs" / "Dictate" / "dictate.log"


def _show_status() -> int:
    """Show system info and model status for troubleshooting."""
    from dictate import __version__
    from dictate.config import is_model_cached, get_cached_model_disk_size, WHISPER_MODEL
    from dictate.presets import (
        Preferences, QUALITY_PRESETS, STT_PRESETS, WRITING_STYLES,
        PREFS_DIR, PREFS_FILE,
    )

    W = "\033[97m"   # bright white
    G = "\033[32m"   # green
    R = "\033[31m"   # red
    D = "\033[2m"    # dim
    B = "\033[1m"    # bold
    N = "\033[0m"    # reset

    print(f"\n{W}{B}Dictate Status{N}  {D}v{__version__}{N}\n")

    # System info
    import platform
    chip = platform.processor() or "unknown"
    mac_ver = platform.mac_ver()[0] or "unknown"
    py_ver = platform.python_version()
    print(f"  {W}System{N}")
    print(f"  macOS {mac_ver} · Python {py_ver} · {chip}")
    print()

    # Check if running
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "dictate.menubar_main"],
            capture_output=True, text=True, check=False,
        )
        pids = [p for p in result.stdout.strip().split("\n") if p and p != str(os.getpid())]
        if pids:
            print(f"  {W}Status{N}  {G}● Running{N} (PID {pids[0]})")
        else:
            print(f"  {W}Status{N}  {D}○ Not running{N}")
    except Exception:
        print(f"  {W}Status{N}  {D}? Unknown{N}")
    print()

    # Models
    print(f"  {W}Models{N}")
    whisper_cached = is_model_cached(WHISPER_MODEL)
    whisper_size = get_cached_model_disk_size(WHISPER_MODEL) if whisper_cached else "not downloaded"
    status = f"{G}✓{N}" if whisper_cached else f"{R}✗{N}"
    print(f"  {status} Whisper: {WHISPER_MODEL} ({whisper_size})")

    # Check Parakeet
    try:
        import parakeet_mlx  # noqa: F401
        print(f"  {G}✓{N} Parakeet: installed")
    except ImportError:
        print(f"  {D}○ Parakeet: not installed{N}")

    # LLM models
    from dictate.config import LLMModel
    for model in LLMModel:
        cached = is_model_cached(model.hf_repo)
        size = get_cached_model_disk_size(model.hf_repo) if cached else "not downloaded"
        status = f"{G}✓{N}" if cached else f"{D}○{N}"
        print(f"  {status} LLM {model.value}: {model.hf_repo} ({size})")
    print()

    # Preferences
    print(f"  {W}Preferences{N}")
    print(f"  Config dir: {PREFS_DIR}")
    if PREFS_FILE.exists():
        prefs = Preferences.load()
        quality = QUALITY_PRESETS[prefs.quality_preset] if prefs.quality_preset < len(QUALITY_PRESETS) else None
        stt = STT_PRESETS[prefs.stt_preset] if prefs.stt_preset < len(STT_PRESETS) else None
        style_name = next((s[1] for s in WRITING_STYLES if s[0] == prefs.writing_style), prefs.writing_style)
        print(f"  Quality: {quality.label if quality else 'Unknown'}")
        print(f"  STT: {stt.label if stt else 'Unknown'}")
        print(f"  Writing style: {style_name}")
        print(f"  LLM cleanup: {'on' if prefs.llm_cleanup else 'off'}")
        print(f"  Input language: {prefs.input_language}")
        print(f"  Output language: {prefs.output_language}")
        print(f"  PTT key: {prefs.ptt_key}")
    else:
        print(f"  {D}No preferences file (will use defaults){N}")
    print()

    # Log file
    if LOG_FILE.exists():
        size = LOG_FILE.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  {W}Logs{N}")
        print(f"  {LOG_FILE} ({size_str})")
    print()

    return 0


def _run_update() -> int:
    """Run pip install --upgrade and restart Dictate."""
    print("Updating Dictate...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "dictate-mlx"],
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"Update failed with exit code {result.returncode}")
            return result.returncode
        print("Update successful!")
    except Exception as e:
        print(f"Update failed: {e}")
        return 1
    
    # Kill any running instance
    print("Restarting Dictate...")
    try:
        subprocess.run(["pkill", "-f", "dictate\\.menubar_main"], capture_output=True, check=False)
        time.sleep(0.5)  # Give it time to shut down
    except Exception:
        pass
    
    # Relaunch Dictate in background
    try:
        subprocess.Popen(
            [sys.executable, "-m", "dictate.menubar_main"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )
        print("Dictate restarted.")
    except Exception as e:
        print(f"Failed to restart: {e}")
        print("Please run 'dictate' manually to start the app.")
    
    return 0


def _acquire_singleton_lock() -> int | None:
    """Acquire an exclusive lock to prevent duplicate instances.

    Returns the lock fd on success, or None if another instance is running.
    The fd must be kept open for the lifetime of the process.
    """
    import errno
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.ftruncate(fd, os.lseek(fd, 0, os.SEEK_CUR))
        return fd
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            return None  # Another instance holds the lock
        # Unexpected OS error — log it so the user can debug
        print(f"Lock file error: {e}", file=sys.stderr)
        return None


def _daemonize() -> None:
    """Re-launch as a detached background process.

    os.fork() is incompatible with macOS AppKit/ObjC — the forked child
    crashes with objc_initializeAfterForkError.  Instead we spawn a fresh
    subprocess with --foreground so the child runs the app directly.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(LOG_FILE, "a")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "dictate.menubar_main", "--foreground"],
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        print(f"Background launch failed ({e}), running in foreground", file=sys.stderr)
        log_fd.close()
        return
    log_fd.close()
    os._exit(0)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("urllib3", "httpx", "mlx", "transformers", "tokenizers", "sounddevice"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> int:
    # Handle --version flag
    if "--version" in sys.argv or "-V" in sys.argv:
        from dictate import __version__
        print(f"dictate {__version__}")
        return 0

    # Handle --help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        from dictate import __version__
        print(f"dictate v{__version__} — push-to-talk voice dictation for macOS")
        print()
        print("Usage: dictate [COMMAND] [OPTIONS]")
        print()
        print("Commands:")
        print("  (default)       Launch Dictate in the menu bar")
        print("  status          Show system info and model status")
        print("  update          Update to the latest version")
        print()
        print("Options:")
        print("  -f, --foreground  Run in foreground (show logs)")
        print("  -V, --version     Show version and exit")
        print("  -h, --help        Show this help and exit")
        print()
        print("https://github.com/0xbrando/dictate")
        return 0

    # Handle status command
    if "status" in sys.argv:
        return _show_status()

    # Handle update command before anything else
    if "update" in sys.argv or "--update" in sys.argv:
        return _run_update()
    
    # Daemonize before anything else — detach from terminal
    foreground = "--foreground" in sys.argv or "-f" in sys.argv
    if not foreground and sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        from dictate import __version__
        Y = "\033[33m"    # yellow/dark orange
        O = "\033[93m"    # bright yellow/orange
        W = "\033[97m"    # bright white
        D = "\033[2m"     # dim
        B = "\033[1m"     # bold
        R = "\033[0m"     # reset
        print(f"""
{O}       ___      __        __
{O}  ____/ (_)____/ /_____ _/ /____
{Y} / __  / / ___/ __/ __ `/ __/ _ \\
{Y}/ /_/ / / /__/ /_/ /_/ / /_/  __/
{Y}\\__,_/_/\\___/\\__/\\__,_/\\__/\\___/{R}

  {W}{B}speak. it types.{R}  {D}v{__version__} · 100% local{R}

  {D}Dictate is now running in your menu bar.
  You can close this terminal — it won't stop the app.{R}

  {W}HOW TO USE{R}
  {D}Hold{R} {O}Left Ctrl{R}       {D}talk, release to transcribe{R}
  {D}Hold{R} {O}Ctrl + Space{R}    {D}lock recording (hands-free){R}
  {D}Tap{R}  {O}Ctrl{R}            {D}to stop locked recording{R}
  {D}Change the key, model, and more from the menu bar icon.{R}

  {W}TIPS{R}
  {D}Parakeet is English-only. Switch to Whisper for other languages
  under Advanced → STT Engine.{R}
  {D}Writing styles (Clean, Formal, Bullets) change how your text
  is polished — find them in the menu bar.{R}
  {D}Add names, slang, or technical terms to your personal dictionary
  so they're always spelled right — Advanced → Dictionary.{R}

  {W}COMMANDS{R}
  {O}dictate{R}          {D}launch dictate{R}
  {O}dictate update{R}   {D}update to the latest version{R}
  {O}dictate -f{R}       {D}run in foreground (debug){R}
""", flush=True)
        _daemonize()

    setup_logging()
    logger = logging.getLogger(__name__)

    lock_fd = _acquire_singleton_lock()
    if lock_fd is None:
        logger.error("Another instance of Dictate is already running. Exiting.")
        print("Dictate is already running.", file=sys.stderr)
        return 1

    logger.info("Starting Dictate menu bar app (pid=%d)", os.getpid())

    try:
        from dictate.menubar import DictateMenuBarApp

        app = DictateMenuBarApp()
        app.start_app()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception:
        logger.exception("Fatal error")
        return 1
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
