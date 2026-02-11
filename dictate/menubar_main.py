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

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

LOCK_FILE = Path.home() / "Library" / "Application Support" / "Dictate" / "dictate.lock"
LOG_FILE = Path.home() / "Library" / "Logs" / "Dictate" / "dictate.log"


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
        subprocess.run(["pkill", "-f", "dictate"], capture_output=True, check=False)
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
    # Handle update command before anything else
    if "update" in sys.argv or "--update" in sys.argv:
        return _run_update()
    
    # Daemonize before anything else — detach from terminal
    foreground = "--foreground" in sys.argv or "-f" in sys.argv
    if not foreground and sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
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
