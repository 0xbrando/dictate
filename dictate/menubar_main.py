"""Entry point for the menu bar app: python -m dictate.menubar_main"""

import fcntl
import logging
import os
import sys
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


def _acquire_singleton_lock() -> int | None:
    """Acquire an exclusive lock to prevent duplicate instances.

    Returns the lock fd on success, or None if another instance is running.
    The fd must be kept open for the lifetime of the process.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.ftruncate(fd, os.lseek(fd, 0, os.SEEK_CUR))
        return fd
    except OSError:
        os.close(fd)
        return None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("urllib3", "httpx", "mlx", "transformers", "tokenizers", "sounddevice"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)

    lock_fd = _acquire_singleton_lock()
    if lock_fd is None:
        logger.error("Another instance of Dictate is already running. Exiting.")
        print("Dictate is already running.", file=sys.stderr)
        return 1

    logger.info("Starting Dictate menu bar app")

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
