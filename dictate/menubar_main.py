"""Entry point for the menu bar app: python -m dictate.menubar_main"""

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


if __name__ == "__main__":
    sys.exit(main())
