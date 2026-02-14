"""Usage statistics tracking for Dictate.

Tracks dictation sessions, word counts, and audio duration.
All stats are stored locally in ~/Library/Application Support/Dictate/stats.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STATS_DIR = Path.home() / "Library" / "Application Support" / "Dictate"
STATS_FILE = STATS_DIR / "stats.json"


@dataclass
class UsageStats:
    """Cumulative usage statistics."""

    total_dictations: int = 0
    total_words: int = 0
    total_characters: int = 0
    total_audio_seconds: float = 0.0
    first_use: float = 0.0  # Unix timestamp
    last_use: float = 0.0   # Unix timestamp
    styles_used: dict[str, int] = field(default_factory=dict)  # style → count

    def record_dictation(
        self,
        text: str,
        audio_seconds: float = 0.0,
        style: str = "clean",
    ) -> None:
        """Record a completed dictation."""
        now = time.time()
        self.total_dictations += 1
        self.total_words += len(text.split())
        self.total_characters += len(text)
        self.total_audio_seconds += audio_seconds
        if self.first_use == 0.0:
            self.first_use = now
        self.last_use = now
        self.styles_used[style] = self.styles_used.get(style, 0) + 1

    def save(self) -> None:
        """Persist stats to disk."""
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            STATS_FILE.write_text(json.dumps(asdict(self), indent=2))
            os.chmod(STATS_FILE, 0o600)
        except OSError:
            logger.exception("Failed to save stats")

    @classmethod
    def load(cls) -> UsageStats:
        """Load stats from disk, or return fresh stats."""
        if not STATS_FILE.exists():
            return cls()
        try:
            data = json.loads(STATS_FILE.read_text())
            return cls(
                total_dictations=data.get("total_dictations", 0),
                total_words=data.get("total_words", 0),
                total_characters=data.get("total_characters", 0),
                total_audio_seconds=data.get("total_audio_seconds", 0.0),
                first_use=data.get("first_use", 0.0),
                last_use=data.get("last_use", 0.0),
                styles_used=data.get("styles_used", {}),
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load stats, starting fresh")
            return cls()

    @staticmethod
    def reset() -> None:
        """Delete stats file to reset all stats."""
        if STATS_FILE.exists():
            try:
                STATS_FILE.unlink()
            except OSError:
                logger.warning("Failed to delete stats file")

    def format_duration(self, seconds: float) -> str:
        """Format seconds into a human-readable duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}m"
        hours = seconds / 3600
        return f"{hours:.1f}h"

    def format_time_ago(self, timestamp: float) -> str:
        """Format a timestamp as a human-readable 'time ago' string."""
        if timestamp == 0.0:
            return "never"
        diff = time.time() - timestamp
        if diff < 60:
            return "just now"
        if diff < 3600:
            mins = int(diff / 60)
            return f"{mins}m ago"
        if diff < 86400:
            hours = int(diff / 3600)
            return f"{hours}h ago"
        days = int(diff / 86400)
        return f"{days}d ago"
