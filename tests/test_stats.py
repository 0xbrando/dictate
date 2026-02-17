"""Tests for the usage statistics module and CLI command."""

import json
import time
from pathlib import Path
from unittest import mock

import pytest


# ── Helpers ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_stats(tmp_path, monkeypatch):
    """Redirect stats to a temp directory."""
    stats_dir = tmp_path / "Dictate"
    stats_dir.mkdir()
    stats_file = stats_dir / "stats.json"
    monkeypatch.setattr("dictate.stats.STATS_DIR", stats_dir)
    monkeypatch.setattr("dictate.stats.STATS_FILE", stats_file)
    # Also isolate prefs for CLI tests
    prefs_dir = tmp_path / "DictatePrefs"
    prefs_dir.mkdir()
    prefs_file = prefs_dir / "preferences.json"
    dict_file = prefs_dir / "dictionary.json"
    monkeypatch.setattr("dictate.presets.PREFS_DIR", prefs_dir)
    monkeypatch.setattr("dictate.presets.PREFS_FILE", prefs_file)
    monkeypatch.setattr("dictate.presets.DICTIONARY_FILE", dict_file)
    return stats_dir, stats_file


# ── UsageStats class ────────────────────────────────────────────


class TestUsageStats:
    def test_fresh_stats(self):
        from dictate.stats import UsageStats
        stats = UsageStats()
        assert stats.total_dictations == 0
        assert stats.total_words == 0
        assert stats.total_characters == 0
        assert stats.total_audio_seconds == 0.0

    def test_record_dictation(self):
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("Hello world test", audio_seconds=3.5, style="clean")
        assert stats.total_dictations == 1
        assert stats.total_words == 3
        assert stats.total_characters == 16
        assert stats.total_audio_seconds == 3.5
        assert stats.styles_used == {"clean": 1}
        assert stats.first_use > 0
        assert stats.last_use > 0

    def test_record_multiple_dictations(self):
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("Hello", audio_seconds=1.0, style="clean")
        stats.record_dictation("Goodbye cruel world", audio_seconds=2.5, style="formal")
        stats.record_dictation("Test", audio_seconds=0.5, style="clean")
        assert stats.total_dictations == 3
        assert stats.total_words == 5  # 1 + 3 + 1
        assert stats.total_audio_seconds == 4.0
        assert stats.styles_used == {"clean": 2, "formal": 1}

    def test_first_use_only_set_once(self):
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("first")
        first = stats.first_use
        time.sleep(0.01)
        stats.record_dictation("second")
        assert stats.first_use == first  # unchanged
        assert stats.last_use > first  # updated

    def test_save_and_load(self, _isolate_stats):
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("Hello world", audio_seconds=2.0, style="clean")
        stats.record_dictation("Test message here", audio_seconds=3.0, style="formal")
        stats.save()

        loaded = UsageStats.load()
        assert loaded.total_dictations == 2
        assert loaded.total_words == 5
        assert loaded.total_characters == 28  # 11 + 17
        assert loaded.total_audio_seconds == 5.0
        assert loaded.styles_used == {"clean": 1, "formal": 1}

    def test_load_no_file(self):
        from dictate.stats import UsageStats
        stats = UsageStats.load()
        assert stats.total_dictations == 0

    def test_load_corrupt_file(self, _isolate_stats):
        stats_dir, stats_file = _isolate_stats
        stats_file.write_text("not json{{{")
        from dictate.stats import UsageStats
        stats = UsageStats.load()
        assert stats.total_dictations == 0  # fresh stats

    def test_load_partial_data(self, _isolate_stats):
        stats_dir, stats_file = _isolate_stats
        stats_file.write_text(json.dumps({"total_dictations": 42}))
        from dictate.stats import UsageStats
        stats = UsageStats.load()
        assert stats.total_dictations == 42
        assert stats.total_words == 0  # default

    def test_reset(self, _isolate_stats):
        stats_dir, stats_file = _isolate_stats
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("test")
        stats.save()
        assert stats_file.exists()
        UsageStats.reset()
        assert not stats_file.exists()

    def test_reset_no_file(self):
        from dictate.stats import UsageStats
        UsageStats.reset()  # should not raise


# ── Format helpers ──────────────────────────────────────────────


class TestFormatHelpers:
    def test_format_duration_seconds(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        assert s.format_duration(30) == "30s"
        assert s.format_duration(0) == "0s"

    def test_format_duration_minutes(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        assert s.format_duration(90) == "1.5m"
        assert s.format_duration(300) == "5.0m"

    def test_format_duration_hours(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        assert s.format_duration(3600) == "1.0h"
        assert s.format_duration(7200) == "2.0h"

    def test_format_time_ago_never(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        assert s.format_time_ago(0.0) == "never"

    def test_format_time_ago_just_now(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        assert s.format_time_ago(time.time() - 10) == "just now"

    def test_format_time_ago_minutes(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        result = s.format_time_ago(time.time() - 300)  # 5 min ago
        assert "m ago" in result

    def test_format_time_ago_hours(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        result = s.format_time_ago(time.time() - 7200)  # 2h ago
        assert "h ago" in result

    def test_format_time_ago_days(self):
        from dictate.stats import UsageStats
        s = UsageStats()
        result = s.format_time_ago(time.time() - 172800)  # 2d ago
        assert "d ago" in result


# ── CLI command ─────────────────────────────────────────────────


class TestStatsCLI:
    def test_stats_no_data(self, capsys):
        from dictate.menubar_main import _show_stats
        rc = _show_stats()
        assert rc == 0
        out = capsys.readouterr().out
        assert "No dictations" in out

    def test_stats_with_data(self, capsys):
        from dictate.stats import UsageStats
        from dictate.menubar_main import _show_stats
        stats = UsageStats()
        stats.record_dictation("Hello world test message", audio_seconds=5.0, style="clean")
        stats.record_dictation("Another test", audio_seconds=3.0, style="formal")
        stats.save()

        rc = _show_stats()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Dictate Stats" in out
        assert "2" in out  # 2 dictations
        assert "Words" in out
        assert "clean" in out
        assert "formal" in out

    def test_stats_shows_avg_words(self, capsys):
        from dictate.stats import UsageStats
        from dictate.menubar_main import _show_stats
        stats = UsageStats()
        stats.record_dictation("one two three four", audio_seconds=2.0)
        stats.record_dictation("five six", audio_seconds=1.0)
        stats.save()

        rc = _show_stats()
        assert rc == 0
        out = capsys.readouterr().out
        assert "3.0" in out  # avg words (4+2)/2 = 3.0

    def test_stats_in_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["dictate", "--help"])
        from dictate.menubar_main import main
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "stats" in out

    def test_main_routes_stats(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["dictate", "stats"])
        from dictate.menubar_main import main
        rc = main()
        assert rc == 0

    def test_stats_style_bars(self, capsys):
        """Style breakdown shows bar visualization."""
        from dictate.stats import UsageStats
        from dictate.menubar_main import _show_stats
        stats = UsageStats()
        for _ in range(10):
            stats.record_dictation("word", style="clean")
        for _ in range(5):
            stats.record_dictation("word", style="formal")
        stats.save()

        rc = _show_stats()
        assert rc == 0
        out = capsys.readouterr().out
        assert "█" in out  # bar chars present
        assert "clean" in out
        assert "formal" in out


# ── Save/load file permissions ──────────────────────────────────


class TestStatsFilePermissions:
    def test_save_creates_file(self, _isolate_stats):
        stats_dir, stats_file = _isolate_stats
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("test")
        stats.save()
        assert stats_file.exists()

    def test_save_file_is_valid_json(self, _isolate_stats):
        stats_dir, stats_file = _isolate_stats
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("hello world")
        stats.save()
        data = json.loads(stats_file.read_text())
        assert data["total_dictations"] == 1
        assert data["total_words"] == 2

    def test_save_file_permissions(self, _isolate_stats):
        import stat
        stats_dir, stats_file = _isolate_stats
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("test")
        stats.save()
        mode = stats_file.stat().st_mode & 0o777
        assert mode == 0o600  # owner read/write only

    def test_save_handles_os_error(self, _isolate_stats, monkeypatch):
        """save() should not raise even if write fails."""
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("test")
        # Make STATS_FILE point to a non-writable location
        monkeypatch.setattr("dictate.stats.STATS_FILE", Path("/nonexistent/dir/stats.json"))
        stats.save()  # Should not raise

    def test_reset_handles_os_error(self, _isolate_stats, monkeypatch):
        """reset() should not raise even if unlink fails."""
        from dictate.stats import UsageStats
        stats = UsageStats()
        stats.record_dictation("test")
        stats.save()
        # Make unlink fail by pointing to a non-existent file
        monkeypatch.setattr("dictate.stats.STATS_FILE", Path("/nonexistent/dir/stats.json"))
        UsageStats.reset()  # Should not raise
