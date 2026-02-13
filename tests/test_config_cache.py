"""Tests for config.py cache management functions: delete_cached_model, get_cached_model_disk_size.

Covers lines 85-99, 111-133 in config.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dictate.config import delete_cached_model, get_cached_model_disk_size


class TestDeleteCachedModel:
    """Tests for delete_cached_model()."""

    def test_delete_existing_model_real(self, tmp_path, monkeypatch):
        """Deletes model dir and returns True using real filesystem."""
        # Set up a fake HF cache structure
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--mlx-community--test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = delete_cached_model("mlx-community/test-model")
        assert result is True
        assert not model_dir.exists()

    def test_delete_nonexistent_model(self, tmp_path, monkeypatch):
        """Returns False when model directory doesn't exist."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        hub_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = delete_cached_model("mlx-community/nonexistent")
        assert result is False

    def test_delete_permission_error(self, tmp_path, monkeypatch):
        """Returns False when rmtree fails."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--mlx-community--protected"
        model_dir.mkdir(parents=True)
        (model_dir / "data.bin").write_bytes(b"\x00")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        import shutil
        original_rmtree = shutil.rmtree
        monkeypatch.setattr(shutil, "rmtree", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("denied")))

        result = delete_cached_model("mlx-community/protected")
        assert result is False
        # Model dir should still exist since rmtree was mocked to fail
        assert model_dir.exists()


class TestGetCachedModelDiskSize:
    """Tests for get_cached_model_disk_size()."""

    def test_model_not_cached(self, tmp_path, monkeypatch):
        """Returns 'Unknown' when model dir doesn't exist."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        hub_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = get_cached_model_disk_size("mlx-community/nonexistent")
        assert result == "Unknown"

    def test_small_model_bytes(self, tmp_path, monkeypatch):
        """Returns size in bytes for very small models."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--test--tiny"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_bytes(b'{"test": true}')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = get_cached_model_disk_size("test/tiny")
        assert "B" in result

    def test_model_in_kb(self, tmp_path, monkeypatch):
        """Returns size in KB."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--test--small"
        model_dir.mkdir(parents=True)
        (model_dir / "data.bin").write_bytes(b"\x00" * 5000)  # ~5KB

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = get_cached_model_disk_size("test/small")
        assert "KB" in result

    def test_model_in_mb(self, tmp_path, monkeypatch):
        """Returns size in MB."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--test--medium"
        model_dir.mkdir(parents=True)
        # ~2MB
        (model_dir / "model.bin").write_bytes(b"\x00" * (2 * 1024 * 1024))

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = get_cached_model_disk_size("test/medium")
        assert "MB" in result

    def test_model_with_subdirs(self, tmp_path, monkeypatch):
        """Calculates size across nested directories."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--test--nested"
        snapshots = model_dir / "snapshots" / "abc123"
        snapshots.mkdir(parents=True)
        (snapshots / "model.safetensors").write_bytes(b"\x00" * 1000)
        (snapshots / "config.json").write_bytes(b'{"test": true}')
        refs = model_dir / "refs"
        refs.mkdir()
        (refs / "main").write_text("abc123")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = get_cached_model_disk_size("test/nested")
        # Should account for all files
        assert result != "Unknown"
        assert "B" in result or "KB" in result

    def test_empty_model_dir(self, tmp_path, monkeypatch):
        """Empty model dir returns 0 B."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--test--empty"
        model_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = get_cached_model_disk_size("test/empty")
        assert result == "0 B"

    def test_stat_error_returns_unknown(self, tmp_path, monkeypatch):
        """Returns 'Unknown' when stat fails."""
        hub_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = hub_dir / "models--test--broken"
        model_dir.mkdir(parents=True)
        (model_dir / "data.bin").write_bytes(b"\x00" * 100)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Make rglob raise an exception
        original_rglob = Path.rglob
        def broken_rglob(self, pattern):
            raise PermissionError("no access")
        monkeypatch.setattr(Path, "rglob", broken_rglob)

        result = get_cached_model_disk_size("test/broken")
        assert result == "Unknown"


class TestWhisperModelSecurityReject:
    """Test the security rejection of non-mlx-community whisper models (line 328-329)."""

    def test_non_mlx_community_model_rejected(self, monkeypatch):
        """Non-mlx-community models are rejected with a warning."""
        from dictate.config import Config

        monkeypatch.setenv("DICTATE_WHISPER_MODEL", "evil/malicious-model")
        config = Config.from_env()
        # Should keep the default, not the evil model
        assert "mlx-community" in config.whisper.model
        assert config.whisper.model != "evil/malicious-model"

    def test_mlx_community_model_accepted(self, monkeypatch):
        """mlx-community models are accepted."""
        from dictate.config import Config

        monkeypatch.setenv("DICTATE_WHISPER_MODEL", "mlx-community/whisper-test-v2")
        config = Config.from_env()
        assert config.whisper.model == "mlx-community/whisper-test-v2"
