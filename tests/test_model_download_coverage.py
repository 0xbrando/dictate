"""Tests for model_download.py uncovered paths (lines 159-162, 185-200, 233).

Covers: download_model cached path, ProgressTqdm inner class, and
is_download_in_progress edge cases.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from dictate.model_download import (
    ProgressTracker,
    TqdmProgressWrapper,
    download_model,
    is_download_in_progress,
)


class TestDownloadModelAlreadyCached:
    """Test download_model when the model is already cached (lines 159-162)."""

    @patch("dictate.model_download.snapshot_download")
    def test_cached_model_skips_download(self, mock_snapshot):
        """When model is cached, download is skipped entirely."""
        with patch("dictate.config.is_model_cached", return_value=True):
            download_model("mlx-community/Qwen2.5-3B-Instruct-4bit")

        mock_snapshot.assert_not_called()

    @patch("dictate.model_download.snapshot_download")
    def test_cached_model_calls_progress_callback_100(self, mock_snapshot):
        """When model is cached, progress callback is called with 100%."""
        callback = MagicMock()

        with patch("dictate.config.is_model_cached", return_value=True):
            download_model("mlx-community/Qwen2.5-3B-Instruct-4bit", progress_callback=callback)

        callback.assert_called_once_with(100.0)
        mock_snapshot.assert_not_called()

    @patch("dictate.model_download.snapshot_download")
    def test_cached_model_no_callback_no_error(self, mock_snapshot):
        """When model is cached with no callback, no error."""
        with patch("dictate.config.is_model_cached", return_value=True):
            download_model("mlx-community/Qwen2.5-3B-Instruct-4bit", progress_callback=None)

        mock_snapshot.assert_not_called()


class TestDownloadModelProgressTqdm:
    """Test the inner ProgressTqdm class created inside download_model (lines 185-200).

    We need to invoke download_model to exercise the inner class, capturing
    the tqdm_class passed to snapshot_download.
    """

    @patch("dictate.model_download.snapshot_download")
    def test_progress_tqdm_class_passed_to_snapshot(self, mock_snapshot):
        """The ProgressTqdm class is passed as tqdm_class to snapshot_download."""
        with patch("dictate.config.is_model_cached", return_value=False):
            download_model("some/model")

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args.kwargs
        assert "tqdm_class" in call_kwargs
        # The class should be callable
        tqdm_cls = call_kwargs["tqdm_class"]
        assert callable(tqdm_cls)

    @patch("dictate.model_download.snapshot_download")
    def test_progress_tqdm_instance_methods(self, mock_snapshot):
        """Exercise ProgressTqdm instance methods (init, update, close, set_description, context manager)."""
        captured_cls = {}

        def capture_tqdm_class(**kwargs):
            captured_cls["cls"] = kwargs.get("tqdm_class")

        mock_snapshot.side_effect = capture_tqdm_class

        with patch("dictate.config.is_model_cached", return_value=False):
            download_model("some/model")

        tqdm_cls = captured_cls["cls"]
        assert tqdm_cls is not None

        # Instantiate it — exercises __init__ (line 185-186)
        instance = tqdm_cls(total=100)

        # Exercise update (line 188-189)
        instance.update(10)

        # Exercise set_description (line 194-195)
        instance.set_description("downloading...")

        # Exercise context manager __enter__/__exit__ (lines 197-200)
        with instance as ctx:
            assert ctx is instance
            ctx.update(5)

        # Exercise close (line 191-192)
        instance.close()

    @patch("dictate.model_download.snapshot_download")
    def test_progress_tqdm_reports_progress_via_callback(self, mock_snapshot):
        """ProgressTqdm reports progress through the callback chain."""
        progress_values = []

        def callback(pct):
            progress_values.append(pct)

        def simulate_download(**kwargs):
            tqdm_cls = kwargs.get("tqdm_class")
            # Simulate HuggingFace's tqdm usage pattern
            bar = tqdm_cls(total=1000)
            bar.update(500)
            bar.update(500)
            bar.close()

        mock_snapshot.side_effect = simulate_download

        with patch("dictate.config.is_model_cached", return_value=False):
            download_model(
                "mlx-community/Qwen2.5-3B-Instruct-4bit",
                progress_callback=callback,
            )

        # Should have received some progress + 100% completion
        assert len(progress_values) > 0
        assert progress_values[-1] == 100.0

    @patch("dictate.model_download.snapshot_download")
    def test_download_with_unknown_model_size(self, mock_snapshot):
        """Download works even when model size is unknown (no estimated_bytes)."""
        callback = MagicMock()

        with patch("dictate.config.is_model_cached", return_value=False):
            download_model("unknown/model-xyz", progress_callback=callback)

        mock_snapshot.assert_called_once()

    @patch("dictate.model_download.snapshot_download")
    def test_download_with_custom_cache_dir(self, mock_snapshot):
        """Download passes cache_dir to snapshot_download."""
        with patch("dictate.config.is_model_cached", return_value=False):
            download_model("some/model", cache_dir="/tmp/test-cache")

        call_kwargs = mock_snapshot.call_args.kwargs
        assert call_kwargs.get("cache_dir") == "/tmp/test-cache"

    @patch("dictate.model_download.snapshot_download")
    def test_download_raises_on_snapshot_failure(self, mock_snapshot):
        """Download raises if snapshot_download fails."""
        mock_snapshot.side_effect = OSError("Network error")

        with patch("dictate.config.is_model_cached", return_value=False):
            with pytest.raises(OSError, match="Network error"):
                download_model("some/model")


class TestIsDownloadInProgressLockHeld:
    """Test is_download_in_progress when lock exists but is free (line 233)."""

    def test_lock_exists_but_free(self):
        """When lock exists but is not held, returns False."""
        model = "test/lock-free-model"
        lock = threading.Lock()

        with patch("dictate.model_download._download_locks", {model: lock}):
            result = is_download_in_progress(model)
            assert result is False

    def test_lock_exists_and_held(self):
        """When lock exists and is held, returns True."""
        model = "test/lock-held-model"
        lock = threading.Lock()
        lock.acquire()

        try:
            with patch("dictate.model_download._download_locks", {model: lock}):
                result = is_download_in_progress(model)
                assert result is True
        finally:
            lock.release()
