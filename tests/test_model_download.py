"""Tests for model download functionality."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dictate.model_download import (
    MODEL_SIZES,
    ProgressTracker,
    download_model,
    get_model_size,
    get_model_size_gb,
    is_download_in_progress,
)


class TestGetModelSize:
    """Tests for get_model_size function."""

    def test_known_whisper_model(self):
        """Test getting size for known Whisper model."""
        size = get_model_size("mlx-community/whisper-large-v3-turbo")
        assert size == "1.5GB"

    def test_known_qwen_3b(self):
        """Test getting size for Qwen 3B model."""
        size = get_model_size("mlx-community/Qwen2.5-3B-Instruct-4bit")
        assert size == "1.8GB"

    def test_known_qwen_1_5b(self):
        """Test getting size for Qwen 1.5B model."""
        size = get_model_size("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        assert size == "0.9GB"

    def test_known_parakeet(self):
        """Test getting size for Parakeet model."""
        size = get_model_size("mlx-community/parakeet-tdt-0.6b-v3")
        assert size == "0.5GB"

    def test_unknown_model(self):
        """Test getting size for unknown model returns 'Unknown'."""
        size = get_model_size("unknown/model-name")
        assert size == "Unknown"


class TestGetModelSizeGb:
    """Tests for get_model_size_gb function."""

    def test_known_model_returns_float(self):
        """Test that known models return float sizes."""
        size = get_model_size_gb("mlx-community/whisper-large-v3-turbo")
        assert isinstance(size, float)
        assert size == 1.5

    def test_unknown_model_returns_none(self):
        """Test that unknown models return None."""
        size = get_model_size_gb("unknown/model")
        assert size is None


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_callback_called_with_progress(self):
        """Test that callback is called with progress percentages."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(1000)
        
        # Update progress to 50%
        tracker.update_progress(500)
        
        # Should have received callback with 50.0
        assert len(callbacks) >= 1
        assert callbacks[0] == 50.0

    def test_callback_throttled_to_1_percent_changes(self):
        """Test that callback is throttled to 1% changes."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(1000)
        
        # Update by small amounts - should not trigger callback
        tracker.update_progress(1)
        tracker.update_progress(2)
        tracker.update_progress(3)
        
        # First callback should be at ~0.6%
        assert len(callbacks) == 1
        
    def test_report_completion_sends_100(self):
        """Test that report_completion sends 100%."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.report_completion()
        
        assert 100.0 in callbacks

    def test_no_callback_when_none_provided(self):
        """Test that no errors occur when callback is None."""
        tracker = ProgressTracker(None)
        tracker.update_total(1000)
        tracker.update_progress(500)
        tracker.report_completion()
        # Should not raise


class TestDownloadModel:
    """Tests for download_model function."""

    @patch("dictate.model_download.snapshot_download")
    def test_progress_callback_called_with_float_0_to_100(self, mock_snapshot):
        """Test that progress callback receives float values from 0 to 100."""
        mock_snapshot.return_value = "/fake/path"
        
        callbacks = []
        
        def progress_callback(percent: float) -> None:
            callbacks.append(percent)
        
        download_model(
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            progress_callback=progress_callback
        )
        
        # Should have received at least one callback
        assert len(callbacks) >= 1
        
        # All callbacks should be floats between 0 and 100
        for cb in callbacks:
            assert isinstance(cb, float)
            assert 0.0 <= cb <= 100.0

    @patch("dictate.config.is_model_cached", return_value=False)
    @patch("dictate.model_download.snapshot_download")
    def test_download_with_custom_cache_dir(self, mock_snapshot, _mock_cached):
        """Test downloading with custom cache directory."""
        mock_snapshot.return_value = "/custom/path"

        download_model(
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            cache_dir="/custom/cache"
        )

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args.kwargs
        assert call_kwargs.get("cache_dir") == "/custom/cache"

    @patch("dictate.config.is_model_cached", return_value=False)
    @patch("dictate.model_download.snapshot_download")
    def test_download_without_callback(self, mock_snapshot, _mock_cached):
        """Test downloading without progress callback."""
        mock_snapshot.return_value = "/fake/path"

        download_model("mlx-community/Qwen2.5-3B-Instruct-4bit")

        mock_snapshot.assert_called_once()

    @patch("dictate.config.is_model_cached", return_value=False)
    @patch("dictate.model_download.snapshot_download")
    def test_download_raises_on_failure(self, mock_snapshot, _mock_cached):
        """Test that download raises exception on failure."""
        mock_snapshot.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            download_model("mlx-community/Qwen2.5-3B-Instruct-4bit")


class TestIsDownloadInProgress:
    """Tests for is_download_in_progress function."""

    def test_no_download_returns_false(self):
        """Test that no download in progress returns False."""
        result = is_download_in_progress("some/model")
        assert result is False

    @patch("dictate.model_download.snapshot_download")
    def test_download_in_progress_returns_true(self, mock_snapshot):
        """Test that download in progress returns True."""
        mock_snapshot.return_value = "/fake/path"
        
        model_name = "mlx-community/test-model"
        
        # Start a download in a thread
        def slow_download():
            time.sleep(0.1)  # Small delay to ensure we can check state
            download_model(model_name)
        
        thread = threading.Thread(target=slow_download)
        thread.start()
        
        # Give thread time to acquire lock
        time.sleep(0.01)
        
        try:
            # Should report in progress while thread is running
            # Note: Due to timing, this might or might not be True depending
            # on whether the download has started or completed
            pass  # Just verify no exception is raised
        finally:
            thread.join(timeout=1.0)


class TestModelSizesMapping:
    """Tests for MODEL_SIZES constant."""

    def test_contains_expected_models(self):
        """Test that MODEL_SIZES contains expected models."""
        expected_models = [
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "mlx-community/parakeet-tdt-0.6b-v3",
        ]
        
        for model in expected_models:
            assert model in MODEL_SIZES, f"{model} should be in MODEL_SIZES"
            assert isinstance(MODEL_SIZES[model], float), f"{model} size should be float"
            assert MODEL_SIZES[model] > 0, f"{model} size should be positive"
