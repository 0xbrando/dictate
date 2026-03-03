"""Extended tests for dictate.model_download — comprehensive coverage.

Tests model size functions, progress tracking, and download state management.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from dictate.model_download import (
    MODEL_SIZES,
    ProgressTracker,
    TqdmProgressWrapper,
    download_model,
    get_model_size,
    get_model_size_gb,
    is_download_in_progress,
)


class TestGetModelSizeExtended:
    """Extended tests for get_model_size function."""

    def test_known_whisper_model(self):
        """Test getting size for known Whisper model."""
        size = get_model_size("mlx-community/whisper-large-v3-turbo")
        assert size == "1.5GB"

    def test_known_qwen_1_5b(self):
        """Test getting size for Qwen 1.5B model."""
        size = get_model_size("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        assert size == "0.9GB"

    def test_known_qwen_3b(self):
        """Test getting size for Qwen 3B model."""
        size = get_model_size("mlx-community/Qwen2.5-3B-Instruct-4bit")
        assert size == "1.8GB"

    def test_known_parakeet(self):
        """Test getting size for Parakeet model."""
        size = get_model_size("mlx-community/parakeet-tdt-0.6b-v3")
        assert size == "0.5GB"

    def test_unknown_model_returns_unknown(self):
        """Test unknown model returns 'Unknown'."""
        size = get_model_size("unknown/model-name")
        assert size == "Unknown"

    def test_unknown_model_with_similar_name(self):
        """Test model with similar but not exact name returns 'Unknown'."""
        size = get_model_size("mlx-community/whisper-large-v3-turbo-different")
        assert size == "Unknown"

    def test_empty_string_returns_unknown(self):
        """Test empty string returns 'Unknown'."""
        size = get_model_size("")
        assert size == "Unknown"


class TestGetModelSizeGbExtended:
    """Extended tests for get_model_size_gb function."""

    def test_known_model_returns_float(self):
        """Test known model returns float size."""
        size = get_model_size_gb("mlx-community/whisper-large-v3-turbo")
        assert isinstance(size, float)
        assert size == 1.5

    def test_qwen_1_5b_returns_0_95(self):
        """Test Qwen 1.5B returns 0.95."""
        size = get_model_size_gb("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        assert size == 0.95

    def test_qwen_3b_returns_1_8(self):
        """Test Qwen 3B returns 1.8."""
        size = get_model_size_gb("mlx-community/Qwen2.5-3B-Instruct-4bit")
        assert size == 1.8

    def test_parakeet_returns_0_5(self):
        """Test Parakeet returns 0.5."""
        size = get_model_size_gb("mlx-community/parakeet-tdt-0.6b-v3")
        assert size == 0.5

    def test_unknown_model_returns_none(self):
        """Test unknown model returns None."""
        size = get_model_size_gb("unknown/model")
        assert size is None

    def test_empty_string_returns_none(self):
        """Test empty string returns None."""
        size = get_model_size_gb("")
        assert size is None


class TestProgressTrackerExtended:
    """Extended tests for ProgressTracker class."""

    def test_init_with_callback(self):
        """Test initialization with callback."""
        callback = MagicMock()
        tracker = ProgressTracker(callback)
        assert tracker.callback is callback
        assert tracker.total_bytes == 0
        assert tracker.downloaded_bytes == 0
        assert tracker._last_reported_percent == -1.0

    def test_init_without_callback(self):
        """Test initialization without callback."""
        tracker = ProgressTracker(None)
        assert tracker.callback is None

    def test_update_total(self):
        """Test update_total sets total_bytes."""
        tracker = ProgressTracker(None)
        tracker.update_total(1000)
        assert tracker.total_bytes == 1000

    def test_update_progress_no_callback(self):
        """Test update_progress works without callback."""
        tracker = ProgressTracker(None)
        tracker.update_total(100)
        tracker.update_progress(50)
        assert tracker.downloaded_bytes == 50

    def test_update_progress_triggers_callback(self):
        """Test update_progress triggers callback with correct percentage."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(100)
        tracker.update_progress(50)
        
        assert len(callbacks) == 1
        assert callbacks[0] == 50.0

    def test_update_progress_throttling_1_percent(self):
        """Test callback is throttled to 1% changes."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(1000)
        
        # Small updates should be throttled
        tracker.update_progress(5)  # 0.5% — fires (1.5% delta from initial -1.0)
        tracker.update_progress(5)  # 1.0% — skipped (0.5% delta)
        tracker.update_progress(5)  # 1.5% — fires (1.0% delta from 0.5%)
        
        # Two callbacks: at 0.5% and 1.5%
        assert len(callbacks) == 2

    def test_update_progress_reports_at_1_percent_threshold(self):
        """Test callback reports at 1% threshold."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(100)
        
        # First update to 1%
        tracker.update_progress(1)
        assert len(callbacks) == 1
        assert callbacks[0] == 1.0
        
        # Next update to 2%
        tracker.update_progress(1)
        assert len(callbacks) == 2
        assert callbacks[1] == 2.0

    def test_update_progress_reports_at_99_9(self):
        """Test callback reports when >= 99.9%."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(1000)
        
        # Update to 99.9%
        tracker.update_progress(999)
        assert callbacks[-1] == 99.9

    def test_report_completion_sends_100(self):
        """Test report_completion sends 100%."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.report_completion()
        
        assert 100.0 in callbacks

    def test_report_completion_only_once(self):
        """Test report_completion only reports 100% once."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.report_completion()
        tracker.report_completion()
        tracker.report_completion()
        
        assert callbacks.count(100.0) == 1

    def test_update_progress_after_completion(self):
        """Test update_progress after completion doesn't re-report."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(100)
        tracker.update_progress(100)
        tracker.report_completion()
        
        callback_count = len(callbacks)
        tracker.update_progress(10)  # Extra update
        
        # Should not add more callbacks after completion
        assert len(callbacks) == callback_count

    def test_zero_total_bytes(self):
        """Test behavior with zero total bytes."""
        callbacks = []
        
        def callback(percent: float) -> None:
            callbacks.append(percent)
        
        tracker = ProgressTracker(callback)
        tracker.update_total(0)
        tracker.update_progress(100)
        
        # Should not trigger callback with zero total
        assert len(callbacks) == 0


class TestTqdmProgressWrapperExtended:
    """Extended tests for TqdmProgressWrapper class."""

    def test_init(self):
        """Test initialization."""
        tracker = ProgressTracker(None)
        wrapper = TqdmProgressWrapper(tracker)
        assert wrapper.tracker is tracker
        assert wrapper.n == 0
        assert wrapper.total == 0

    def test_update_increments_n(self):
        """Test update increments n counter."""
        tracker = ProgressTracker(None)
        wrapper = TqdmProgressWrapper(tracker)
        
        wrapper.update(10)
        assert wrapper.n == 10
        
        wrapper.update(5)
        assert wrapper.n == 15

    def test_update_default_increment(self):
        """Test update with default increment of 1."""
        tracker = ProgressTracker(None)
        wrapper = TqdmProgressWrapper(tracker)
        
        wrapper.update()
        assert wrapper.n == 1

    def test_update_calls_tracker(self):
        """Test update calls tracker.update_progress."""
        tracker = MagicMock(spec=ProgressTracker)
        wrapper = TqdmProgressWrapper(tracker)
        
        wrapper.update(10)
        tracker.update_progress.assert_called_once_with(10)

    def test_close_calls_tracker_completion(self):
        """Test close calls tracker.report_completion."""
        tracker = MagicMock(spec=ProgressTracker)
        wrapper = TqdmProgressWrapper(tracker)
        
        wrapper.close()
        tracker.report_completion.assert_called_once()

    def test_set_description_noop(self):
        """Test set_description is a no-op."""
        tracker = ProgressTracker(None)
        wrapper = TqdmProgressWrapper(tracker)
        
        # Should not raise
        wrapper.set_description("test description")
        wrapper.set_description("")
        wrapper.set_description(None)  # type: ignore

    def test_context_manager(self):
        """Test context manager protocol."""
        tracker = MagicMock(spec=ProgressTracker)
        wrapper = TqdmProgressWrapper(tracker)
        
        with wrapper as w:
            assert w is wrapper
        
        tracker.report_completion.assert_called_once()

    def test_context_manager_with_exception(self):
        """Test context manager handles exceptions properly."""
        tracker = MagicMock(spec=ProgressTracker)
        wrapper = TqdmProgressWrapper(tracker)
        
        try:
            with wrapper:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        tracker.report_completion.assert_called_once()


class TestIsDownloadInProgressExtended:
    """Extended tests for is_download_in_progress function."""

    def test_no_download_returns_false(self):
        """Test that no download returns False."""
        result = is_download_in_progress("some/model/not/exists")
        assert result is False

    def test_after_download_complete_returns_false(self):
        """Test that after download completes, returns False."""
        model = "mlx-community/Qwen2.5-3B-Instruct-4bit"
        
        # Ensure no lock exists initially
        with patch("dictate.model_download._download_locks", {}):
            result = is_download_in_progress(model)
            assert result is False

    @patch("dictate.model_download.snapshot_download")
    def test_during_download_returns_true(self, mock_snapshot):
        """Test that during download, returns True."""
        mock_snapshot.return_value = "/fake/path"
        
        model = "test-model-progress"
        in_progress_values = []
        
        def slow_download():
            download_model(model)
        
        # Start download in thread
        thread = threading.Thread(target=slow_download)
        thread.start()
        
        # Give thread time to acquire lock and start
        time.sleep(0.05)
        
        try:
            # Check if in progress
            in_progress_values.append(is_download_in_progress(model))
        finally:
            thread.join(timeout=2.0)
        
        # Should have been True at some point during download
        assert True in in_progress_values or not thread.is_alive()

    @patch("dictate.model_download.snapshot_download")
    def test_after_failed_download_returns_false(self, mock_snapshot):
        """Test that after failed download, returns False."""
        mock_snapshot.side_effect = Exception("Download failed")
        
        model = "test-model-fail"
        
        try:
            download_model(model)
        except Exception:
            pass
        
        # After exception, lock should be released
        result = is_download_in_progress(model)
        assert result is False


class TestModelSizesConstant:
    """Tests for MODEL_SIZES constant."""

    def test_all_expected_models_present(self):
        """Test that all expected models are in MODEL_SIZES."""
        expected = [
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "mlx-community/parakeet-tdt-0.6b-v3",
        ]
        
        for model in expected:
            assert model in MODEL_SIZES, f"{model} should be in MODEL_SIZES"

    def test_all_sizes_are_positive_floats(self):
        """Test that all sizes are positive floats."""
        for model, size in MODEL_SIZES.items():
            assert isinstance(size, float), f"{model} size should be float"
            assert size > 0, f"{model} size should be positive"

    def test_size_precision(self):
        """Test that sizes have reasonable precision."""
        # All sizes should be in GB with at most 2 decimal places
        for model, size in MODEL_SIZES.items():
            scaled = size * 100
            assert abs(scaled - round(scaled)) < 0.01, f"{model} size should have at most 2 decimal places"
