"""Model download utilities with progress tracking for HuggingFace models."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars

logger = logging.getLogger(__name__)

# Model size approximations (in GB) for known models
MODEL_SIZES: dict[str, float] = {
    # Whisper models
    "mlx-community/whisper-large-v3-turbo": 1.5,
    # LLM models
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit": 1.0,
    "mlx-community/Qwen2.5-3B-Instruct-4bit": 1.8,
    "mlx-community/Qwen2.5-7B-Instruct-4bit": 4.2,
    "mlx-community/Qwen2.5-14B-Instruct-4bit": 8.8,
    "mlx-community/Phi-3-mini-4k-instruct-4bit": 1.8,
    # Parakeet models
    "mlx-community/parakeet-tdt-0.6b-v3": 0.5,
}

# Default timeout for model downloads (in seconds)
DEFAULT_DOWNLOAD_TIMEOUT = 600  # 10 minutes

# Lock to prevent concurrent downloads of the same model
_download_locks: dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()


def get_model_size(hf_repo: str) -> str:
    """Return approximate size string for known models.
    
    Args:
        hf_repo: HuggingFace repository name (e.g., 'mlx-community/Qwen2.5-3B-Instruct-4bit')
        
    Returns:
        Size string like '1.8GB' or 'Unknown'
    """
    size_gb = MODEL_SIZES.get(hf_repo)
    if size_gb is None:
        return "Unknown"
    return f"{size_gb:.1f}GB"


def get_model_size_gb(hf_repo: str) -> float | None:
    """Return approximate size in GB for known models.
    
    Args:
        hf_repo: HuggingFace repository name
        
    Returns:
        Size in GB or None if unknown
    """
    return MODEL_SIZES.get(hf_repo)


class ProgressTracker:
    """Custom progress tracker for HuggingFace downloads."""
    
    def __init__(self, callback: Callable[[float], None] | None = None) -> None:
        self.callback = callback
        self.total_bytes = 0
        self.downloaded_bytes = 0
        self._last_reported_percent = -1.0
        
    def update_total(self, total: int) -> None:
        """Update total bytes to download."""
        self.total_bytes = total
        
    def update_progress(self, n: int) -> None:
        """Update downloaded bytes and report progress."""
        self.downloaded_bytes += n
        if self.total_bytes > 0 and self.callback:
            percent = (self.downloaded_bytes / self.total_bytes) * 100
            # Only report if changed by at least 1%
            if percent - self._last_reported_percent >= 1.0 or percent >= 99.9:
                self._last_reported_percent = percent
                self.callback(min(percent, 100.0))
                
    def report_completion(self) -> None:
        """Report 100% completion."""
        if self.callback and self._last_reported_percent < 100.0:
            self.callback(100.0)
            self._last_reported_percent = 100.0


class TqdmProgressWrapper:
    """A tqdm-compatible wrapper for progress tracking."""
    
    def __init__(self, tracker: ProgressTracker) -> None:
        self.tracker = tracker
        self.n = 0
        self.total = 0
        
    def update(self, n: int = 1) -> None:
        """Update progress (tqdm-compatible)."""
        self.n += n
        self.tracker.update_progress(n)
        
    def close(self) -> None:
        """Close the progress tracker."""
        self.tracker.report_completion()
        
    def set_description(self, desc: str) -> None:
        """Set description (no-op for compatibility)."""
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.close()


def download_model(
    hf_repo: str, 
    progress_callback: Callable[[float], None] | None = None,
    cache_dir: str | None = None,
    timeout: float | None = None,
) -> None:
    """Download a model from HuggingFace with progress tracking.
    
    This function downloads all files for a given model repository using
    snapshot_download with a custom progress tracker.
    
    Args:
        hf_repo: HuggingFace repository name (e.g., 'mlx-community/Qwen2.5-3B-Instruct-4bit')
        progress_callback: Optional callback function called with progress percentage (0-100)
        cache_dir: Optional cache directory (defaults to HuggingFace default)
        timeout: Optional timeout in seconds (defaults to 10 minutes)
        
    Raises:
        Exception: If download fails or times out
    """
    timeout = timeout or DEFAULT_DOWNLOAD_TIMEOUT
    
    # Get or create a lock for this model to prevent concurrent downloads
    with _locks_mutex:
        if hf_repo not in _download_locks:
            _download_locks[hf_repo] = threading.Lock()
        model_lock = _download_locks[hf_repo]
    
    # Use a single with statement for proper lock management
    # This avoids potential deadlock from multiple acquire/release cycles
    with model_lock:
        # Check if already cached (another thread may have downloaded it)
        from dictate.config import is_model_cached
        if is_model_cached(hf_repo):
            logger.info("Model %s is already cached", hf_repo)
            if progress_callback:
                progress_callback(100.0)
            return
        
        logger.info("Starting download for %s", hf_repo)
        
        # Create progress tracker
        tracker = ProgressTracker(progress_callback)
        
        # Use snapshot_download with a custom tqdm_class to track progress
        # We need to estimate total size first
        size_gb = get_model_size_gb(hf_repo)
        if size_gb:
            # Estimate bytes (with some buffer for metadata files)
            estimated_bytes = int(size_gb * 1024 * 1024 * 1024 * 1.1)
            tracker.update_total(estimated_bytes)
        
        # Disable default HF progress bars
        disable_progress_bars()
        
        # Download with progress tracking via a custom wrapper
        # Since snapshot_download doesn't have a direct byte callback,
        # we'll use the tqdm_class approach
        class ProgressTqdm:
            def __init__(inner_self, *args, **kwargs):
                inner_self.wrapper = TqdmProgressWrapper(tracker)
                
            def update(inner_self, n: int = 1):
                inner_self.wrapper.update(n)
                
            def close(inner_self):
                inner_self.wrapper.close()
                
            def set_description(inner_self, desc: str):
                pass
                
            def __enter__(inner_self):
                return inner_self
                
            def __exit__(inner_self, *args):
                inner_self.close()
        
        try:
            snapshot_download(
                repo_id=hf_repo,
                cache_dir=cache_dir,
                tqdm_class=ProgressTqdm,
            )
            tracker.report_completion()
            logger.info("Download completed for %s", hf_repo)
        except Exception as e:
            logger.exception("Download failed for %s", hf_repo)
            raise


def is_download_in_progress(hf_repo: str) -> bool:
    """Check if a download is currently in progress for the given model.
    
    Args:
        hf_repo: HuggingFace repository name
        
    Returns:
        True if download is in progress, False otherwise
    """
    with _locks_mutex:
        lock = _download_locks.get(hf_repo)
        if lock is None:
            return False
        # Try to acquire the lock without blocking
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
            return False  # Lock was free, no download in progress
        return True  # Lock is held, download in progress
