#!/usr/bin/env python3
"""Benchmark STT and LLM cleanup performance across models.

Usage:
    python benchmark.py                  # benchmark all available engines
    python benchmark.py --stt-only       # only benchmark STT
    python benchmark.py --llm-only       # only benchmark LLM cleanup
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Sample text for LLM cleanup testing
SAMPLE_TEXTS = [
    "hello world",
    "um so basically I was thinking about going to the store",
    "The quick brown fox jumps over the lazy dog.",
    "i need to pick up some groceries and then maybe go to the gym after that and then come back home and cook dinner for the family",
    "can you fix the authentication bug in the login flow its been broken since last tuesday when we deployed the new version",
]


def generate_test_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a test audio signal (sine wave at 440Hz, simulating speech-level audio)."""
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    # Mix of frequencies to simulate voice-like audio
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return (audio * 32767).astype(np.int16)


def benchmark_whisper(audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Benchmark Whisper large-v3-turbo."""
    from dictate.config import WhisperConfig
    from dictate.transcribe import WhisperTranscriber

    config = WhisperConfig(model="mlx-community/whisper-large-v3-turbo")
    transcriber = WhisperTranscriber(config)

    print("\n--- Whisper large-v3-turbo ---")
    print("Loading model...", end=" ", flush=True)
    t0 = time.time()
    transcriber.load_model()
    print(f"({time.time() - t0:.1f}s)")

    # Warm up
    transcriber.transcribe(audio, sample_rate)

    # Benchmark 5 runs
    times = []
    for i in range(5):
        t0 = time.time()
        text = transcriber.transcribe(audio, sample_rate)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s  ->  {text[:60]}..." if len(text) > 60 else f"  Run {i+1}: {elapsed:.3f}s  ->  {text}")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.3f}s  (min={min(times):.3f}s, max={max(times):.3f}s)")


def benchmark_parakeet(audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Benchmark Parakeet TDT 0.6B v3."""
    try:
        from parakeet_mlx import from_pretrained
    except ImportError:
        print("\n--- Parakeet TDT 0.6B v3 ---")
        print("  SKIPPED: parakeet-mlx not installed (pip install parakeet-mlx)")
        return

    from dictate.config import WhisperConfig, STTEngine
    from dictate.transcribe import ParakeetTranscriber

    config = WhisperConfig(
        model="mlx-community/parakeet-tdt-0.6b-v3",
        engine=STTEngine.PARAKEET,
    )
    transcriber = ParakeetTranscriber(config)

    print("\n--- Parakeet TDT 0.6B v3 ---")
    print("Loading model...", end=" ", flush=True)
    t0 = time.time()
    transcriber.load_model()
    print(f"({time.time() - t0:.1f}s)")

    # Warm up
    transcriber.transcribe(audio, sample_rate)

    # Benchmark 5 runs
    times = []
    for i in range(5):
        t0 = time.time()
        text = transcriber.transcribe(audio, sample_rate)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s  ->  {text[:60]}..." if len(text) > 60 else f"  Run {i+1}: {elapsed:.3f}s  ->  {text}")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.3f}s  (min={min(times):.3f}s, max={max(times):.3f}s)")


def benchmark_llm_cleanup() -> None:
    """Benchmark LLM cleanup across model sizes."""
    from dictate.config import LLMConfig, LLMModel
    from dictate.transcribe import TextCleaner

    models = [
        ("Qwen3 0.6B", LLMModel.QWEN3_0_6B),
        ("Qwen3 1.7B", LLMModel.QWEN3_1_7B),
        ("Qwen2.5 3B", LLMModel.QWEN_3B),
    ]

    for label, model_choice in models:
        config = LLMConfig(model_choice=model_choice, enabled=True)
        cleaner = TextCleaner(config)

        print(f"\n--- LLM: {label} ({config.model}) ---")
        print("Loading model...", end=" ", flush=True)
        t0 = time.time()
        cleaner.load_model()
        print(f"({time.time() - t0:.1f}s)")

        for text in SAMPLE_TEXTS:
            t0 = time.time()
            result = cleaner.cleanup(text)
            elapsed = time.time() - t0
            words = len(text.split())
            changed = "changed" if result != text else "unchanged"
            print(f"  {words:2d} words -> {elapsed*1000:.0f}ms ({changed}): {result[:50]}")


def benchmark_smart_skip() -> None:
    """Show which texts would be skipped by the smart skip heuristic."""
    from dictate.transcribe import _looks_clean

    print("\n--- Smart Skip Heuristic ---")
    test_cases = [
        "Hello world.",
        "Hello",
        "um so basically",
        "The quick brown fox jumps over the lazy dog.",
        "i need to go to the store",
        "Can you fix the bug?",
        "OK",
        "this is a test without punctuation and it goes on and on",
        "I'll be there in five minutes.",
    ]
    for text in test_cases:
        skip = _looks_clean(text)
        words = len(text.split())
        print(f"  {'SKIP' if skip else 'LLM ':4s} ({words:2d}w): {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Dictate performance")
    parser.add_argument("--stt-only", action="store_true", help="Only benchmark STT engines")
    parser.add_argument("--llm-only", action="store_true", help="Only benchmark LLM cleanup")
    parser.add_argument("--skip-heuristic", action="store_true", help="Show smart skip heuristic results")
    args = parser.parse_args()

    print("=" * 60)
    print("Dictate Performance Benchmark")
    print("=" * 60)

    # Get chip info
    import platform
    print(f"Platform: {platform.platform()}")
    try:
        import subprocess
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=2,
        ).strip()
        print(f"Chip: {chip}")
    except Exception:
        print(f"Arch: {platform.machine()}")

    if args.skip_heuristic:
        benchmark_smart_skip()
        return

    audio = generate_test_audio(duration_s=3.0)
    print(f"Test audio: {len(audio)/16000:.1f}s at 16kHz")

    if not args.llm_only:
        benchmark_whisper(audio)
        benchmark_parakeet(audio)

    if not args.stt_only:
        benchmark_llm_cleanup()

    benchmark_smart_skip()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
