"""Tests for dictate.audio — VAD state, tone synthesis, and audio utilities."""

from __future__ import annotations

import numpy as np
import pytest

from dictate.audio import (
    FADE_DURATION_SECONDS,
    MIN_CHUNK_DURATION_SECONDS,
    VADState,
    _synth_chime,
    _synth_click,
    _synth_marimba,
    _synth_simple,
    _synth_soft_pop,
    _synth_warm,
)


# ── VADState ──────────────────────────────────────────────────


class TestVADState:
    def test_initial_state(self):
        vad = VADState()
        assert vad.in_speech is False
        assert vad.last_speech_time == 0.0
        assert len(vad.current_chunk) == 0

    def test_reset(self):
        vad = VADState(in_speech=True, last_speech_time=123.0)
        vad.current_chunk.append(np.zeros(100, dtype=np.float32))
        vad.reset(pre_roll_samples=4000)
        assert vad.in_speech is False
        assert vad.last_speech_time == 0.0
        assert len(vad.current_chunk) == 0
        assert vad.pre_roll.maxlen == 4000


# ── Tone synthesizers ─────────────────────────────────────────


class TestToneSynth:
    """Test that all tone synthesizers produce valid audio arrays."""

    @pytest.fixture
    def sr(self):
        return 44100

    def test_simple_tone(self, sr):
        tone = _synth_simple(880, 0.04, 0.15, sr)
        assert isinstance(tone, np.ndarray)
        assert len(tone) == int(sr * 0.04)
        assert tone.max() <= 0.15 + 0.01  # small float tolerance

    def test_soft_pop(self, sr):
        tone = _synth_soft_pop(880, 0.15, sr)
        assert isinstance(tone, np.ndarray)
        assert len(tone) > 0
        # Exponential decay means end should be near zero
        assert abs(tone[-1]) < 0.01

    def test_chime(self, sr):
        tone = _synth_chime(880, 0.15, sr)
        assert isinstance(tone, np.ndarray)
        assert len(tone) > 0

    def test_warm(self, sr):
        tone = _synth_warm(880, 0.15, sr)
        assert isinstance(tone, np.ndarray)
        assert len(tone) > 0

    def test_click(self, sr):
        tone = _synth_click(1000, 0.15, sr)
        assert isinstance(tone, np.ndarray)
        assert len(tone) > 0
        # Click is very short
        assert len(tone) == int(sr * 0.015)

    def test_marimba(self, sr):
        tone = _synth_marimba(880, 0.15, sr)
        assert isinstance(tone, np.ndarray)
        assert len(tone) > 0

    def test_zero_volume(self, sr):
        tone = _synth_simple(880, 0.04, 0.0, sr)
        assert np.allclose(tone, 0.0)

    def test_all_synths_return_float(self, sr):
        """All tone synthesizers should return float arrays."""
        for synth in [_synth_simple, _synth_soft_pop, _synth_chime,
                      _synth_warm, _synth_click, _synth_marimba]:
            if synth == _synth_simple:
                tone = synth(880, 0.04, 0.15, sr)
            else:
                tone = synth(880, 0.15, sr)
            assert tone.dtype in (np.float32, np.float64), \
                f"{synth.__name__} returned {tone.dtype}"
