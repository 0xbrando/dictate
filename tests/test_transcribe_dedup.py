"""Tests for _dedup_transcription off-by-one path (transcribe.py lines 126-133).

The off-by-one loop handles odd word counts where the even-split check misses.
"""

from __future__ import annotations

from dictate.transcribe import _dedup_transcription


class TestDedupOffByOne:
    """Exercise the off-by-one repetition check loop (n >= 5 branch)."""

    def test_five_words_no_repeat_exercises_loop(self):
        """5 words, even check fails, off-by-one loop runs (no match)."""
        # 5 words: half=2, even check "hello world" vs "this is" → fail
        # Loop runs with split_at=2,3 → no match → returns original
        result = _dedup_transcription("hello world this is different")
        assert result == "hello world this is different"

    def test_seven_words_no_repeat(self):
        """7 words, even check fails, off-by-one loop runs."""
        result = _dedup_transcription("one two three four five six seven")
        assert result == "one two three four five six seven"

    def test_nine_words_no_repeat(self):
        """9 words, exercises the loop with larger input."""
        result = _dedup_transcription("a b c d e f g h i")
        assert result == "a b c d e f g h i"

    def test_five_words_even_half_matches(self):
        """5 words where even check catches the repeat."""
        # "go home go home extra" → half=2, even: "go home" vs "go home" → MATCH
        result = _dedup_transcription("go home go home extra")
        assert result == "go home"

    def test_six_words_exact_repeat(self):
        """6 words, clean even split catches it."""
        result = _dedup_transcription("the cat sat the cat sat")
        assert result == "the cat sat"

    def test_odd_count_with_partial_overlap(self):
        """Odd word count with partial overlap but not a repeat."""
        result = _dedup_transcription("hello world hello test world")
        assert result == "hello world hello test world"

    def test_exactly_five_words_all_same(self):
        """All same word, 5 count → even check catches it."""
        result = _dedup_transcription("yes yes yes yes yes")
        assert result == "yes yes"

    def test_long_odd_no_repeat(self):
        """11 words, exercises the loop with a longer odd input."""
        text = "this is a sentence that has eleven words in it now extra"
        result = _dedup_transcription(text)
        assert result == text
