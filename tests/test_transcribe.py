"""Tests for dictate.transcribe — postprocessing, smart skip, and dedup logic."""

from __future__ import annotations

import pytest

from dictate.transcribe import _dedup_transcription, _looks_clean, _postprocess


# ── _postprocess ──────────────────────────────────────────────


class TestPostprocess:
    def test_strips_special_tokens(self):
        assert _postprocess("Hello world<|end|>") == "Hello world"
        assert _postprocess("Hello<|endoftext|>") == "Hello"
        assert _postprocess("Test<|im_end|>") == "Test"
        assert _postprocess("Done<|eot_id|>") == "Done"
        assert _postprocess("Fin</s>") == "Fin"

    def test_strips_multiple_tokens(self):
        assert _postprocess("Hello<|end|><|endoftext|>") == "Hello"

    def test_strips_preambles(self):
        assert _postprocess("Sure, here's the corrected text: Hello world.") == "Hello world."
        assert _postprocess("Here's the corrected text: Test.") == "Test."
        assert _postprocess("Corrected: Fixed text.") == "Fixed text."
        assert _postprocess("Output: Result here.") == "Result here."
        assert _postprocess("Certainly! Done.") == "Done."

    def test_strips_preambles_case_insensitive(self):
        assert _postprocess("SURE, HERE'S THE CORRECTED TEXT: Hello.") == "Hello."

    def test_strips_surrounding_quotes(self):
        assert _postprocess('"Hello world."') == "Hello world."
        assert _postprocess("'Hello world.'") == "Hello world."

    def test_does_not_strip_mismatched_quotes(self):
        assert _postprocess('"Hello world.') == '"Hello world.'
        assert _postprocess("Hello world.\"") == 'Hello world."'

    def test_strips_leading_newlines(self):
        assert _postprocess("\n\nHello world.") == "Hello world."

    def test_detects_repeated_first_line(self):
        result = _postprocess("Hello world.\nHello world.\nExtra line.")
        assert result == "Hello world."

    def test_no_false_positive_repetition(self):
        result = _postprocess("Hello world.\nDifferent line.\nThird line.")
        assert "Different line." in result

    def test_empty_input(self):
        assert _postprocess("") == ""
        assert _postprocess("   ") == ""

    def test_only_special_tokens(self):
        assert _postprocess("<|end|>") == ""

    def test_preamble_then_quotes(self):
        """Preamble stripping should happen before quote stripping."""
        result = _postprocess('Sure, here\'s the corrected text: "Hello world."')
        assert result == "Hello world."

    def test_preserves_interior_newlines(self):
        result = _postprocess("Line one.\nLine two.\nLine three.")
        assert "Line one." in result
        assert "Line two." in result


# ── _looks_clean ──────────────────────────────────────────────


class TestLooksClean:
    def test_short_capitalized_with_period(self):
        assert _looks_clean("Hello world.") is True

    def test_short_capitalized_with_question(self):
        assert _looks_clean("What time?") is True

    def test_short_without_punctuation_under_4_words(self):
        # Under 4 words, no punctuation required
        assert _looks_clean("Hello") is True
        assert _looks_clean("Do it") is True

    def test_4_plus_words_need_punctuation(self):
        assert _looks_clean("Hello world how are") is False
        assert _looks_clean("Hello world how are.") is True

    def test_lowercase_start_fails(self):
        assert _looks_clean("hello world.") is False

    def test_number_start_passes(self):
        assert _looks_clean("3 items left.") is True

    def test_quote_start_passes(self):
        assert _looks_clean('"Hello world."') is True

    def test_filler_words_fail(self):
        assert _looks_clean("Um hello there.") is False
        assert _looks_clean("Uh what.") is False
        assert _looks_clean("Like totally.") is False
        assert _looks_clean("You know what.") is False
        assert _looks_clean("Basically done.") is False

    def test_too_long_fails(self):
        assert _looks_clean("This is a sentence that has way more than eight words in it.") is False

    def test_empty_fails(self):
        assert _looks_clean("") is False

    def test_exactly_8_words(self):
        assert _looks_clean("One two three four five six seven eight.") is True

    def test_9_words_fails(self):
        assert _looks_clean("One two three four five six seven eight nine.") is False


# ── _dedup_transcription ──────────────────────────────────────


class TestDedupTranscription:
    def test_exact_duplicate(self):
        assert _dedup_transcription("hello world hello world") == "hello world"

    def test_no_duplicate(self):
        text = "hello world foo bar"
        assert _dedup_transcription(text) == text

    def test_short_text_untouched(self):
        assert _dedup_transcription("hi") == "hi"
        assert _dedup_transcription("hi there") == "hi there"

    def test_odd_word_count_duplicate(self):
        # "hello world test hello world test" has 6 words, split at 3
        result = _dedup_transcription("hello world test hello world test")
        assert result == "hello world test"

    def test_case_insensitive_dedup(self):
        result = _dedup_transcription("Hello World hello world")
        # Should dedup, keeping the first half's casing
        assert result.lower() == "hello world"

    def test_does_not_dedup_partial_overlap(self):
        text = "hello world hello foo"
        assert _dedup_transcription(text) == text

    def test_empty_string(self):
        assert _dedup_transcription("") == ""
