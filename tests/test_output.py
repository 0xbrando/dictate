"""Tests for dictate.output — text aggregation and output handlers."""

from __future__ import annotations

import pytest

from dictate.output import MAX_AGGREGATED_CHARS, TextAggregator


class TestTextAggregator:
    def test_empty_initially(self):
        agg = TextAggregator()
        assert agg.full_text == ""

    def test_single_append(self):
        agg = TextAggregator()
        result = agg.append("Hello world.")
        assert result == "Hello world."
        assert agg.full_text == "Hello world."

    def test_multiple_appends(self):
        agg = TextAggregator()
        agg.append("First line.")
        agg.append("Second line.")
        assert agg.full_text == "First line.\nSecond line."

    def test_strips_whitespace(self):
        agg = TextAggregator()
        agg.append("  Hello.  ")
        assert agg.full_text == "Hello."

    def test_clear(self):
        agg = TextAggregator()
        agg.append("Something.")
        agg.clear()
        assert agg.full_text == ""

    def test_append_after_clear(self):
        agg = TextAggregator()
        agg.append("Before.")
        agg.clear()
        agg.append("After.")
        assert agg.full_text == "After."

    def test_truncation_at_max_chars(self):
        agg = TextAggregator()
        # Write more than MAX_AGGREGATED_CHARS
        big_text = "x" * (MAX_AGGREGATED_CHARS + 1000)
        agg.append(big_text)
        assert len(agg.full_text) <= MAX_AGGREGATED_CHARS

    def test_newline_joining(self):
        agg = TextAggregator()
        agg.append("Line one.  ")  # trailing space should be stripped
        agg.append("Line two.")
        # The aggregator rstrips previous text before joining
        assert agg.full_text == "Line one.\nLine two."

    def test_return_value_matches_full_text(self):
        agg = TextAggregator()
        result1 = agg.append("A.")
        result2 = agg.append("B.")
        assert result2 == agg.full_text
