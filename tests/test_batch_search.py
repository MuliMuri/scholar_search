"""batch_search.py 测试."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from scripts.batch_search import run_batch, _dedup_key


class TestDedupKey:
    def test_lowercases(self):
        assert _dedup_key({"title": "ABC"}) == "abc"

    def test_empty_title(self):
        assert _dedup_key({"title": ""}) == ""

    def test_missing_title(self):
        assert _dedup_key({}) == ""


class TestRunBatch:
    def test_basic(self):
        papers = [{"title": "Paper A"}, {"title": "Paper B"}]
        with patch("scripts.batch_search.bing_search", return_value=papers) as mock_search:
            with patch("scripts.batch_search.time.sleep"):
                result = run_batch(["test query"], engine="bing", num_results=5)
        assert len(result) == 2
        mock_search.assert_called_once()

    def test_dedup_across_keywords(self):
        papers1 = [{"title": "Paper A"}, {"title": "Paper B"}]
        papers2 = [{"title": "Paper A"}, {"title": "Paper C"}]
        with patch("scripts.batch_search.bing_search", side_effect=[papers1, papers2]):
            with patch("scripts.batch_search.time.sleep"):
                result = run_batch(["q1", "q2"], engine="bing", num_results=3)
        assert len(result) == 3

    def test_skips_empty_and_comments(self):
        papers = [{"title": "X"}]
        with patch("scripts.batch_search.bing_search", return_value=papers):
            with patch("scripts.batch_search.time.sleep"):
                result = run_batch(["", "# comment", "real query"], engine="bing", num_results=1)
        assert len(result) == 1

    def test_search_error_continues(self):
        with patch("scripts.batch_search.bing_search", side_effect=RuntimeError("fail")):
            with patch("scripts.batch_search.time.sleep"):
                result = run_batch(["q1", "q2"], engine="bing", num_results=3)
        assert len(result) == 0

    def test_no_dedup(self):
        papers = [{"title": "X"}, {"title": "X"}]
        with patch("scripts.batch_search.bing_search", return_value=papers):
            with patch("scripts.batch_search.time.sleep"):
                result = run_batch(["q1"], engine="bing", num_results=3, dedup=False)
        assert len(result) == 2

    def test_google_engine(self):
        papers = [{"title": "G"}]
        with patch("scripts.batch_search.gs_search", return_value=papers):
            with patch("scripts.batch_search.time.sleep"):
                result = run_batch(["q1"], engine="google", num_results=3)
        assert len(result) == 1
