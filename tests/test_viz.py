"""viz.py 测试 — matplotlib 图表 + HTTP 服务端."""

import json
import pytest
from unittest.mock import patch

from scholar_search.viz import (
    generate_relevance_chart,
    _parse_input,
    _bar_chart,
    _keyword_chart,
    _empty_chart,
    _truncate,
    _escape,
)


def sample_papers(n=5):
    return [{"title": f"Paper {i}", "relevance_score": 85 - i * 15,
             "abstract": f"Abstract for paper {i} about neural networks."} for i in range(n)]


# ---- _escape / _truncate ----


class TestEscape:
    def test_ampersand(self):
        assert "&amp;" in _escape("a & b")

    def test_empty(self):
        assert _escape("") == ""


class TestTruncate:
    def test_short(self):
        assert _truncate("hello", 10) == "hello"

    def test_long(self):
        r = _truncate("very long title here yes", 15)
        assert len(r) == 15
        assert r.endswith("...")


# ---- _parse_input ----


class TestParseInput:
    def test_list(self):
        assert len(_parse_input([{"title": "A"}])) == 1

    def test_wrapped(self):
        assert len(_parse_input({"ranked_papers": [{"title": "A"}]})) == 1

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="无法从输入中提取"):
            _parse_input("not json")

    def test_empty_list(self):
        assert _parse_input([]) == []


# ---- _bar_chart ----


class TestBarChart:
    def test_returns_png(self):
        png = _bar_chart(sample_papers(3), "test")
        assert png[:4] == b"\x89PNG"
        assert len(png) > 1000

    def test_single_paper(self):
        png = _bar_chart(sample_papers(1), "")
        assert png[:4] == b"\x89PNG"

    def test_truncates_to_20(self):
        png = _bar_chart(sample_papers(22), "")
        assert png[:4] == b"\x89PNG"


# ---- _keyword_chart ----


class TestKeywordChart:
    def test_returns_png(self):
        png = _keyword_chart(sample_papers(4), "topic")
        assert png[:4] == b"\x89PNG"

    def test_empty_texts(self):
        papers = [{"title": "", "abstract": ""}, {"title": "", "abstract": ""}]
        png = _keyword_chart(papers, "")
        assert png[:4] == b"\x89PNG"


# ---- _empty_chart ----


class TestEmptyChart:
    def test_returns_png(self):
        png = _empty_chart("Title", "reason here")
        assert png[:4] == b"\x89PNG"


# ---- generate_relevance_chart ----


class TestGenerateRelevanceChart:
    def test_basic(self):
        with patch("scholar_search.viz._start_server"):
            result = json.loads(generate_relevance_chart(
                json.dumps({"ranked_papers": sample_papers(4)}), "test"
            ))
        assert "url" in result
        assert result["url"].startswith("http://localhost:")
        assert "message" in result
        assert "`http://" in result["message"]
        assert "bar.png" in result["message"]
        assert len(result["charts"]) == 3

    def test_chart_urls(self):
        with patch("scholar_search.viz._start_server"):
            result = json.loads(generate_relevance_chart(
                json.dumps({"ranked_papers": sample_papers(3)}), "x"
            ))
        for ch in result["charts"]:
            assert "name" in ch
            assert "url" in ch
            assert ch["url"].startswith("http://")

    def test_empty_papers_raises(self):
        with pytest.raises(ValueError, match="不包含有效的论文数据"):
            generate_relevance_chart(json.dumps([]))

    def test_raw_list_input(self):
        with patch("scholar_search.viz._start_server"):
            result = json.loads(generate_relevance_chart(sample_papers(2), ""))
        assert len(result["charts"]) == 3


# ---- cluster chart (basic smoke) ----


class TestClusterChart:
    def test_returns_png(self):
        from scholar_search.viz import _cluster_chart
        png = _cluster_chart(sample_papers(6), "clustering")
        assert png[:4] == b"\x89PNG"

    def test_too_few_papers(self):
        from scholar_search.viz import _cluster_chart
        png = _cluster_chart(sample_papers(1), "")
        assert png[:4] == b"\x89PNG"
