"""bing_search.py 测试 — Bing 学术搜索与解析."""

import pytest
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from scholar_search.bing_search import (
    _bing_pub_to_dict,
    _parse_bing_search_page,
    search_papers_bing,
    get_paper_detail_bing,
    _fetch_bing_profile_abstract,
)


def _make_bing_result_html(title="Test Paper",
                           authors="Fang Liu · Juan Wang",
                           year="2023",
                           venue="IEEE Access",
                           citations=6,
                           abstract="To improve the accuracy of recommendation algorithms.",
                           profile_id="a41ed3dcdc67eb299cf6bf515857bce5"):
    """构造 Bing 学术搜索结果 HTML."""
    author_links = " · ".join(
        f'<a href="/academic/search?q={a.strip().replace(" ", "+")}">{a.strip()}</a>'
        for a in authors.split("·")
    )
    return f"""<li class="aca_algo">
<h2><a href="/academic/profile?id={profile_id}&encoded=0&v=paper_preview">{title}</a></h2>
<div class="aca_caption">
<div class="caption_author">{author_links}</div>
<div class="caption_venue">{year} · {venue}<span class="caption_cite_count">|</span>被引数：{citations}</div>
<div class="caption_abstract"><p>{abstract}</p></div>
</div>
</li>"""


# ---- _bing_pub_to_dict ----


class TestBingPubToDict:
    def test_full_parsing(self):
        html = _make_bing_result_html(
            title="GNN Paper",
            authors="Alice · Bob",
            year="2022",
            venue="Nature",
            citations=42,
            abstract="A great paper about GNN.",
            profile_id="abc123",
        )
        soup = BeautifulSoup(html, "lxml").select_one("li.aca_algo")
        result = _bing_pub_to_dict(soup)
        assert result["title"] == "GNN Paper"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["year"] == "2022"
        assert result["venue"] == "Nature"
        assert result["citations"] == 42
        assert result["abstract"] == "A great paper about GNN."
        assert result["profile_id"] == "abc123"
        assert result["engine"] == "bing"

    def test_invalid_year(self):
        """非标准年份格式不提取."""
        html = _make_bing_result_html(year="no year here")
        soup = BeautifulSoup(html, "lxml").select_one("li.aca_algo")
        result = _bing_pub_to_dict(soup)
        assert result["year"] == ""


# ---- _parse_bing_search_page ----


class TestParseBingSearchPage:
    def test_parses_multiple(self):
        html = _make_bing_result_html(title="Paper A") + _make_bing_result_html(title="Paper B")
        papers = _parse_bing_search_page(html)
        assert len(papers) == 2
        assert papers[0]["title"] == "Paper A"
        assert papers[1]["title"] == "Paper B"

    def test_empty_page(self):
        assert _parse_bing_search_page("<html></html>") == []


# ---- search_papers_bing ----


class TestSearchPapersBing:
    def test_basic_search(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        papers = [{"title": "Bing A"}, {"title": "Bing B"}]
        with patch("scholar_search.bing_search._bing_search_request",
                   side_effect=[papers, []]):
            with patch("scholar_search.bing_search.time.sleep"):
                results = search_papers_bing("test", num_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Bing A"

    def test_retry_exhausted(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_RETRIES", "2")
        with patch("scholar_search.bing_search._bing_search_request",
                   side_effect=RuntimeError("Bing error")):
            with patch("scholar_search.bing_search.time.sleep"):
                with pytest.raises(RuntimeError, match="Bing 学术搜索失败"):
                    search_papers_bing("test")


# ---- get_paper_detail_bing ----


class TestGetPaperDetailBing:
    def test_success_with_profile(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        mock_paper = {"title": "Bing Paper", "profile_id": "abc", "abstract": "Snippet."}
        with patch("scholar_search.bing_search._bing_search_request",
                   return_value=[mock_paper]):
            with patch("scholar_search.bing_search._fetch_bing_profile_abstract",
                       return_value="Full abstract from profile."):
                with patch("scholar_search.bing_search.time.sleep"):
                    result = get_paper_detail_bing("test")
        assert result["abstract"] == "Full abstract from profile."

    def test_no_profile_fallback(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        mock_paper = {"title": "Bing Paper", "profile_id": "", "abstract": "Snippet only."}
        with patch("scholar_search.bing_search._bing_search_request",
                   return_value=[mock_paper]):
            with patch("scholar_search.bing_search.time.sleep"):
                result = get_paper_detail_bing("test")
        assert result["abstract"] == "Snippet only."

    def test_not_found(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        with patch("scholar_search.bing_search._bing_search_request", return_value=[]):
            with patch("scholar_search.bing_search.time.sleep"):
                result = get_paper_detail_bing("x")
        assert result is None


# ---- _fetch_bing_profile_abstract ----


class TestFetchBingProfileAbstract:
    def test_success(self):
        html = '<html><div class="acapp_abstract_content">Full abstract here.</div></html>'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.bing_search._make_session", return_value=mock_session):
            result = _fetch_bing_profile_abstract("abc123")
        assert result == "Full abstract here."

    def test_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.bing_search._make_session", return_value=mock_session):
            result = _fetch_bing_profile_abstract("abc123")
        assert result is None

    def test_exception_returns_none(self):
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.side_effect = Exception("Network error")

        with patch("scholar_search.bing_search._make_session", return_value=mock_session):
            result = _fetch_bing_profile_abstract("abc123")
        assert result is None
