"""search.py 测试 — 基于 requests + BeautifulSoup 的谷歌学术搜索."""

import time
import pytest
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from scholar_search.search import (
    _pub_to_dict,
    _parse_search_page,
    _check_captcha,
    search_papers,
    get_paper_detail,
    search_by_url,
    _jitter,
    _is_rate_limit_error,
    _run_with_timeout,
    _make_session,
    _fetch_external_abstract,
    _fetch_arxiv_abstract,
    _fetch_meta_abstract,
)


def _make_result_html(title="Test Paper", authors_text="A Smith, B Jones - NeurIPS 2023",
                      abstract="A great paper.", citations=42, url="https://arxiv.org/123",
                      full_abstract=""):
    fma = ""
    if full_abstract:
        fma = f"""<div class="gs_fma_abs"><div class="gs_fma_grad"></div>
<div class="gs_fma_snp"><div><div>{full_abstract}</div></div></div>
<div class="gs_fma_fons"><div class="gs_fma_fon">Publisher</div></div></div>"""
    return f"""<div class="gs_r gs_or gs_scl">
<div class="gs_ri">
<h3 class="gs_rt"><a href="{url}">{title}</a></h3>
<div class="gs_a">{authors_text}</div>
<div class="gs_rs">{abstract}</div>
<div class="gs_fl"><a href="#">Cited by {citations}</a></div>
</div>
{fma}
</div>"""


# ---- _jitter ----


class TestJitter:
    def test_returns_in_range(self):
        for _ in range(20):
            v = _jitter(2.0, 0.5)
            assert 1.0 <= v <= 3.0


# ---- _is_rate_limit_error ----


class TestIsRateLimit:
    def test_403(self):
        assert _is_rate_limit_error(RuntimeError("HTTP 403 Forbidden"))

    def test_429(self):
        assert _is_rate_limit_error(RuntimeError("429 Too Many"))

    def test_captcha(self):
        assert _is_rate_limit_error(RuntimeError("captcha required"))

    def test_normal_error(self):
        assert not _is_rate_limit_error(RuntimeError("timeout"))


# ---- _run_with_timeout ----


class TestRunWithTimeout:
    def test_returns_result(self):
        assert _run_with_timeout(lambda: 99, timeout=5) == 99

    def test_timeout_raises(self):
        def slow():
            time.sleep(10)
            return 1
        with pytest.raises(TimeoutError):
            _run_with_timeout(slow, timeout=0.1)

    def test_exception_propagates(self):
        with pytest.raises(ValueError, match="boom"):
            _run_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), timeout=5)


# ---- _pub_to_dict ----


class TestPubToDict:
    def test_full_pub(self):
        html = _make_result_html("GNN Papers", "Alice, Bob - ICML 2023", "Deep learning.", 15, "http://x.com")
        soup = BeautifulSoup(html, "lxml").select_one("div.gs_r.gs_or.gs_scl")
        result = _pub_to_dict(soup)
        assert result["title"] == "GNN Papers"
        assert result["authors"] == ["Alice", "Bob"]
        assert result["year"] == "2023"
        assert result["citations"] == 15
        assert result["abstract"] == "Deep learning."
        assert result["url"] == "http://x.com"

    def test_fma_abs_priority(self):
        """gs_fma_abs (展开面板) 的完整摘要优先级高于 gs_rs 的截断版."""
        short = "Short snippet..."
        full = "This is the full abstract with much more content than the snippet."
        html = _make_result_html(
            "Paper X", "Author - 2024", short, 10, "http://x.com",
            full_abstract=full,
        )
        soup = BeautifulSoup(html, "lxml").select_one("div.gs_r.gs_or.gs_scl")
        result = _pub_to_dict(soup)
        assert result["abstract"] == full

    def test_fma_abs_empty_falls_back_to_gs_rs(self):
        """gs_fma_abs 存在但 gs_fma_snp 为空时，回退到 gs_rs."""
        html = """<div class="gs_r gs_or gs_scl">
<div class="gs_ri">
<h3 class="gs_rt"><a href="http://x.com">Paper Y</a></h3>
<div class="gs_a">Author - 2024</div>
<div class="gs_rs">Snippet from gs_rs.</div>
<div class="gs_fl"><a href="#">Cited by 5</a></div>
</div>
<div class="gs_fma_abs"><div class="gs_fma_grad"></div><div class="gs_fma_snp"></div></div>
</div>"""
        soup = BeautifulSoup(html, "lxml").select_one("div.gs_r.gs_or.gs_scl")
        result = _pub_to_dict(soup)
        assert result["abstract"] == "Snippet from gs_rs."


# ---- _parse_search_page ----


class TestParseSearchPage:
    def test_parses_multiple(self):
        html = (_make_result_html("A", "X", "abs", 1, "http://a.com")
                + _make_result_html("B", "Y", "abs2", 2, "http://b.com"))
        papers = _parse_search_page(html)
        assert len(papers) == 2
        assert papers[0]["title"] == "A"
        assert papers[1]["title"] == "B"


# ---- _check_captcha ----


class TestCheckCaptcha:
    def test_no_captcha(self):
        _check_captcha("<html>normal page</html>")

    def test_captcha_raises(self):
        with pytest.raises(RuntimeError, match="验证码"):
            _check_captcha("<html>sorry we need a captcha</html>")


# ---- _make_session ----


class TestMakeSession:
    def test_returns_session_with_headers(self):
        s = _make_session()
        assert "User-Agent" in s.headers
        assert s.headers["Accept-Language"] == "en-US,en;q=0.9"
        # 验证 SSL retry adapter 已挂载
        assert "https://" in s.adapters


# ---- search_papers ----


class TestSearchPapers:
    def test_basic_search(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        papers_first = [{"title": "Paper A"}, {"title": "Paper B"}]
        with patch("scholar_search.search._search_request", side_effect=[papers_first, []]):
            with patch("scholar_search.search.time.sleep"):
                results = search_papers("test", num_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Paper A"

    def test_num_results_cap(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        papers = [{"title": f"P{i}"} for i in range(10)]
        with patch("scholar_search.search._search_request", side_effect=[papers, []]):
            with patch("scholar_search.search.time.sleep"):
                results = search_papers("test", num_results=3)
        assert len(results) == 3

    def test_retry_exhausted(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_RETRIES", "2")
        with patch("scholar_search.search._search_request", side_effect=RuntimeError("Boom")):
            with patch("scholar_search.search.time.sleep"):
                with pytest.raises(RuntimeError, match="谷歌学术搜索失败"):
                    search_papers("test", num_results=5)


# ---- get_paper_detail ----


class TestGetPaperDetail:
    def test_success(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        mock_paper = {"title": "Found", "url": ""}
        with patch("scholar_search.search._search_request", return_value=[mock_paper]):
            with patch("scholar_search.search._fetch_external_abstract", return_value=None):
                with patch("scholar_search.search.time.sleep"):
                    result = get_paper_detail("test")
        assert result == mock_paper

    def test_not_found(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        with patch("scholar_search.search._search_request", return_value=[]):
            with patch("scholar_search.search.time.sleep"):
                result = get_paper_detail("x")
        assert result is None

    def test_full_abstract_enrichment(self, monkeypatch):
        """get_paper_detail 应使用外部源的完整摘要替换 Google Scholar 片段."""
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        mock_paper = {"title": "X", "abstract": "Short...", "url": "https://arxiv.org/abs/1234.5678"}
        full_abstract = "This is the complete abstract with all details."

        with patch("scholar_search.search._search_request", return_value=[mock_paper]):
            with patch("scholar_search.search._fetch_external_abstract", return_value=full_abstract):
                with patch("scholar_search.search.time.sleep"):
                    result = get_paper_detail("test")
        assert result["abstract"] == full_abstract

    def test_external_abstract_failed_falls_back(self, monkeypatch):
        """外部摘要获取失败时保留 Google Scholar 片段."""
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        mock_paper = {"title": "X", "abstract": "Snippet.", "url": "https://springer.com/x"}
        with patch("scholar_search.search._search_request", return_value=[mock_paper]):
            with patch("scholar_search.search._fetch_external_abstract", return_value=None):
                with patch("scholar_search.search.time.sleep"):
                    result = get_paper_detail("test")
        assert result["abstract"] == "Snippet."


# ---- dedup ----


class TestDeduplicatePapers:
    def test_empty_and_single(self):
        from scholar_search.search import _deduplicate_papers
        assert _deduplicate_papers([]) == []
        single = [{"title": "Only", "citations": 5}]
        assert _deduplicate_papers(single) == single

    def test_exact_same_title_dedup(self):
        from scholar_search.search import _deduplicate_papers
        papers = [
            {"title": "GNN for Recommendation Systems", "citations": 10},
            {"title": "GNN for Recommendation Systems", "citations": 42},
        ]
        result = _deduplicate_papers(papers)
        assert len(result) == 1

    def test_different_title_kept(self):
        from scholar_search.search import _deduplicate_papers
        papers = [
            {"title": "GNN for Recommendation", "citations": 10},
            {"title": "NLP Transformers Survey", "citations": 5},
        ]
        result = _deduplicate_papers(papers)
        assert len(result) == 2

    def test_similar_but_different_kept(self):
        from scholar_search.search import _deduplicate_papers
        papers = [
            {"title": "graph neural networks for recommendation", "citations": 5},
            {"title": "graph neural network for recommender systems", "citations": 15},
        ]
        result = _deduplicate_papers(papers)
        assert len(result) == 2

    def test_case_insensitive_dedup(self):
        from scholar_search.search import _deduplicate_papers
        papers = [
            {"title": "Deep Learning Survey", "citations": 10},
            {"title": "deep learning survey", "citations": 42},
        ]
        result = _deduplicate_papers(papers)
        assert len(result) == 1


# ---- search_by_url ----


class TestSearchByUrl:
    def test_success(self, monkeypatch):
        monkeypatch.delenv("SCHOLAR_RETRIES", raising=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _make_result_html("URL Paper")
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            with patch("scholar_search.search.time.sleep"):
                result = search_by_url("https://scholar.google.com/xxx")
        assert result is not None
        assert result["title"] == "URL Paper"

    def test_403_retry_exhausted(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_RETRIES", "2")
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            with patch("scholar_search.search.time.sleep"):
                with pytest.raises(RuntimeError, match="通过 URL 获取论文失败"):
                    search_by_url("https://x.com")


# ---- _fetch_external_abstract ----

class TestFetchExternalAbstract:
    def test_empty_url_returns_none(self):
        assert _fetch_external_abstract("") is None
        assert _fetch_external_abstract(None) is None

    def test_arxiv_url_dispatches(self):
        """arxiv URL 应走 arxiv API 路径."""
        with patch("scholar_search.search._fetch_arxiv_abstract", return_value="Full arxiv abstract") as mock_arxiv:
            with patch("scholar_search.search._fetch_meta_abstract") as mock_meta:
                result = _fetch_external_abstract("https://arxiv.org/abs/1234.5678")
        assert result == "Full arxiv abstract"
        mock_arxiv.assert_called_once_with("1234.5678")
        mock_meta.assert_not_called()

    def test_non_arxiv_url_uses_meta(self):
        """非 arxiv URL 应走 meta 标签路径."""
        with patch("scholar_search.search._fetch_meta_abstract", return_value="Meta abstract") as mock_meta:
            result = _fetch_external_abstract("https://springer.com/article/123")
        assert result == "Meta abstract"
        mock_meta.assert_called_once_with("https://springer.com/article/123")

    def test_both_fail_returns_none(self):
        with patch("scholar_search.search._fetch_arxiv_abstract", return_value=None):
            with patch("scholar_search.search._fetch_meta_abstract", return_value=None):
                result = _fetch_external_abstract("https://other.com/paper")
        assert result is None


# ---- _fetch_arxiv_abstract ----

class TestFetchArxivAbstract:
    def test_success(self):
        xml_resp = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test Paper</title>
    <summary>  Full abstract text here.  </summary>
  </entry>
</feed>'''
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = xml_resp
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_arxiv_abstract("1234.5678")
        assert result == "Full abstract text here."

    def test_http_error_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_arxiv_abstract("1234.5678")
        assert result is None

    def test_exception_returns_none(self):
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.side_effect = Exception("Network error")

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_arxiv_abstract("1234.5678")
        assert result is None


# ---- _fetch_meta_abstract ----

class TestFetchMetaAbstract:
    def test_citation_abstract_priority(self):
        """citation_abstract 优先级最高."""
        long_abstract = "A" * 100
        html = f'<html><head><meta name="citation_abstract" content="{long_abstract}">'
        html += '<meta name="description" content="Short desc"></head></html>'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_meta_abstract("https://example.com/paper")
        assert result == long_abstract

    def test_description_fallback(self):
        """无 citation_abstract 时用 description."""
        long_desc = "B" * 100
        html = f'<html><head><meta name="description" content="{long_desc}"></head></html>'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_meta_abstract("https://example.com/paper")
        assert result == long_desc

    def test_short_description_filtered(self):
        """太短的 description 被过滤."""
        html = '<html><head><meta name="description" content="Short"></head></html>'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = html
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_meta_abstract("https://example.com/paper")
        assert result is None

    def test_pdf_skipped(self):
        """PDF 响应跳过 meta 提取."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_meta_abstract("https://example.com/paper.pdf")
        assert result is None

    def test_http_error_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = lambda s, *a: None
        mock_session.get.return_value = mock_resp

        with patch("scholar_search.search._make_session", return_value=mock_session):
            result = _fetch_meta_abstract("https://example.com/paper")
        assert result is None
