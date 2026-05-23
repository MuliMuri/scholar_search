"""server.py 测试 — MCP Server 工具函数."""

import json
import pytest
from unittest.mock import patch


# ---- search_papers tool ----


class TestSearchPapersTool:
    @pytest.fixture
    def tool_func(self):
        from server import search_papers
        return search_papers

    @pytest.mark.asyncio
    async def test_success(self, tool_func):
        mock_results = [
            {"title": "Paper A", "authors": ["Alice"]},
            {"title": "Paper B", "authors": ["Bob"]},
        ]

        with patch("server._search_papers", return_value=mock_results) as mock_search:
            result = await tool_func(query="test", num_results=5)

        data = json.loads(result)
        assert data["query"] == "test"
        assert data["total_found"] == 2
        assert len(data["papers"]) == 2
        assert data["papers"][0]["title"] == "Paper A"
        mock_search.assert_called_once_with(
            query="test", num_results=5, year_low=None, year_high=None
        )

    @pytest.mark.asyncio
    async def test_with_year_filters(self, tool_func):
        mock_results = [{"title": "X", "year": 2022}]

        with patch("server._search_papers", return_value=mock_results) as mock_search:
            result = await tool_func(query="test", year_low=2020, year_high=2024)

        data = json.loads(result)
        assert data["total_found"] == 1
        mock_search.assert_called_once_with(
            query="test", num_results=10, year_low=2020, year_high=2024
        )

    @pytest.mark.asyncio
    async def test_runtime_error(self, tool_func):
        with patch("server._search_papers", side_effect=RuntimeError("Proxy error")):
            result = await tool_func(query="test")

        data = json.loads(result)
        assert "error" in data
        assert "Proxy error" in data["error"]

    @pytest.mark.asyncio
    async def test_empty_results(self, tool_func):
        with patch("server._search_papers", return_value=[]):
            result = await tool_func(query="obscure_query")

        data = json.loads(result)
        assert data["total_found"] == 0
        assert data["papers"] == []


# ---- get_paper_detail tool ----


class TestGetPaperDetailTool:
    @pytest.fixture
    def tool_func(self):
        from server import get_paper_detail
        return get_paper_detail

    @pytest.mark.asyncio
    async def test_by_title_success(self, tool_func):
        mock_paper = {"title": "Test Paper", "abstract": "Test abstract"}

        with patch("server._get_paper_detail", return_value=mock_paper) as mock_get:
            result = await tool_func(title="Test Paper")

        data = json.loads(result)
        assert data["title"] == "Test Paper"
        mock_get.assert_called_once_with("Test Paper")

    @pytest.mark.asyncio
    async def test_by_url_success(self, tool_func):
        mock_paper = {"title": "URL Paper"}

        with patch("server._search_by_url", return_value=mock_paper) as mock_url:
            result = await tool_func(url="https://scholar.google.com/xxx")

        data = json.loads(result)
        assert data["title"] == "URL Paper"
        mock_url.assert_called_once_with("https://scholar.google.com/xxx")

    @pytest.mark.asyncio
    async def test_url_priority_over_title(self, tool_func):
        """同时提供 title 和 url 时优先使用 url."""
        mock_url_paper = {"title": "URL Paper"}

        with patch("server._search_by_url", return_value=mock_url_paper) as mock_url:
            with patch("server._get_paper_detail") as mock_title:
                result = await tool_func(title="Title Paper", url="https://x.com")

        mock_url.assert_called_once()
        mock_title.assert_not_called()
        data = json.loads(result)
        assert data["title"] == "URL Paper"

    @pytest.mark.asyncio
    async def test_no_params(self, tool_func):
        result = await tool_func(title="", url="")

        data = json.loads(result)
        assert "error" in data
        assert "title" in data["error"] or "url" in data["error"]

    @pytest.mark.asyncio
    async def test_not_found(self, tool_func):
        with patch("server._get_paper_detail", return_value=None):
            result = await tool_func(title="Nonexistent")

        data = json.loads(result)
        assert "error" in data
        assert "未找到" in data["error"]

    @pytest.mark.asyncio
    async def test_runtime_error(self, tool_func):
        with patch("server._get_paper_detail", side_effect=RuntimeError("Timeout")):
            result = await tool_func(title="test")

        data = json.loads(result)
        assert "error" in data
        assert "Timeout" in data["error"]


# ---- analyze_relevance tool ----


class TestAnalyzeRelevanceTool:
    @pytest.fixture
    def tool_func(self):
        from server import analyze_relevance
        return analyze_relevance

    @pytest.mark.asyncio
    async def test_success(self, tool_func):
        mock_result = {
            "ranked_papers": [
                {"title": "A", "relevance_score": 85.0},
                {"title": "B", "relevance_score": 30.0},
            ],
            "summary": {"avg_score": 57.5, "top_keywords": ["neural"], "cluster_hint": "hint"},
        }

        with patch("server._analyze_relevance", return_value=mock_result) as mock_analyze:
            result = await tool_func(topic="test", papers_json='[{"title":"A"},{"title":"B"}]')

        data = json.loads(result)
        assert data["ranked_papers"][0]["title"] == "A"
        assert data["summary"]["avg_score"] == 57.5
        mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_value_error(self, tool_func):
        with patch("server._analyze_relevance", side_effect=ValueError("Invalid input")):
            result = await tool_func(topic="test", papers_json="[]")

        data = json.loads(result)
        assert "error" in data
        assert "Invalid input" in data["error"]

    @pytest.mark.asyncio
    async def test_runtime_error(self, tool_func):
        with patch("server._analyze_relevance", side_effect=RuntimeError("Analysis failed")):
            result = await tool_func(topic="test", papers_json='[{"title":"X"}]')

        data = json.loads(result)
        assert "error" in data


# ---- FastMCP instance ----


class TestFastMCPInstance:
    def test_mcp_name(self):
        from server import mcp
        assert mcp.name == "scholar-search-mcp"

    def test_tools_registered(self):
        from server import mcp
        tool_names = [t.name for t in mcp._tool_manager._tools.values()]
        assert "search_papers" in tool_names
        assert "get_paper_detail" in tool_names
        assert "analyze_relevance" in tool_names
        assert "generate_relevance_chart" in tool_names

    def test_chart_tool_runs(self):
        import asyncio
        from server import generate_relevance_chart
        with patch("server._generate_chart") as mock_chart:
            mock_chart.return_value = '{"url": "file:///tmp/test.html", "path": "/tmp/test.html", "filename": "test.html"}'
            result = asyncio.run(generate_relevance_chart(topic="t", papers_json='[{"title":"X"}]'))
            assert "file://" in result

    def test_chart_tool_error_path(self):
        """generate_relevance_chart 错误处理 (lines 152-153)."""
        import asyncio
        from server import generate_relevance_chart
        with patch("server._generate_chart", side_effect=ValueError("bad data")):
            result = asyncio.run(generate_relevance_chart(topic="t", papers_json="x"))
            assert "error" in result

    def test_main_function_calls_run(self):
        from server import main
        assert callable(main)
        with patch("server.mcp.run") as mock_run:
            main()
            mock_run.assert_called_once_with(transport="stdio")

    def test_dunder_main_guard(self):
        """覆盖 server.py 的 if __name__ == '__main__' 分支 (line 123)."""
        import runpy
        from mcp.server.fastmcp import FastMCP
        with patch.object(FastMCP, "run") as mock_run:
            runpy.run_path("C:/Users/mulim/Desktop/Project/scholar_search/server.py",
                           run_name="__main__")
            mock_run.assert_called_once_with(transport="stdio")
