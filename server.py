"""Google Scholar MCP Server — 供 CherryStudio 等 AI 客户端调用.

Tool 列表:
- search_papers: 搜索谷歌学术论文
- get_paper_detail: 获取单篇论文详细信息
- analyze_relevance: 论文主题相关性分析
- generate_relevance_chart: 生成相关性 HTML 柱状图
"""

import asyncio
import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from scholar_search.search import search_papers as _search_papers
from scholar_search.search import get_paper_detail as _get_paper_detail
from scholar_search.search import search_by_url as _search_by_url
from scholar_search.analysis import analyze_relevance as _analyze_relevance
from scholar_search.viz import generate_relevance_chart as _generate_chart

# 日志配置：INFO 及以上写入文件，stderr 仅输出 WARNING 及以上
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scholar_search.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),  # stderr — CherryStudio 捕获此输出
    ],
)
# stderr 只输出 WARNING 及以上
logging.getLogger().handlers[1].setLevel(logging.WARNING)
# 关闭第三方库的 INFO 日志
logging.getLogger("scholarly").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

logger = logging.getLogger("scholar-mcp")

mcp = FastMCP("scholar-search-mcp")


@mcp.tool()
async def search_papers(
    query: str,
    num_results: int = 10,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
) -> str:
    """搜索谷歌学术论文，自动去重.

    基于标题词汇重叠率（>=75%）自动去重，保留引用量更高的版本。
    单次请求最多返回 30 条结果，已内置请求间隔和限流重试。

    Args:
        query: 搜索关键词，英文效果最佳，如 "graph neural network recommendation"
        num_results: 返回论文数量，1-30，默认 10
        year_low: 发表年份下限，如 2020
        year_high: 发表年份上限，如 2024
    """
    logger.info("搜索: %s (num=%d, year=%s-%s)", query, num_results, year_low, year_high)

    try:
        papers = await asyncio.to_thread(
            _search_papers,
            query=query,
            num_results=num_results,
            year_low=year_low,
            year_high=year_high,
        )
    except RuntimeError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    result = {
        "query": query,
        "total_found": len(papers),
        "papers": papers,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_paper_detail(
    title: str = "",
    url: str = "",
) -> str:
    """获取单篇论文的详细信息，包括摘要、引用量等.

    Args:
        title: 论文标题（精确匹配效果更好）
        url: 论文的 Google Scholar URL（与 title 二选一）
    """
    logger.info("获取详情: title=%s, url=%s", title, url)

    try:
        if url:
            paper = await asyncio.to_thread(_search_by_url, url)
        elif title:
            paper = await asyncio.to_thread(_get_paper_detail, title)
        else:
            return json.dumps({"error": "请提供 title 或 url 参数"}, ensure_ascii=False)

        if paper is None:
            return json.dumps({"error": "未找到匹配的论文"}, ensure_ascii=False)

    except RuntimeError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps(paper, ensure_ascii=False, indent=2)


@mcp.tool()
async def analyze_relevance(
    topic: str,
    papers_json: str,
) -> str:
    """分析一组论文与研究主题的相关性，返回排序结果和分析摘要.

    典型用法：先调用 search_papers 获取论文列表，再将返回的 JSON 传入此方法。

    Args:
        topic: 研究主题描述（英文 1-3 句话），如 "using graph neural networks for collaborative filtering recommendation systems"
        papers_json: search_papers 返回的 JSON 字符串，需包含 papers 数组
    """
    logger.info("相关性分析: topic=%s", topic[:80])

    try:
        result = _analyze_relevance(topic=topic, papers_json=papers_json)
    except (ValueError, RuntimeError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def generate_relevance_chart(
    topic: str = "",
    papers_json: str = "",
) -> str:
    """生成多角度论文相关性分析图表，启动本地 HTTP 服务端返回链接.

    生成 3 张图表：相关性柱状图、K-Means 聚类散点图、TF-IDF 关键词分析。
    图表通过本地 HTTP 服务端 (http://localhost:8765) 提供，在浏览器中打开。
    返回 message 字段含 markdown 链接和原始 URL，客户端可渲染或复制。

    典型用法：先调用 analyze_relevance 获取排序结果，再将其 JSON 传入此方法。

    Args:
        topic: 研究主题描述（图表标题）
        papers_json: analyze_relevance 返回的 JSON 字符串，需包含 ranked_papers 数组
    """
    logger.info("生成图表: topic=%s", topic[:60])

    try:
        result = _generate_chart(topic=topic, papers_json=papers_json)
    except (ValueError, RuntimeError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return result


def main() -> None:
    """启动 MCP Server (stdio 模式)."""
    logger.info("Scholar Search MCP Server 启动中...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
