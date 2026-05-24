"""Bing 学术搜索 — 免代理、Google Scholar 的备选/默认方案.

搜索页: https://cn.bing.com/academic/search?q=...&offset=N
详情页: https://cn.bing.com/academic/profile?id=...
"""

import re
import time
import random
import logging
import concurrent.futures

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup

from .config import get_timeout, get_retries

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://cn.bing.com/academic/search"
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]), raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        "User-Agent": _random_ua(),
    })
    return s


def _jitter(base: float, factor: float = 0.5) -> float:
    return base * (1 + random.uniform(-factor, factor))


def _run_with_timeout(func, *args, timeout: int | None = None):
    t = timeout or get_timeout()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        return future.result(timeout=t)


def _bing_pub_to_dict(result_soup: BeautifulSoup) -> dict:
    """解析单个 Bing 学术搜索结果."""
    # 标题
    title_tag = result_soup.select_one("h2 a")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    url = ""
    profile_id = ""
    if title_tag:
        href = title_tag.get("href", "")
        url = f"https://cn.bing.com{href}" if href.startswith("/") else href
        # 提取 profile id
        id_match = re.search(r'id=([a-f0-9]+)', href)
        if id_match:
            profile_id = id_match.group(1)

    # 作者
    authors = []
    author_tags = result_soup.select("div.caption_author a")
    for a in author_tags:
        name = a.get_text(" ", strip=True)
        if name:
            authors.append(name)

    # 年份 / 期刊 / 引用数
    venue_tag = result_soup.select_one("div.caption_venue")
    venue_text = venue_tag.get_text(" ", strip=True) if venue_tag else ""

    year = ""
    year_match = re.search(r"\b(19|20)\d{2}\b", venue_text)
    if year_match:
        year = year_match.group(0)

    venue = ""
    if "·" in venue_text:
        after_dot = venue_text.split("·", 1)[1].strip()
        venue = after_dot.split("|")[0].strip()
    if not venue:
        venue = venue_text.split("|")[0].strip() if "|" in venue_text else ""

    citations = 0
    cite_tag = result_soup.select_one("span.caption_cite_count")
    if cite_tag:
        # 引用数在 span 之后: <span>|</span>被引数：42
        tail_text = cite_tag.next_sibling
        if tail_text:
            cite_match = re.search(r"(\d+)", str(tail_text))
            if cite_match:
                citations = int(cite_match.group(1))

    # 摘要片段
    abstract_tag = result_soup.select_one("div.caption_abstract p")
    abstract = abstract_tag.get_text(" ", strip=True) if abstract_tag else ""

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "citations": citations,
        "abstract": abstract,
        "url": url,
        "publisher": "",
        "author_ids": [],
        "profile_id": profile_id,
        "engine": "bing",
    }


def _parse_bing_search_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    papers = []
    for result in soup.select("li.aca_algo"):
        papers.append(_bing_pub_to_dict(result))
    return papers


def _bing_search_request(query: str, offset: int = 0,
                         year_low: int | None = None,
                         year_high: int | None = None) -> list[dict]:
    """执行单次 Bing 学术搜索请求."""
    params: dict = {"q": query, "offset": offset}
    if year_low:
        params["year_from"] = str(year_low)
    if year_high:
        params["year_to"] = str(year_high)

    session = _make_session()
    try:
        resp = session.get(_SEARCH_URL, params=params, timeout=get_timeout())
        if resp.status_code == 200:
            return _parse_bing_search_page(resp.text)
        elif resp.status_code in (403, 429):
            raise RuntimeError(f"Bing 学术限流 (HTTP {resp.status_code})")
        else:
            raise RuntimeError(f"Bing 学术返回 HTTP {resp.status_code}")
    finally:
        session.close()


def search_papers_bing(
    query: str,
    num_results: int = 10,
    year_low: int | None = None,
    year_high: int | None = None,
) -> list[dict]:
    """通过 Bing 学术搜索论文."""
    num_results = max(1, min(num_results, 30))

    def _do_search():
        results = []
        offset = 0
        time.sleep(_jitter(0.5))

        while len(results) < num_results:
            for attempt in range(get_retries()):
                try:
                    page = _bing_search_request(query, offset, year_low, year_high)
                    for paper in page:
                        if len(results) >= num_results:
                            break
                        results.append(paper)
                    break
                except RuntimeError as e:
                    if attempt >= get_retries() - 1:
                        raise RuntimeError(
                            f"Bing 学术搜索失败（已重试 {get_retries()} 次）: {e}"
                        ) from e
                    wait = min(2 ** attempt, 30)
                    logger.warning("Bing 搜索 '%s' 失败，%ds 后重试 (%d/%d)",
                                   query[:60], int(wait), attempt + 1, get_retries())
                    time.sleep(wait)

            if not page:
                break
            offset += 10
            time.sleep(_jitter(1.0, 0.4))

        return results

    return _run_with_timeout(_do_search)


def get_paper_detail_bing(title: str) -> dict | None:
    """根据标题搜索 Bing 学术，返回论文详情（含完整摘要）."""

    def _do_fetch():
        time.sleep(_jitter(0.5))
        for attempt in range(get_retries()):
            try:
                papers = _bing_search_request(title)
                if not papers:
                    return None
                paper = papers[0]
                # 尝试从 profile 页获取完整摘要
                profile_id = paper.get("profile_id", "")
                if profile_id:
                    full = _fetch_bing_profile_abstract(profile_id)
                    if full:
                        paper["abstract"] = full
                return paper
            except RuntimeError as e:
                if attempt >= get_retries() - 1:
                    raise RuntimeError(f"Bing 论文详情获取失败: {e}") from e
                time.sleep(min(2 ** attempt, 30))
        return None

    return _run_with_timeout(_do_fetch)


def _fetch_bing_profile_abstract(profile_id: str) -> str | None:
    """从 Bing 学术 profile 页获取完整摘要."""
    try:
        session = _make_session()
        try:
            url = f"https://cn.bing.com/academic/profile?id={profile_id}&encoded=0&v=paper_preview"
            resp = session.get(url, timeout=get_timeout())
            if resp.status_code != 200:
                return None
            soup = BeautifulSoup(resp.text, "lxml")
            content = soup.select_one("div.acapp_abstract_content")
            if content:
                return content.get_text(" ", strip=True)
        finally:
            session.close()
    except Exception as e:
        logger.debug("Bing profile abstract fetch failed: %s", e)
    return None
