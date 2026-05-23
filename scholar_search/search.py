"""谷歌学术搜索封装 — 基于 requests + BeautifulSoup 直接解析."""

import os
import random
import re
import xml.etree.ElementTree as ET
import time
import logging
import concurrent.futures

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from .config import get_proxy, get_timeout, get_retries

logger = logging.getLogger(__name__)

# Google Scholar 搜索基 URL
_SEARCH_URL = "https://scholar.google.com/scholar"
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

_proxy_url = get_proxy()
_proxies = {"http": _proxy_url, "https": _proxy_url} if _proxy_url else None
if _proxies:
    logger.info("代理已配置: %s", _proxy_url)
    os.environ.setdefault("HTTP_PROXY", _proxy_url)
    os.environ.setdefault("HTTPS_PROXY", _proxy_url)
else:
    logger.info("未配置代理，将直连谷歌学术")


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _make_session() -> requests.Session:
    s = requests.Session()

    # SSL 偶发断连自动重试：total=3 次，退避 0.5s，覆盖 SSLError
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    s.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": _random_ua(),
    })
    if _proxies:
        s.proxies.update(_proxies)
    return s


def _jitter(base: float, factor: float = 0.5) -> float:
    return base * (1 + random.uniform(-factor, factor))


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("403", "429", "rate", "captcha", "blocked"))


def _run_with_timeout(func, *args, timeout: int | None = None):
    t = timeout or get_timeout()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        return future.result(timeout=t)


def _pub_to_dict(paper_soup: BeautifulSoup) -> dict:
    """从单个论文的 HTML 片段解析字段."""
    # ---- 标题 ----
    title_tag = paper_soup.select_one("h3.gs_rt")
    raw_title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # 移除 [PDF][HTML][BOOK][CITATION] 等格式标签和前缀
    raw_title = re.sub(r"\[(PDF|HTML|BOOK|B|C|CITATION)\]", " ", raw_title, flags=re.I)
    raw_title = re.sub(r"\s+", " ", raw_title).strip()

    # 若标题形如 "Author, Author, ... - Real Title"，提取 Real Title 部分
    title = raw_title
    if " - " in raw_title and "," in raw_title.split(" - ")[0]:
        parts = raw_title.split(" - ", 1)
        prefix = parts[0]
        suffix = parts[1]
        # 前缀中逗号占比高 → 很可能是作者列表
        comma_ratio = prefix.count(",") / max(len(prefix), 1)
        if comma_ratio > 0.02 or len(prefix) < len(suffix):
            title = suffix.strip().strip("–").strip()

    url = ""
    if title_tag:
        a = title_tag.find("a")
        if a and a.get("href"):
            url = a["href"]

    # ---- 作者 ----
    authors_tag = paper_soup.select_one("div.gs_a")
    authors_text = authors_tag.get_text(" ", strip=True) if authors_tag else ""

    # 移除 URL 中的作者
    authors_text = re.sub(r"\s+", " ", authors_text)
    # gs_a 格式: "Author1, Author2, ... - Venue Year - Publisher"
    author_part = authors_text.split(" - ")[0] if " - " in authors_text else authors_text
    # 过滤掉看起来不像人名的 token（年份、URL片段等）
    authors = []
    for chunk in author_part.split(","):
        chunk = chunk.strip()
        if chunk and not re.match(r"^\d{4}$", chunk) and len(chunk) > 1:
            authors.append(chunk)

    # ---- 年份 ----
    year = ""
    year_match = re.search(r"\b(19|20)\d{2}\b", authors_text)
    if year_match:
        year = year_match.group(0)

    # ---- 摘要 ----
    # 优先取 gs_fma_abs（展开面板中的完整摘要），否则用 gs_rs（截断版）
    abstract = ""
    fma_abs = paper_soup.select_one("div.gs_fma_abs")
    if fma_abs:
        snp = fma_abs.select_one("div.gs_fma_snp")
        if snp:
            abstract = snp.get_text(" ", strip=True)
    if not abstract:
        abstract_tag = paper_soup.select_one("div.gs_rs")
        abstract = abstract_tag.get_text(" ", strip=True) if abstract_tag else ""

    # ---- 引用量 ----
    citations = 0
    for a_tag in paper_soup.select("a"):
        a_text = a_tag.get_text(" ", strip=True)
        if "Cited by" in a_text:
            match = re.search(r"(\d+)", a_text)
            if match:
                citations = int(match.group(1))
                break

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": "",
        "citations": citations,
        "abstract": abstract,
        "url": url,
        "publisher": "",
        "author_ids": [],
    }


def _parse_search_page(html: str) -> list[dict]:
    """解析 Google Scholar 搜索结果页面."""
    soup = BeautifulSoup(html, "lxml")
    papers = []
    for result in soup.select("div.gs_r.gs_or.gs_scl"):
        papers.append(_pub_to_dict(result))
    return papers


def _search_request(query: str, start: int = 0, year_low: int | None = None,
                    year_high: int | None = None) -> list[dict]:
    """执行单次搜索请求."""
    params = {
        "q": query,
        "hl": "en",
        "as_sdt": "0,33",
        "start": start,
    }
    if year_low:
        params["as_ylo"] = year_low
    if year_high:
        params["as_yhi"] = year_high

    session = _make_session()
    try:
        resp = session.get(_SEARCH_URL, params=params, timeout=get_timeout())
        if resp.status_code == 200:
            _check_captcha(resp.text)
            return _parse_search_page(resp.text)
        elif resp.status_code in (403, 429):
            raise RuntimeError(f"谷歌学术限流 (HTTP {resp.status_code})")
        else:
            raise RuntimeError(f"谷歌学术返回 HTTP {resp.status_code}")
    finally:
        session.close()


def _deduplicate_papers(papers: list[dict]) -> list[dict]:
    """精确标题去重，保留先出现的版本."""
    seen_titles = set()
    result = []
    for p in papers:
        key = p.get("title", "").strip().lower()
        # 规范化：移除末尾的 [PDF][HTML] 等标签
        key = re.sub(r"\s*\[[A-Z]+\]\s*$", "", key).strip()
        if key and key not in seen_titles:
            seen_titles.add(key)
            result.append(p)
        elif not key:
            result.append(p)
    return result


def _check_captcha(html: str) -> None:
    """检测验证码页面."""
    if "sorry" in html.lower() and "captcha" in html.lower():
        raise RuntimeError("谷歌学术要求验证码，请稍后再试或更换代理IP")


def search_papers(
    query: str,
    num_results: int = 10,
    year_low: int | None = None,
    year_high: int | None = None,
) -> list[dict]:
    """搜索谷歌学术论文，返回标准化论文列表."""
    num_results = max(1, min(num_results, 30))

    def _do_search():
        results = []
        start = 0

        # 预搜索延迟
        time.sleep(_jitter(0.8))

        while len(results) < num_results:
            for attempt in range(get_retries()):
                try:
                    page = _search_request(query, start, year_low, year_high)
                    for paper in page:
                        if len(results) >= num_results:
                            break
                        results.append(paper)
                    break
                except RuntimeError as e:
                    is_rate = _is_rate_limit_error(e)
                    wait = _jitter(30, 0.6) if is_rate else (2 ** attempt)
                    wait = min(wait, 120)
                    if attempt >= get_retries() - 1:
                        raise RuntimeError(
                            f"谷歌学术搜索失败（已重试 {get_retries()} 次）: {e}"
                        ) from e
                    logger.warning(
                        "搜索 '%s' %s，%ds 后重试 (%d/%d)",
                        query[:60], "被限流" if is_rate else "失败", int(wait), attempt + 1, get_retries(),
                    )
                    time.sleep(wait)

            if not page:
                break
            start += 10
            # 页间间隔
            time.sleep(_jitter(2.0, 0.4))

        return _deduplicate_papers(results)

    return _run_with_timeout(_do_search)


def get_paper_detail(title: str) -> dict | None:
    """根据标题搜索单篇论文，尝试从外部源获取完整摘要."""

    def _do_fetch():
        time.sleep(_jitter(0.5))
        for attempt in range(get_retries()):
            try:
                papers = _search_request(title)
                if papers:
                    paper = papers[0]
                    # 尝试从外部 URL 获取完整摘要
                    url = paper.get("url", "")
                    full_abstract = _fetch_external_abstract(url)
                    if full_abstract:
                        paper["abstract"] = full_abstract
                    return paper
                return None
            except RuntimeError as e:
                is_rate = _is_rate_limit_error(e)
                wait = _jitter(20, 0.5) if is_rate else (2 ** attempt)
                wait = min(wait, 90)
                if attempt >= get_retries() - 1:
                    raise RuntimeError(f"获取论文详情失败: {e}") from e
                time.sleep(wait)
        return None

    return _run_with_timeout(_do_fetch)


def _fetch_external_abstract(paper_url: str) -> str | None:
    """尝试从论文外部来源获取完整摘要.

    策略:
    1. arxiv URL → arxiv API (结构化 XML, 可靠)
    2. 其他 URL → 请求页面, 从 meta 标签提取
    3. 失败返回 None (调用方回退到 Google Scholar 片段)
    """
    if not paper_url:
        return None

    # ---- arxiv ----
    arxiv_match = re.search(r'arxiv\.org/abs/([\w.-]+)', paper_url)
    if arxiv_match:
        return _fetch_arxiv_abstract(arxiv_match.group(1))

    # ---- 通用 meta 标签 ----
    return _fetch_meta_abstract(paper_url)


def _fetch_arxiv_abstract(arxiv_id: str) -> str | None:
    """通过 arxiv API 获取完整摘要."""
    try:
        session = _make_session()
        try:
            resp = session.get(
                f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1",
                timeout=get_timeout(),
            )
            if resp.status_code != 200:
                return None
            ns = {"a": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            summary = root.find(".//a:summary", ns)
            if summary is not None and summary.text:
                return summary.text.strip().replace("\n", " ")
        finally:
            session.close()
    except Exception as e:
        logger.debug("arxiv abstract fetch failed: %s", e)
    return None


def _fetch_meta_abstract(paper_url: str) -> str | None:
    """从页面 meta 标签提取摘要 (citation_abstract > description > og:description)."""
    try:
        session = _make_session()
        try:
            resp = session.get(paper_url, timeout=get_timeout())
            if resp.status_code != 200:
                return None

            # 跳过非 HTML 响应 (PDF 等)
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" in content_type.lower() or "application/" in content_type.lower():
                return None

            soup = BeautifulSoup(resp.text, "lxml")
            for sel in [
                'meta[name="citation_abstract"]',
                'meta[name="description"]',
                'meta[name="dc.description"]',
                'meta[name="abstract"]',
                'meta[property="og:description"]',
            ]:
                tag = soup.select_one(sel)
                if tag and tag.get("content"):
                    text = tag["content"].strip()
                    if len(text) > 80:  # 过滤太短的描述
                        return text
        finally:
            session.close()
    except Exception as e:
        logger.debug("meta abstract fetch failed: %s", e)
    return None


def search_by_url(paper_url: str) -> dict | None:
    """通过 Google Scholar URL 获取论文信息."""

    def _do_fetch():
        time.sleep(_jitter(0.5))
        for attempt in range(get_retries()):
            try:
                session = _make_session()
                try:
                    resp = session.get(paper_url, timeout=get_timeout())
                    if resp.status_code == 200:
                        _check_captcha(resp.text)
                        results = _parse_search_page(resp.text)
                        return results[0] if results else None
                    elif resp.status_code in (403, 429):
                        raise RuntimeError(f"谷歌学术限流 (HTTP {resp.status_code})")
                finally:
                    session.close()
            except RuntimeError as e:
                is_rate = _is_rate_limit_error(e)
                wait = _jitter(20, 0.5) if is_rate else (2 ** attempt)
                wait = min(wait, 90)
                if attempt >= get_retries() - 1:
                    raise RuntimeError(f"通过 URL 获取论文失败: {e}") from e
                time.sleep(wait)
        return None

    return _run_with_timeout(_do_fetch)
