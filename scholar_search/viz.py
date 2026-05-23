"""相关性图表生成 — matplotlib 渲染多角度图表 + 轻量 HTTP 服务端."""

import io
import json
import logging
import socketserver
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from .config import get_chart_port

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

logger = logging.getLogger(__name__)

_SERVER: HTTPServer | None = None
_SERVER_LOCK = threading.Lock()

# 图表缓存 (endpoint -> bytes)
_charts: dict[str, bytes] = {}
_index_html: str = ""


def _find_font() -> str | None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ---- 图表生成 ----


def _bar_chart(papers: list[dict], topic: str) -> bytes:
    """相关性水平柱状图."""
    papers = papers[:20]
    titles = [_truncate(p.get("title", "Untitled"), 55) for p in papers]
    scores = [float(p.get("relevance_score", 0)) for p in papers]
    colors = ["#27ae60" if s >= 50 else "#f39c12" if s >= 20 else "#e74c3c" for s in scores]
    font = _find_font() or "sans-serif"

    fig, ax = plt.subplots(figsize=(12, max(4, len(papers) * 0.42)))
    y = range(len(papers))
    bars = ax.barh(y, scores, color=colors, edgecolor="white", height=0.6)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", fontsize=8, fontfamily=font)
    ax.set_yticks(y)
    ax.set_yticklabels(titles, fontsize=8, fontfamily=font)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Relevance Score (0-100)", fontfamily=font)
    title_text = f"Relevance: {topic}" if topic else "Paper Relevance"
    ax.set_title(title_text, fontweight="bold", fontfamily=font)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = [Patch(facecolor="#27ae60", label="High (>=50)"),
              Patch(facecolor="#f39c12", label="Medium (20-49)"),
              Patch(facecolor="#e74c3c", label="Low (<20)")]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _cluster_chart(papers: list[dict], topic: str) -> bytes:
    """K-Means 聚类散点图 (TF-IDF → KMeans → PCA 2D)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    texts = [(p.get("title") or "") + " " + (p.get("abstract") or "") for p in papers]
    texts = [t.strip() or "no content" for t in texts]

    vec = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf = vec.fit_transform(texts)

    n_clusters = min(3, max(2, len(papers)))
    if tfidf.shape[0] < 3 or tfidf.shape[1] < 2:
        return _empty_chart("聚类分析", "论文数/特征数不足，无法聚类")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf)

    if tfidf.shape[1] >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(tfidf.toarray())
    else:
        return _empty_chart("聚类分析", "特征维度不足，无法降维")

    font = _find_font() or "sans-serif"
    palette = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, (x, y) in enumerate(coords):
        c = palette[labels[i] % len(palette)]
        ax.scatter(x, y, c=c, s=120, edgecolors="white", linewidth=0.8, zorder=3)
        short = _truncate(papers[i].get("title", ""), 30)
        ax.annotate(short, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, fontfamily=font, alpha=0.85)

    ax.set_title(f"K-Means Clustering ({n_clusters} groups): {topic}" if topic else "K-Means Clustering",
                 fontweight="bold", fontfamily=font)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontfamily=font)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontfamily=font)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _keyword_chart(papers: list[dict], topic: str) -> bytes:
    """TF-IDF 关键词重要性柱状图."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [(p.get("title") or "") + " " + (p.get("abstract") or "") for p in papers]
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return _empty_chart("关键词分析", "无有效文本")

    vec = TfidfVectorizer(max_features=1000, stop_words="english",
                          token_pattern=r"[a-zA-Z]{3,}")
    tfidf = vec.fit_transform(texts)
    summed = tfidf.sum(axis=0).A1
    top_n = min(15, len(summed))
    if top_n == 0:
        return _empty_chart("关键词分析", "无法提取关键词")
    idx = summed.argsort()[-top_n:][::-1]
    words = [vec.get_feature_names_out()[i] for i in idx]
    values = [float(summed[i]) for i in idx]

    font = _find_font() or "sans-serif"
    colors = ["#3498db"] * top_n
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    y = range(top_n)
    ax.barh(y, values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(words, fontsize=10, fontfamily=font)
    ax.invert_yaxis()
    ax.set_xlabel("TF-IDF Sum", fontfamily=font)
    title_text = f"Top Keywords: {topic}" if topic else "Top Keywords"
    ax.set_title(title_text, fontweight="bold", fontfamily=font)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _empty_chart(title: str, reason: str) -> bytes:
    """生成「无法生成」提示图."""
    font = _find_font() or "sans-serif"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, f"{title}\n{reason}", ha="center", va="center",
            fontsize=14, fontfamily=font, color="#999", transform=ax.transAxes)
    ax.set_axis_off()
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---- HTTP 服务端 ----


class _ChartHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.debug("HTTP %s", fmt % args)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path in _charts:
            self._serve_png(_charts[self.path])
        else:
            self.send_error(404)

    def _serve_html(self):
        body = _index_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_png(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _start_server(port: int) -> None:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            try:
                _SERVER.shutdown()
            except Exception:
                pass
            _SERVER = None
        _SERVER = _ThreadedServer(("127.0.0.1", port), _ChartHandler)
        t = threading.Thread(target=_SERVER.serve_forever, daemon=True)
        t.start()


# ---- 主入口 ----


def generate_relevance_chart(papers_json: str, topic: str = "") -> str:
    """生成多角度相关性图表并启动本地 HTTP 服务.

    Args:
        topic: 研究主题
        papers_json: analyze_relevance 返回的 JSON 字符串

    Returns:
        JSON 含 url / message / charts 列表
    """
    papers = _parse_input(papers_json)
    if not papers:
        raise ValueError("papers_json 不包含有效的论文数据")

    port = get_chart_port()
    base = f"http://localhost:{port}"

    global _charts, _index_html
    _charts = {
        "/bar.png": _bar_chart(papers, topic),
        "/cluster.png": _cluster_chart(papers, topic),
        "/keywords.png": _keyword_chart(papers, topic),
    }

    chart_links = "\n".join(
        f"- [{name.lstrip('/')}]({base}/{name.lstrip('/')})"
        for name in ["/bar.png", "/cluster.png", "/keywords.png"]
    )
    _index_html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>Scholar Relevance: {_escape(topic) or 'Analysis'}</title>
<style>
 body {{ font-family: 'Segoe UI','Microsoft YaHei',sans-serif;
       max-width:1000px; margin:20px auto; padding:0 20px; background:#fafafa }}
 h1 {{ font-size:18px; border-bottom:2px solid #3498db; padding-bottom:8px }}
 .chart {{ margin:24px 0; text-align:center }}
 .chart img {{ max-width:100%; border:1px solid #ddd; border-radius:4px; box-shadow:0 2px 8px rgba(0,0,0,.08) }}
 .chart p {{ color:#666; font-size:14px; margin-top:6px }}
</style></head><body>
<h1>Relevance Analysis: {_escape(topic) or 'Papers'}</h1>
<div class="chart"><p>Relevance Scores</p>
<img src="{base}/bar.png" alt="bar chart"></div>
<div class="chart"><p>K-Means Clustering</p>
<img src="{base}/cluster.png" alt="cluster"></div>
<div class="chart"><p>Top Keywords</p>
<img src="{base}/keywords.png" alt="keywords"></div>
</body></html>"""

    _start_server(port)

    return json.dumps({
        "url": base + "/",
        "port": port,
        "charts": [
            {"name": "相关性柱状图", "url": base + "/bar.png"},
            {"name": "K-Means 聚类", "url": base + "/cluster.png"},
            {"name": "关键词分析", "url": base + "/keywords.png"},
        ],
        "message": (
            f"## 相关性分析图表\n\n"
            f"[打开图表面板]({base}/)\n\n"
            f"如链接无法点击，请复制以下地址在浏览器中打开：\n"
            f"`{base}/`\n\n"
            f"### 单独图表链接\n"
            f"{chart_links}"
        ),
    }, ensure_ascii=False)


def _parse_input(papers_json: str) -> list[dict]:
    if isinstance(papers_json, str):
        try:
            data = json.loads(papers_json)
        except json.JSONDecodeError:
            raise ValueError("无法从输入中提取论文列表")
    else:
        data = papers_json
    if isinstance(data, dict):
        for key in ("ranked_papers", "papers", "results"):
            if key in data:
                return data[key]
    if isinstance(data, list):
        return data
    raise ValueError("无法从输入中提取论文列表")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
