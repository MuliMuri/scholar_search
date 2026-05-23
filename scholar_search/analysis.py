"""论文相关性分析 — 基于 TF-IDF 向量化和余弦相似度，对搜索结果按主题相关性排序."""

import json
import logging
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# 常见停用词（学术场景补充）
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "we", "you", "they",
    "this", "that", "these", "those", "it", "its", "our", "their", "his",
    "her", "he", "she", "them", "us", "no", "not", "nor", "so", "as",
    "if", "than", "then", "also", "very", "too", "just", "about", "into",
    "over", "such", "only", "other", "new", "most", "more", "some", "each",
    "both", "between", "after", "before", "through", "during", "because",
    "based", "using", "used", "show", "found", "results", "one", "two",
    "first", "approach", "method", "methods", "paper", "study", "research",
    "et", "al", "ieee", "acm", "springer", "arxiv", "proposed", "present",
    "problem", "problems", "work", "well", "many", "however", "without",
    "different", "several", "yet", "still", "thus", "since", "often",
    "important", "provide", "need", "way",
}


def _tokenize(text: str) -> list[str]:
    """简单分词并滤除停用词."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _paper_text(p: dict) -> str:
    """拼接论文的标题与摘要作为分析文本."""
    parts = []
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)
    return " ".join(parts)


def analyze_relevance(topic: str, papers_json: str) -> dict:
    """对论文列表进行主题相关性分析.

    Args:
        topic: 研究主题描述（英文，1-3 句话）
        papers_json: 论文列表的 JSON 字符串，每篇需含 title 和 abstract

    Returns:
        {
            "ranked_papers": [...],   # 按相关性降序排列，每篇附加 relevance_score
            "summary": {
                "avg_score": float,   # 平均相关性分
                "top_keywords": [...], # 所有论文中的热门关键词
                "cluster_hint": str,   # 研究方向聚类提示
            }
        }
    """
    if not topic.strip():
        raise ValueError("topic 不能为空")

    papers = _parse_papers(papers_json)
    if not papers:
        raise ValueError("papers_json 不包含有效论文数据")

    # 构建 TF-IDF 语料
    corpus = [topic] + [_paper_text(p) for p in papers]
    has_content = any(len(text.strip()) > 0 for text in corpus[1:])

    if not has_content:
        # 所有论文都缺少摘要时，仅用标题做相似度
        corpus = [topic] + [(p.get("title") or "") for p in papers]

    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize,
        token_pattern=None,
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2),
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # 词汇量过少时的退化处理 — 回退到默认 tokenizer
        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=1000,
            ngram_range=(1, 1),
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # 完全无法提取任何词汇时，返回零分
            return _empty_relevance_result(papers)

    topic_vec = tfidf_matrix[0:1]
    paper_vecs = tfidf_matrix[1:]

    similarities = cosine_similarity(topic_vec, paper_vecs).flatten()
    scores = (similarities * 100).round(1)

    # 按分数排序
    ranked = []
    for i, paper in enumerate(papers):
        paper["relevance_score"] = float(scores[i])
        ranked.append(paper)

    ranked.sort(key=lambda x: x["relevance_score"], reverse=True)

    # 摘要分析
    all_keywords = _extract_top_keywords(
        corpus[1:], vectorizer, tfidf_matrix[1:], top_n=15
    )

    avg_score = round(float(scores.mean()), 1)
    cluster_hint = _generate_cluster_hint(ranked, avg_score)

    return {
        "ranked_papers": ranked,
        "summary": {
            "avg_score": avg_score,
            "top_keywords": all_keywords,
            "cluster_hint": cluster_hint,
        },
    }


def _empty_relevance_result(papers: list[dict]) -> dict:
    """无法提取词汇时返回的零分结果."""
    for p in papers:
        p["relevance_score"] = 0.0
    return {
        "ranked_papers": papers,
        "summary": {
            "avg_score": 0.0,
            "top_keywords": [],
            "cluster_hint": "无法从论文文本中提取有效关键词，请检查论文数据是否包含英文标题或摘要",
        },
    }


def _parse_papers(papers_json: str) -> list[dict]:
    """解析论文 JSON，兼容字符串和列表."""
    if isinstance(papers_json, str):
        data = json.loads(papers_json)
    else:
        data = papers_json

    if isinstance(data, dict):
        # 可能是 {"ranked_papers": [...]} 或 {"papers": [...]} 包装
        for key in ("ranked_papers", "papers", "results"):
            if key in data:
                data = data[key]
                break
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError("papers_json 需为论文列表或包含论文列表的对象")

    return data


def _extract_top_keywords(
    texts: list[str],
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 15,
) -> list[str]:
    """从论文语料中提取 TF-IDF 权重最高的关键词."""
    feature_names = vectorizer.get_feature_names_out()
    summed = tfidf_matrix.sum(axis=0).A1  # 各词在所有论文中的 TF-IDF 总和
    top_indices = summed.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if summed[i] > 0]


def _generate_cluster_hint(ranked_papers: list[dict], avg_score: float) -> str:
    """根据得分分布生成研究方向聚类提示."""
    high_count = sum(1 for p in ranked_papers if p["relevance_score"] >= 50)
    mid_count = sum(1 for p in ranked_papers if 20 <= p["relevance_score"] < 50)
    low_count = len(ranked_papers) - high_count - mid_count

    if high_count == 0 and avg_score < 10:
        return (
            "搜索结果与主题关联度较低，可能该方向研究较少或搜索关键词不够精确。"
            "建议尝试同义词或更广泛的关键词重新搜索。"
        )

    hints = []
    if high_count > 0:
        years = [p.get("year") for p in ranked_papers[:high_count] if p.get("year")]
        if years:
            y_min, y_max = min(years), max(years)
            hints.append(
                f"发现 {high_count} 篇高度相关论文（得分 >= 50），"
                f"时间跨度 {y_min}-{y_max}"
            )
        else:
            hints.append(f"发现 {high_count} 篇高度相关论文（得分 >= 50）")

    if mid_count > 0:
        hints.append(f"{mid_count} 篇中度相关论文（20-49 分）可作参考")

    if low_count > 0 and low_count == len(ranked_papers):
        hints.append("所有论文相关性均较低，建议扩大搜索范围或调整关键词")

    return "；".join(hints) if hints else "相关性分析完成"
