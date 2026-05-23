"""analysis.py 测试 — 论文相关性分析."""

import json
import pytest
from scholar_search.analysis import (
    analyze_relevance,
    _tokenize,
    _paper_text,
    _parse_papers,
    _generate_cluster_hint,
    _extract_top_keywords,
)


# ---- _tokenize ----


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Graph Neural Networks for Recommendation")
        assert "graph" in tokens
        assert "neural" in tokens
        assert "networks" in tokens
        assert "recommendation" in tokens
        assert "for" not in tokens

    def test_short_words_filtered(self):
        tokens = _tokenize("a an in on at to be")
        assert tokens == []

    def test_numbers_filtered(self):
        tokens = _tokenize("the 123 4567 test")
        assert "test" in tokens
        assert "123" not in tokens
        assert "4567" not in tokens

    def test_mixed_case(self):
        tokens = _tokenize("Graph Neural NETWORKS")
        assert "graph" in tokens
        assert "neural" in tokens
        assert "networks" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []


# ---- _paper_text ----


class TestPaperText:
    def test_title_and_abstract(self):
        p = {"title": "GNN for RS", "abstract": "We propose a novel method."}
        result = _paper_text(p)
        assert result == "GNN for RS We propose a novel method."

    def test_title_only(self):
        p = {"title": "GNN for RS", "abstract": ""}
        result = _paper_text(p)
        assert result == "GNN for RS"

    def test_abstract_only(self):
        p = {"title": "", "abstract": "Some abstract here."}
        result = _paper_text(p)
        assert result == "Some abstract here."

    def test_both_empty(self):
        p = {"title": "", "abstract": ""}
        result = _paper_text(p)
        assert result == ""

    def test_stripped(self):
        p = {"title": "  Title  ", "abstract": "  Abstract  "}
        result = _paper_text(p)
        assert result == "Title Abstract"


# ---- _parse_papers ----


class TestParsePapers:
    def test_json_string_list(self):
        papers = [{"title": "A"}, {"title": "B"}]
        result = _parse_papers(json.dumps(papers))
        assert len(result) == 2

    def test_json_string_wrapped_ranked_papers(self):
        papers = [{"title": "A"}]
        data = {"ranked_papers": papers}
        result = _parse_papers(json.dumps(data))
        assert len(result) == 1
        assert result[0]["title"] == "A"

    def test_json_string_wrapped_papers(self):
        papers = [{"title": "B"}]
        data = {"papers": papers}
        result = _parse_papers(json.dumps(data))
        assert len(result) == 1
        assert result[0]["title"] == "B"

    def test_json_string_wrapped_results(self):
        papers = [{"title": "C"}]
        data = {"results": papers}
        result = _parse_papers(json.dumps(data))
        assert len(result) == 1

    def test_list_input(self):
        papers = [{"title": "X"}]
        result = _parse_papers(papers)
        assert len(result) == 1

    def test_single_dict_input(self):
        data = {"title": "Single"}
        result = _parse_papers(data)
        assert len(result) == 1
        assert result[0]["title"] == "Single"

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            _parse_papers("not a list or dict")

    def test_empty_list(self):
        result = _parse_papers([])
        assert result == []

    def test_valid_json_not_list_or_dict(self):
        """json.loads 成功但结果不是 dict 也不是 list — line 153."""
        with pytest.raises(ValueError, match="论文列表"):
            _parse_papers("123")

    def test_valid_json_boolean(self):
        with pytest.raises(ValueError, match="论文列表"):
            _parse_papers("true")


# ---- analyze_relevance ----


class TestAnalyzeRelevance:
    def sample_papers(self):
        return [
            {
                "title": "Graph Neural Networks for Recommendation",
                "abstract": "We propose a GNN-based collaborative filtering method.",
                "year": 2023,
            },
            {
                "title": "Attention Mechanisms in NLP",
                "abstract": "This paper surveys attention mechanisms in transformers.",
                "year": 2022,
            },
            {
                "title": "Deep Learning for Recommender Systems",
                "abstract": "A survey of deep learning for recommendation including GNNs.",
                "year": 2021,
            },
        ]

    def test_basic_flow(self):
        papers = self.sample_papers()
        topic = "graph neural networks for recommendation systems"
        result = analyze_relevance(topic, json.dumps(papers))
        assert "ranked_papers" in result
        assert "summary" in result
        assert len(result["ranked_papers"]) == 3
        top_paper = result["ranked_papers"][0]
        assert "relevance_score" in top_paper
        assert top_paper["title"] == "Graph Neural Networks for Recommendation"
        assert "avg_score" in result["summary"]
        assert "top_keywords" in result["summary"]
        assert "cluster_hint" in result["summary"]

    def test_empty_topic_raises(self):
        with pytest.raises(ValueError, match="topic 不能为空"):
            analyze_relevance("   ", json.dumps([{"title": "X"}]))

    def test_empty_papers_raises(self):
        with pytest.raises(ValueError, match="papers_json 不包含有效论文数据"):
            analyze_relevance("topic", json.dumps([]))

    def test_papers_without_abstracts(self):
        papers = [
            {"title": "GNN Recommendation", "abstract": ""},
            {"title": "NLP Transformers", "abstract": ""},
        ]
        result = analyze_relevance("graph neural networks", json.dumps(papers))
        assert len(result["ranked_papers"]) == 2
        assert result["ranked_papers"][0]["title"] == "GNN Recommendation"

    def test_scores_are_sorted(self):
        papers = self.sample_papers()
        result = analyze_relevance("recommendation systems", json.dumps(papers))
        scores = [p["relevance_score"] for p in result["ranked_papers"]]
        assert scores == sorted(scores, reverse=True)

    def test_list_input_accepted(self):
        papers = self.sample_papers()
        result = analyze_relevance("test", papers)
        assert len(result["ranked_papers"]) == 3

    def test_wrapped_input_accepted(self):
        papers = self.sample_papers()
        result = analyze_relevance("test", json.dumps({"papers": papers}))
        assert len(result["ranked_papers"]) == 3

    def test_identical_papers_same_score(self):
        papers = [
            {"title": "Same Title", "abstract": "Same abstract here for testing."},
            {"title": "Same Title", "abstract": "Same abstract here for testing."},
        ]
        result = analyze_relevance("some topic", json.dumps(papers))
        assert result["ranked_papers"][0]["relevance_score"] == result["ranked_papers"][1]["relevance_score"]

    def test_single_paper(self):
        papers = [{"title": "Only Paper", "abstract": "Testing single paper analysis."}]
        result = analyze_relevance("testing analysis", json.dumps(papers))
        assert len(result["ranked_papers"]) == 1
        assert isinstance(result["summary"]["avg_score"], float)

    def test_degenerate_vocabulary(self):
        """空词汇触发 TfidfVectorizer ValueError → 退化处理 (lines 93-104)."""
        # 全部 1-2 字母单词 → tokenizer 返回 [] → 空词汇 → 退路也失败 → 零分结果
        papers = [{"title": "a b", "abstract": "c d"}, {"title": "e f", "abstract": "g h"}]
        result = analyze_relevance("x y", json.dumps(papers))
        assert len(result["ranked_papers"]) == 2
        assert result["summary"]["avg_score"] == 0.0
        assert result["ranked_papers"][0]["relevance_score"] == 0.0

    def test_all_papers_empty_content(self):
        """所有论文都无内容时触发 title 退化 (line 81)."""
        papers = [
            {"title": "", "abstract": ""},
            {"title": "   ", "abstract": "   "},
        ]
        result = analyze_relevance("test topic here", json.dumps(papers))
        assert len(result["ranked_papers"]) == 2


# ---- _generate_cluster_hint ----


class TestGenerateClusterHint:
    def test_all_high_relevance(self):
        papers = [
            {"relevance_score": 80.0, "year": 2023},
            {"relevance_score": 70.0, "year": 2022},
        ]
        hint = _generate_cluster_hint(papers, 75.0)
        assert "高度相关" in hint

    def test_all_low_under_10(self):
        papers = [
            {"relevance_score": 5.0, "year": 2019},
            {"relevance_score": 3.0, "year": 2018},
            {"relevance_score": 7.0, "year": 2020},
        ]
        hint = _generate_cluster_hint(papers, 5.0)
        assert "关联度较低" in hint or "扩大搜索" in hint

    def test_mixed_relevance(self):
        papers = [
            {"relevance_score": 80.0, "year": 2023},
            {"relevance_score": 35.0, "year": 2022},
            {"relevance_score": 5.0, "year": 2021},
        ]
        hint = _generate_cluster_hint(papers, 40.0)
        assert "高度相关" in hint
        assert "中度相关" in hint

    def test_mid_only(self):
        papers = [
            {"relevance_score": 30.0, "year": 2022},
            {"relevance_score": 25.0, "year": 2021},
        ]
        hint = _generate_cluster_hint(papers, 27.5)
        assert "中度相关" in hint

    def test_no_year_info(self):
        papers = [
            {"relevance_score": 80.0},
            {"relevance_score": 70.0},
        ]
        hint = _generate_cluster_hint(papers, 75.0)
        assert "高度相关" in hint

    def test_returns_string(self):
        hint = _generate_cluster_hint([], 0.0)
        assert isinstance(hint, str)

    def test_all_low_relevance_hint(self):
        """所有论文低相关性且 average >= 10 — line 199."""
        papers = [
            {"relevance_score": 5.0, "year": 2020},
            {"relevance_score": 15.0, "year": 2021},
            {"relevance_score": 10.0, "year": 2022},
        ]
        hint = _generate_cluster_hint(papers, 10.0)
        assert "均较低" in hint


# ---- _extract_top_keywords ----


class TestExtractTopKeywords:
    def test_returns_keywords(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [
            "graph neural networks for recommendation",
            "deep learning for graph analysis",
        ]
        vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None)
        tfidf = vectorizer.fit_transform(texts)

        keywords = _extract_top_keywords(texts, vectorizer, tfidf, top_n=3)
        assert len(keywords) <= 3
        assert all(isinstance(kw, str) for kw in keywords)
