"""批量学术搜索 — 按关键词列表搜索, 去重后输出 JSONL.

用法:
    python scripts/batch_search.py keywords.txt --output papers.jsonl
    python scripts/batch_search.py keywords.txt --engine bing --num-results 20 --year-low 2020

关键词文件每行一个搜索主题:
    一行一个关键词/主题, 英文效果最佳.
    graph neural network recommendation
    transformer attention mechanism survey
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholar_search.search import search_papers as gs_search
from scholar_search.bing_search import search_papers_bing as bing_search


def _dedup_key(p: dict) -> str:
    return p.get("title", "").strip().lower()


def run_batch(
    keywords: list[str],
    engine: str = "bing",
    num_results: int = 20,
    year_low: int | None = None,
    year_high: int | None = None,
    dedup: bool = True,
) -> list[dict]:
    all_papers: list[dict] = []
    seen: set[str] = set()

    search_fn = gs_search if engine == "google" else bing_search

    for i, kw in enumerate(keywords):
        kw = kw.strip()
        if not kw or kw.startswith("#"):
            continue

        print(f"[{i + 1}/{len(keywords)}] 搜索: {kw[:80]}", file=sys.stderr)
        try:
            papers = search_fn(
                query=kw,
                num_results=num_results,
                year_low=year_low,
                year_high=year_high,
            )
        except Exception as e:
            print(f"  ! 失败: {e}", file=sys.stderr)
            continue

        new_count = 0
        for p in papers:
            if not dedup:
                all_papers.append(p)
                new_count += 1
                continue
            key = _dedup_key(p)
            if key and key not in seen:
                seen.add(key)
                all_papers.append(p)
                new_count += 1

        print(f"  -> {len(papers)} 条, 去重后新增 {new_count}, 累计 {len(all_papers)}", file=sys.stderr)

        if i < len(keywords) - 1:
            delay = 1.5 + random.uniform(0, 1.5)
            time.sleep(delay)

    return all_papers


def main() -> None:
    parser = argparse.ArgumentParser(description="批量学术搜索, 输出 JSONL")
    parser.add_argument("keywords_file", help="关键词文件, 每行一个搜索主题")
    parser.add_argument("--output", "-o", default="papers.jsonl", help="输出 JSONL 文件 (默认 papers.jsonl)")
    parser.add_argument("--engine", default="bing", choices=["bing", "google"], help="搜索引擎 (默认 bing)")
    parser.add_argument("--num-results", type=int, default=20, help="每个关键词搜索条数 (默认 20)")
    parser.add_argument("--year-low", type=int, help="发表年份下限")
    parser.add_argument("--year-high", type=int, help="发表年份上限")
    parser.add_argument("--no-dedup", action="store_true", help="不去重")
    args = parser.parse_args()

    with open(args.keywords_file, "r", encoding="utf-8") as f:
        keywords = [line for line in f]

    if not keywords:
        print("关键词文件为空", file=sys.stderr)
        sys.exit(1)

    papers = run_batch(
        keywords=keywords,
        engine=args.engine,
        num_results=args.num_results,
        year_low=args.year_low,
        year_high=args.year_high,
        dedup=not args.no_dedup,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"完成: {len(papers)} 篇论文 -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
