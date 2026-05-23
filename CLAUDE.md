# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

一个基于 Python MCP (Model Context Protocol) 的谷歌学术搜索工具，供 CherryStudio 等 AI 客户端调用：
- 谷歌学术论文搜索（支持年份过滤、完整摘要提取）
- 论文详情获取（含摘要、引用量）
- 论文主题相关性分析（TF-IDF + 余弦相似度）
- 多角度相关性图表生成（Matplotlib 柱状图/聚类/关键词 + HTTP 本地服务）

## 开发环境

Python 路径: `C:/Users/mulim/.conda/envs/MCP/python.exe` (conda 环境 `MCP`, Python 3.11)

```bash
# 运行命令时指定 Python 路径
"C:/Users/mulim/.conda/envs/MCP/python.exe" ...

# 或用完整路径跑 pip
"C:/Users/mulim/.conda/envs/MCP/python.exe" -m pip install <package>
```

## 常用命令

```bash
# 启动 MCP Server（stdio 模式）
"C:/Users/mulim/.conda/envs/MCP/python.exe" server.py

# 安装/更新依赖
"C:/Users/mulim/.conda/envs/MCP/python.exe" -m pip install -e .

# 运行测试（默认跳过 test_search.py，速度快 ~7s）
"C:/Users/mulim/.conda/envs/MCP/python.exe" -m pytest tests/ --cov=scholar_search --cov=server --cov-report=term-missing --cov-report=xml

# 包含所有测试（含网络 mock 测试，~40s）
"C:/Users/mulim/.conda/envs/MCP/python.exe" -m pytest tests/ --ignore= --cov=scholar_search --cov=server --cov-report=term-missing --cov-report=xml

# Lint 检查
"C:/Users/mulim/.conda/envs/MCP/python.exe" -m flake8 --max-line-length=127 .

# 导入测试
"C:/Users/mulim/.conda/envs/MCP/python.exe" -c "from scholar_search.search import search_papers; from scholar_search.analysis import analyze_relevance"
```

## 架构

```
scholar_search/
├── server.py              # MCP Server 入口，注册 4 个 Tool
├── scholar_search/        # 核心包
│   ├── config.py          # 代理配置（SCHOLAR_PROXY > HTTP_PROXY > 默认 localhost:7890）
│   ├── search.py          # requests + bs4 直连解析，摘要优先 gs_fma_abs → 外部源 (arxiv API / meta 标签)
│   ├── analysis.py        # TF-IDF + 余弦相似度相关性分析，含自定义停用词表
│   └── viz.py             # Matplotlib 多角度图表（柱状图/K-Means 聚类/关键词）+ HTTP 本地服务端
├── tests/                 # pytest 测试 (137 tests, 100% 覆盖)
│   ├── test_config.py
│   ├── test_search.py
│   ├── test_analysis.py
│   ├── test_viz.py
│   └── test_server.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

- **MCP 框架**: 官方 `mcp` SDK v1.27，`FastMCP` + `@mcp.tool()` 装饰器注册，`mcp.run(transport="stdio")` 启动
- **学术搜索**: `requests` + `BeautifulSoup(lxml)` 直接解析 Google Scholar HTML，无第三方依赖
- **代理**: 默认 `http://localhost:7890`，可通过 `SCHOLAR_PROXY` 环境变量覆盖，设 `SCHOLAR_NO_PROXY=1` 禁用
- **搜索节流**: 预搜索 ~0.8s 随机延迟，每条结果 1.5~3s 随机抖动避免固定频率被反爬识别
- **智能重试**: 限流错误 (403/429) 等待 20~60s，普通网络错误指数退避，上限 120s
- **异步封装**: 阻塞调用通过 `asyncio.to_thread()` 在线程池执行，避免阻塞 MCP 事件循环
- **相关性分析**: 自定义英文停用词表 + `TfidfVectorizer(ngram_range=(1,2))` + `cosine_similarity`，得分 0-100
- **图表生成**: Matplotlib (Agg backend) 生成 PNG，内存缓存 + ThreadingHTTPServer 本地服务（默认 127.0.0.1:8765），返回 markdown 链接 + 原始 URL，不自动打开浏览器。含三张图表：相关性柱状图 / K-Means 聚类散点图 / TF-IDF 关键词重要性图

## MCP Tool 清单

| Tool | 参数 | 功能 |
|------|------|------|
| `search_papers` | query, num_results(1-30), year_low?, year_high? | 搜索论文（Google Scholar 摘要片段，快速浏览） |
| `get_paper_detail` | title? or url? | 单篇论文详情，自动从外部源（arxiv API / meta 标签）获取完整摘要 |
| `analyze_relevance` | topic, papers_json | 相关性排序 + 关键词 + 方向聚类摘要 |
| `generate_relevance_chart` | topic?, papers_json | Matplotlib 多角度图表 + HTTP 本地服务端 |

## 代码规范

- **PEP 8**，最大行长 127
- 通过 `flake8 --max-line-length=127 .` 零违规
- 函数无类型注解时应从上下文可推断类型
- 测试覆盖率 100%，新增代码必须有对应测试
