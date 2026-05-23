# Scholar Search MCP

基于 Python MCP (Model Context Protocol) 的谷歌学术搜索工具，供 CherryStudio 等 AI 客户端调用。

## 功能

| Tool | 说明 |
|------|------|
| `search_papers` | 谷歌学术论文搜索，支持年份过滤、自动精确去重、SSL 断连重试 |
| `get_paper_detail` | 获取单篇论文详细信息（摘要、引用量） |
| `analyze_relevance` | TF-IDF + 余弦相似度相关性排序，关键词提取，方向聚类摘要 |
| `generate_relevance_chart` | Matplotlib 多角度图表（柱状图 / K-Means 聚类 / 关键词）+ 本地 HTTP 服务端 |

## 环境要求

- Python >= 3.10
- conda 环境 `MCP`（或任意虚拟环境）
- 谷歌学术需 HTTP 代理（Clash / V2Ray 等）

## 安装

```bash
conda activate MCP
pip install -r requirements.txt
# 或
pip install -e .
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SCHOLAR_PROXY` | `http://localhost:7890` | 代理地址，优先级最高 |
| `HTTP_PROXY` / `HTTPS_PROXY` | - | 标准代理环境变量（备选） |
| `SCHOLAR_NO_PROXY` | - | 设为 `1`/`true`/`yes` 禁用代理 |
| `SCHOLAR_TIMEOUT` | `30` | 单次 HTTP 请求超时（秒） |
| `SCHOLAR_RETRIES` | `3` | 搜索失败最大重试次数 |
| `SCHOLAR_CHART_PORT` | `8765` | 图表 HTTP 服务端端口 |

### 代理配置

三层优先级，从高到低：

```
SCHOLAR_PROXY  >  HTTP_PROXY / HTTPS_PROXY  >  默认 http://localhost:7890
```

## CherryStudio 接入

### 1. MCP 配置

```json
{
  "scholar-search": {
    "command": "C:/Users/mulim/.conda/envs/MCP/python.exe",
    "args": ["C:/Users/mulim/Desktop/Project/scholar_search/server.py"],
    "env": {
      "SCHOLAR_PROXY": "http://localhost:7890",
      "SCHOLAR_CHART_PORT": "8765"
    }
  }
}
```

> 路径需替换为实际路径。macOS/Linux 用户去掉盘符，使用 Unix 路径风格。

### 2. 在对话中使用

CherryStudio 对话时，直接描述你的研究需求即可，AI 会自动调用工具链：

**示例**：
> "搜索 2020 年后关于 graph neural network for recommendation system 的论文，取前 10 篇，做相关性分析并生成图表"

**典型调用链**：
```
search_papers → analyze_relevance → generate_relevance_chart → 浏览器打开 http://localhost:8765
```

### 3. 图表查看

`generate_relevance_chart` 会启动本地 HTTP 服务并返回链接：
```
http://localhost:8765/
```

包含三个图表：
- **相关性柱状图** `/bar.png`
- **K-Means 聚类散点图** `/cluster.png`
- **关键词重要性图** `/keywords.png`

在浏览器中打开后不会自动刷新，重新调用工具即可更新数据。

## 开发

```bash
# 运行测试（默认跳过网络 mock 测试，~7s）
pytest

# 包含网络 mock 测试（~40s）
pytest tests/ --ignore=

# 带 coverage.xml 输出
pytest --cov=scholar_search --cov=server --cov-report=term-missing --cov-report=xml

# Lint 检查 (PEP 8, max-line=127)
flake8 --max-line-length=127 .

# 启动调试
python server.py
```

## 项目结构

```
scholar_search/
├── server.py              # MCP Server 入口 (FastMCP, 4 个 Tool)
├── scholar_search/        # 核心包
│   ├── config.py          # 代理 / 超时 / 重试 / 端口配置
│   ├── search.py          # requests + BeautifulSoup 直连解析
│   ├── analysis.py        # TF-IDF + 余弦相似度 + 方向聚类
│   └── viz.py             # Matplotlib 多图表 + HTTP 服务端
├── tests/                 # pytest (93+25 tests)
│   ├── test_config.py
│   ├── test_search.py
│   ├── test_analysis.py
│   ├── test_viz.py
│   └── test_server.py
├── requirements.txt
├── pyproject.toml
├── CLAUDE.md
└── .gitignore
```

## License

MIT
