"""配置管理模块 — 代理、超时、重试、图表端口等参数的读取与默认值."""

import os

DEFAULT_PROXY = "http://localhost:7890"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3


def get_proxy() -> str | None:
    """获取代理地址，三层优先级：
    1. SCHOLAR_PROXY 环境变量
    2. HTTP_PROXY / HTTPS_PROXY 环境变量
    3. 默认值 DEFAULT_PROXY
    返回 None 表示不使用代理。
    """
    proxy = os.environ.get("SCHOLAR_PROXY")
    if proxy:
        return proxy

    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http_proxy or https_proxy:
        return https_proxy or http_proxy

    # 检查是否显式禁用代理
    if os.environ.get("SCHOLAR_NO_PROXY", "").lower() in ("1", "true", "yes"):
        return None

    return DEFAULT_PROXY


def get_timeout() -> int:
    return int(os.environ.get("SCHOLAR_TIMEOUT", str(DEFAULT_TIMEOUT)))


def get_retries() -> int:
    return int(os.environ.get("SCHOLAR_RETRIES", str(DEFAULT_RETRIES)))


def get_chart_port() -> int:
    return int(os.environ.get("SCHOLAR_CHART_PORT", "8765"))
