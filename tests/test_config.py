"""config.py 测试 — 代理配置、超时、重试参数读取."""

import os
from scholar_search import config

# 保存原始环境变量，避免 search.py 模块加载时 setdefault 的干扰
_ENV_KEYS = [
    "SCHOLAR_PROXY", "SCHOLAR_NO_PROXY", "SCHOLAR_TIMEOUT", "SCHOLAR_RETRIES",
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
]


def _clear_proxy_env():
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


class TestGetProxy:
    def test_scholar_proxy_env(self):
        _clear_proxy_env()
        os.environ["SCHOLAR_PROXY"] = "http://custom:9999"
        try:
            assert config.get_proxy() == "http://custom:9999"
        finally:
            _clear_proxy_env()

    def test_http_proxy_env(self):
        _clear_proxy_env()
        os.environ["HTTP_PROXY"] = "http://http_proxy:8080"
        try:
            assert config.get_proxy() == "http://http_proxy:8080"
        finally:
            _clear_proxy_env()

    def test_https_proxy_env(self):
        _clear_proxy_env()
        os.environ["HTTPS_PROXY"] = "https://https_proxy:443"
        try:
            assert config.get_proxy() == "https://https_proxy:443"
        finally:
            _clear_proxy_env()

    def test_lowercase_http_proxy(self):
        _clear_proxy_env()
        os.environ["http_proxy"] = "http://lowercase:3128"
        try:
            assert config.get_proxy() == "http://lowercase:3128"
        finally:
            _clear_proxy_env()

    def test_lowercase_https_proxy(self):
        _clear_proxy_env()
        os.environ["https_proxy"] = "https://lowercase_https:8443"
        try:
            assert config.get_proxy() == "https://lowercase_https:8443"
        finally:
            _clear_proxy_env()

    def test_https_over_http_priority(self):
        """HTTP_PROXY 和 HTTPS_PROXY 同时设置时，返回 HTTPS_PROXY."""
        _clear_proxy_env()
        os.environ["HTTP_PROXY"] = "http://http:80"
        os.environ["HTTPS_PROXY"] = "https://https:443"
        try:
            assert config.get_proxy() == "https://https:443"
        finally:
            _clear_proxy_env()

    def test_no_proxy_explicit(self):
        _clear_proxy_env()
        os.environ["SCHOLAR_NO_PROXY"] = "1"
        try:
            assert config.get_proxy() is None
        finally:
            _clear_proxy_env()

    def test_no_proxy_true(self):
        _clear_proxy_env()
        os.environ["SCHOLAR_NO_PROXY"] = "true"
        try:
            assert config.get_proxy() is None
        finally:
            _clear_proxy_env()

    def test_no_proxy_yes(self):
        _clear_proxy_env()
        os.environ["SCHOLAR_NO_PROXY"] = "yes"
        try:
            assert config.get_proxy() is None
        finally:
            _clear_proxy_env()

    def test_default_fallback(self):
        _clear_proxy_env()
        assert config.get_proxy() == config.DEFAULT_PROXY

    def test_no_proxy_false_value_uses_default(self):
        """SCHOLAR_NO_PROXY=0 不是禁用信号，应走默认值."""
        _clear_proxy_env()
        os.environ["SCHOLAR_NO_PROXY"] = "0"
        try:
            assert config.get_proxy() == config.DEFAULT_PROXY
        finally:
            _clear_proxy_env()


class TestGetTimeout:
    def test_default(self):
        os.environ.pop("SCHOLAR_TIMEOUT", None)
        assert config.get_timeout() == config.DEFAULT_TIMEOUT

    def test_custom(self):
        os.environ["SCHOLAR_TIMEOUT"] = "60"
        try:
            assert config.get_timeout() == 60
        finally:
            os.environ.pop("SCHOLAR_TIMEOUT", None)


class TestGetRetries:
    def test_default(self):
        os.environ.pop("SCHOLAR_RETRIES", None)
        assert config.get_retries() == config.DEFAULT_RETRIES

    def test_custom(self):
        os.environ["SCHOLAR_RETRIES"] = "5"
        try:
            assert config.get_retries() == 5
        finally:
            os.environ.pop("SCHOLAR_RETRIES", None)


class TestGetChartPort:
    def test_default(self):
        os.environ.pop("SCHOLAR_CHART_PORT", None)
        assert config.get_chart_port() == 8765

    def test_custom(self):
        os.environ["SCHOLAR_CHART_PORT"] = "9999"
        try:
            assert config.get_chart_port() == 9999
        finally:
            os.environ.pop("SCHOLAR_CHART_PORT", None)
