"""Tests for dictate.llm_discovery — LLM endpoint discovery."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from dictate.llm_discovery import (
    DEFAULT_ENDPOINT,
    _normalize_endpoint,
    _try_openai_models,
    _try_ollama_tags,
    discover_llm,
    get_display_name,
    DiscoveredModel,
)


class MockResponse:
    """Mock urllib response."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestNormalizeEndpoint:
    def test_already_normalized(self):
        assert _normalize_endpoint("localhost:11434") == "localhost:11434"
        assert _normalize_endpoint("192.168.1.1:8000") == "192.168.1.1:8000"

    def test_removes_http_prefix(self):
        assert _normalize_endpoint("http://localhost:11434") == "localhost:11434"
        assert _normalize_endpoint("http://localhost:11434/") == "localhost:11434"

    def test_removes_https_prefix(self):
        assert _normalize_endpoint("https://localhost:11434") == "localhost:11434"

    def test_removes_path(self):
        assert _normalize_endpoint("localhost:11434/v1/chat") == "localhost:11434"
        assert _normalize_endpoint("http://localhost:11434/api/tags") == "localhost:11434"

    def test_strips_whitespace(self):
        assert _normalize_endpoint("  localhost:11434  ") == "localhost:11434"


class TestTryOpenAIModels:
    def test_success_with_data_array(self):
        response_data = json.dumps({
            "data": [
                {"id": "qwen3-coder:30b", "object": "model"},
                {"id": "llama3.1", "object": "model"},
            ]
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = _try_openai_models("localhost:11434")

        assert result is not None
        assert result.name == "qwen3-coder:30b"
        assert result.endpoint == "localhost:11434"
        assert result.is_available is True

    def test_success_with_plain_list(self):
        response_data = json.dumps([
            {"id": "gpt-4", "object": "model"},
        ]).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = _try_openai_models("localhost:1234")

        assert result is not None
        assert result.name == "gpt-4"

    def test_empty_models_list(self):
        response_data = json.dumps({"data": []}).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_no_id_field(self):
        response_data = json.dumps({
            "data": [{"object": "model"}]  # No id field
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_connection_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_timeout_error(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            result = _try_openai_models("localhost:11434")

        assert result is None


class TestTryOllamaTags:
    def test_success(self):
        response_data = json.dumps({
            "models": [
                {"name": "qwen3-coder:30b", "modified_at": "2024-01-01"},
                {"name": "llama3.1:latest", "modified_at": "2024-01-02"},
            ]
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = _try_ollama_tags("localhost:11434")

        assert result is not None
        assert result.name == "qwen3-coder:30b"
        assert result.endpoint == "localhost:11434"
        assert result.is_available is True

    def test_empty_models_list(self):
        response_data = json.dumps({"models": []}).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_no_name_field(self):
        response_data = json.dumps({
            "models": [{"modified_at": "2024-01-01"}]  # No name field
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_connection_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = _try_ollama_tags("localhost:11434")

        assert result is None


class TestDiscoverLLM:
    def test_default_endpoint(self):
        response_data = json.dumps({
            "data": [{"id": "test-model", "object": "model"}]
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm()  # No endpoint specified

        assert result.name == "test-model"
        assert result.endpoint == DEFAULT_ENDPOINT

    def test_openai_compatible_found_first(self):
        """OpenAI endpoint should be tried first and return if successful."""
        response_data = json.dumps({
            "data": [{"id": "openai-model", "object": "model"}]
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm("localhost:1234")

        assert result.name == "openai-model"

    def test_fallback_to_ollama(self):
        """If OpenAI endpoint fails, should fallback to Ollama."""
        def mock_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "/v1/models" in url:
                raise Exception("OpenAI endpoint not found")
            elif "/api/tags" in url:
                return MockResponse(json.dumps({
                    "models": [{"name": "ollama-model"}]
                }).encode())
            raise Exception(f"Unexpected URL: {url}")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm("localhost:11434")

        assert result.name == "ollama-model"

    def test_nothing_found(self):
        """If neither endpoint works, return unavailable result."""
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = discover_llm("localhost:9999")

        assert result.is_available is False
        assert result.name == ""
        assert result.endpoint == "localhost:9999"

    def test_endpoint_normalization(self):
        response_data = json.dumps({
            "data": [{"id": "test-model", "object": "model"}]
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm("http://localhost:11434/v1/models")

        # Should normalize and still work
        assert result.name == "test-model"


class TestGetDisplayName:
    def test_discovered_model(self):
        response_data = json.dumps({
            "data": [{"id": "qwen3-coder:30b", "object": "model"}]
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = get_display_name("localhost:11434")

        assert result == "qwen3 coder"

    def test_no_model_found(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = get_display_name("localhost:9999")

        assert result == "No local model found"

    def test_default_endpoint(self):
        response_data = json.dumps({
            "data": [{"id": "test-model", "object": "model"}]
        }).encode()

        def mock_urlopen(req, **kwargs):
            return MockResponse(response_data)

        with patch("urllib.request.urlopen", mock_urlopen):
            result = get_display_name()  # Uses default endpoint

        assert result == "test model"


class TestDiscoveredModel:
    def test_dataclass(self):
        model = DiscoveredModel(name="test", endpoint="localhost:11434", is_available=True)
        assert model.name == "test"
        assert model.endpoint == "localhost:11434"
        assert model.is_available is True

    def test_unavailable(self):
        model = DiscoveredModel(name="", endpoint="localhost:9999", is_available=False)
        assert model.is_available is False
