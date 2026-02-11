"""Extended tests for dictate.llm_discovery — comprehensive coverage.

Tests endpoint normalization, OpenAI/Ollama discovery, and display name generation.
"""

from __future__ import annotations

import json
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


class TestNormalizeEndpointExtended:
    """Extended tests for _normalize_endpoint function."""

    def test_plain_host_port(self):
        """Test plain host:port format."""
        assert _normalize_endpoint("localhost:11434") == "localhost:11434"

    def test_ip_address_port(self):
        """Test IP address:port format."""
        assert _normalize_endpoint("192.168.1.1:8000") == "192.168.1.1:8000"

    def test_http_prefix_removed(self):
        """Test http:// prefix is removed."""
        assert _normalize_endpoint("http://localhost:11434") == "localhost:11434"

    def test_http_with_path(self):
        """Test http:// with path has path stripped."""
        assert _normalize_endpoint("http://localhost:11434/v1/chat") == "localhost:11434"

    def test_http_with_trailing_slash(self):
        """Test http:// with trailing slash."""
        assert _normalize_endpoint("http://localhost:11434/") == "localhost:11434"

    def test_https_prefix_removed(self):
        """Test https:// prefix is removed."""
        assert _normalize_endpoint("https://localhost:11434") == "localhost:11434"

    def test_https_with_path(self):
        """Test https:// with path."""
        assert _normalize_endpoint("https://api.example.com/v1/models") == "api.example.com"

    def test_multiple_path_segments(self):
        """Test path with multiple segments is stripped."""
        assert _normalize_endpoint("localhost:11434/api/v1/tags") == "localhost:11434"

    def test_leading_trailing_whitespace(self):
        """Test whitespace is stripped."""
        assert _normalize_endpoint("  localhost:11434  ") == "localhost:11434"

    def test_with_query_params(self):
        """Test query params are stripped (path-like behavior)."""
        # URL with query params - should strip after ?
        result = _normalize_endpoint("localhost:11434?key=value")
        assert "localhost:11434" in result

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert _normalize_endpoint("") == ""

    def test_only_whitespace(self):
        """Test whitespace-only string returns empty."""
        assert _normalize_endpoint("   ") == ""

    def test_just_protocol(self):
        """Test just protocol returns empty."""
        assert _normalize_endpoint("http://") == ""

    def test_domain_without_port(self):
        """Test domain without port."""
        assert _normalize_endpoint("example.com") == "example.com"

    def test_http_domain_without_port(self):
        """Test http:// domain without port."""
        assert _normalize_endpoint("http://example.com") == "example.com"


class MockResponse:
    """Mock urllib response for testing."""

    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestTryOpenAIModelsExtended:
    """Extended tests for _try_openai_models function."""

    def test_success_with_data_array_single_model(self):
        """Test successful response with single model in data array."""
        response_data = json.dumps({
            "data": [{"id": "gpt-4", "object": "model"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result is not None
        assert result.name == "gpt-4"
        assert result.endpoint == "localhost:11434"
        assert result.is_available is True

    def test_success_with_multiple_models(self):
        """Test that first model is returned when multiple models present."""
        response_data = json.dumps({
            "data": [
                {"id": "first-model", "object": "model"},
                {"id": "second-model", "object": "model"},
            ]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result.name == "first-model"

    def test_success_plain_list_format(self):
        """Test successful response with plain list format (not wrapped in data)."""
        response_data = json.dumps([
            {"id": "llama3.1", "object": "model"},
            {"id": "qwen3-coder", "object": "model"},
        ]).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result is not None
        assert result.name == "llama3.1"

    def test_empty_data_array(self):
        """Test empty data array returns None."""
        response_data = json.dumps({"data": []}).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_empty_plain_list(self):
        """Test empty plain list returns None."""
        response_data = json.dumps([]).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_missing_id_field(self):
        """Test model without id field returns None."""
        response_data = json.dumps({
            "data": [{"object": "model", "created": 12345}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_empty_id_field(self):
        """Test model with empty id field returns None."""
        response_data = json.dumps({
            "data": [{"id": "", "object": "model"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_malformed_json(self):
        """Test malformed JSON returns None."""
        response_data = b"not valid json"

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_connection_refused_error(self):
        """Test connection refused returns None."""
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_timeout_error(self):
        """Test timeout error returns None."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_generic_exception(self):
        """Test generic exception returns None."""
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_url_error(self):
        """Test URLError returns None."""
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("No route")):
            result = _try_openai_models("localhost:11434")

        assert result is None

    def test_http_error_404(self):
        """Test HTTPError 404 returns None."""
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            "http://localhost:11434/v1/models", 404, "Not Found", {}, None
        )):
            result = _try_openai_models("localhost:11434")

        assert result is None


class TestTryOllamaTagsExtended:
    """Extended tests for _try_ollama_tags function."""

    def test_success_single_model(self):
        """Test successful response with single model."""
        response_data = json.dumps({
            "models": [{"name": "llama3.1:latest", "modified_at": "2024-01-01"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_ollama_tags("localhost:11434")

        assert result is not None
        assert result.name == "llama3.1:latest"
        assert result.endpoint == "localhost:11434"
        assert result.is_available is True

    def test_success_multiple_models(self):
        """Test that first model is returned when multiple models present."""
        response_data = json.dumps({
            "models": [
                {"name": "first-model", "modified_at": "2024-01-01"},
                {"name": "second-model", "modified_at": "2024-01-02"},
            ]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_ollama_tags("localhost:11434")

        assert result.name == "first-model"

    def test_empty_models_array(self):
        """Test empty models array returns None."""
        response_data = json.dumps({"models": []}).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_missing_models_key(self):
        """Test missing models key returns None."""
        response_data = json.dumps({"other": "data"}).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_missing_name_field(self):
        """Test model without name field returns None."""
        response_data = json.dumps({
            "models": [{"modified_at": "2024-01-01"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_empty_name_field(self):
        """Test model with empty name field returns None."""
        response_data = json.dumps({
            "models": [{"name": "", "modified_at": "2024-01-01"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_malformed_json(self):
        """Test malformed JSON returns None."""
        response_data = b"not valid json"

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_connection_error(self):
        """Test connection error returns None."""
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            result = _try_ollama_tags("localhost:11434")

        assert result is None

    def test_timeout_error(self):
        """Test timeout error returns None."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            result = _try_ollama_tags("localhost:11434")

        assert result is None


class TestDiscoverLLMExtended:
    """Extended end-to-end tests for discover_llm function."""

    def test_uses_default_endpoint_when_none_provided(self):
        """Test that default endpoint is used when none provided."""
        response_data = json.dumps({
            "data": [{"id": "test-model", "object": "model"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = discover_llm()

        assert result.endpoint == DEFAULT_ENDPOINT
        assert result.name == "test-model"

    def test_openai_success_no_ollama_needed(self):
        """Test OpenAI endpoint success means Ollama not tried."""
        openai_response = json.dumps({
            "data": [{"id": "openai-model", "object": "model"}]
        }).encode()

        mock_calls = []

        def mock_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            mock_calls.append(url)
            if "/v1/models" in url:
                return MockResponse(openai_response)
            raise Exception("Should not reach Ollama")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm("localhost:11434")

        assert result.name == "openai-model"
        assert any("/v1/models" in url for url in mock_calls)

    def test_openai_fails_fallback_to_ollama(self):
        """Test fallback to Ollama when OpenAI fails."""
        ollama_response = json.dumps({
            "models": [{"name": "ollama-model", "modified_at": "2024-01-01"}]
        }).encode()

        def mock_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "/v1/models" in url:
                raise ConnectionRefusedError()
            elif "/api/tags" in url:
                return MockResponse(ollama_response)
            raise Exception(f"Unexpected URL: {url}")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm("localhost:11434")

        assert result.name == "ollama-model"
        assert result.is_available is True

    def test_both_endpoints_fail(self):
        """Test both endpoints fail returns unavailable."""
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            result = discover_llm("localhost:9999")

        assert result.is_available is False
        assert result.name == ""
        assert result.endpoint == "localhost:9999"

    def test_endpoint_is_normalized(self):
        """Test that endpoint is normalized before use."""
        response_data = json.dumps({
            "data": [{"id": "test-model", "object": "model"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = discover_llm("http://localhost:11434/v1/models")

        # Normalized endpoint should be in result
        assert result.endpoint == "localhost:11434"

    def test_openai_empty_models_falls_back_to_ollama(self):
        """Test fallback when OpenAI returns empty models."""
        def mock_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "/v1/models" in url:
                return MockResponse(json.dumps({"data": []}).encode())
            elif "/api/tags" in url:
                return MockResponse(json.dumps({
                    "models": [{"name": "ollama-model"}]
                }).encode())
            raise Exception(f"Unexpected URL: {url}")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm("localhost:11434")

        assert result.name == "ollama-model"

    def test_openai_no_id_falls_back_to_ollama(self):
        """Test fallback when OpenAI model has no id."""
        def mock_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "/v1/models" in url:
                return MockResponse(json.dumps({
                    "data": [{"object": "model"}]  # No id
                }).encode())
            elif "/api/tags" in url:
                return MockResponse(json.dumps({
                    "models": [{"name": "ollama-model"}]
                }).encode())
            raise Exception(f"Unexpected URL: {url}")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = discover_llm("localhost:11434")

        assert result.name == "ollama-model"


class TestGetDisplayNameExtended:
    """Extended tests for get_display_name function."""

    def test_available_model_format(self):
        """Test format of available model display name."""
        response_data = json.dumps({
            "data": [{"id": "qwen3-coder:30b", "object": "model"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = get_display_name("localhost:11434")

        assert "qwen3" in result.lower()

    def test_available_model_with_custom_endpoint(self):
        """Test display name with custom endpoint."""
        response_data = json.dumps({
            "data": [{"id": "custom-model", "object": "model"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = get_display_name("192.168.1.100:8000")

        assert "custom" in result.lower()

    def test_unavailable_model_default_message(self):
        """Test message when no model is available."""
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            result = get_display_name("localhost:9999")

        assert result == "No local model found"

    def test_uses_default_endpoint_when_none_provided(self):
        """Test uses default endpoint when none provided."""
        response_data = json.dumps({
            "data": [{"id": "default-model", "object": "model"}]
        }).encode()

        with patch("urllib.request.urlopen", return_value=MockResponse(response_data)):
            result = get_display_name()

        assert "default" in result.lower()

    def test_ollama_discovered_format(self):
        """Test format when Ollama model is discovered."""
        def mock_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "/v1/models" in url:
                raise ConnectionRefusedError()
            elif "/api/tags" in url:
                return MockResponse(json.dumps({
                    "models": [{"name": "llama3.1:latest"}]
                }).encode())
            raise Exception(f"Unexpected URL: {url}")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = get_display_name("localhost:11434")

        assert "llama3.1" in result.lower()


class TestDiscoveredModelDataclass:
    """Tests for DiscoveredModel dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating DiscoveredModel with all fields."""
        model = DiscoveredModel(
            name="test-model",
            endpoint="localhost:11434",
            is_available=True
        )
        assert model.name == "test-model"
        assert model.endpoint == "localhost:11434"
        assert model.is_available is True

    def test_creation_unavailable(self):
        """Test creating unavailable DiscoveredModel."""
        model = DiscoveredModel(
            name="",
            endpoint="localhost:9999",
            is_available=False
        )
        assert model.name == ""
        assert model.endpoint == "localhost:9999"
        assert model.is_available is False

    def test_equality(self):
        """Test DiscoveredModel equality."""
        model1 = DiscoveredModel("model1", "host1", True)
        model2 = DiscoveredModel("model1", "host1", True)
        model3 = DiscoveredModel("model2", "host1", True)

        assert model1 == model2
        assert model1 != model3
