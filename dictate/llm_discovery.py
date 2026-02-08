"""LLM endpoint discovery for local models."""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "localhost:11434"
API_TIMEOUT_SECONDS = 5


@dataclass
class DiscoveredModel:
    """Result of LLM endpoint discovery."""

    name: str
    endpoint: str
    is_available: bool


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint to host:port format."""
    endpoint = endpoint.strip()
    # Remove protocol prefix if present
    if endpoint.startswith("http://"):
        endpoint = endpoint[7:]
    elif endpoint.startswith("https://"):
        endpoint = endpoint[8:]
    # Remove trailing slash and path
    endpoint = endpoint.split("/")[0]
    return endpoint


def _try_openai_models(endpoint: str) -> DiscoveredModel | None:
    """Try to get model info from OpenAI-compatible /v1/models endpoint."""
    try:
        url = f"http://{endpoint}/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())

        # Extract model name from response
        if isinstance(data, dict) and "data" in data:
            models = data["data"]
            if models and len(models) > 0:
                # Get first model's id
                model_name = models[0].get("id", "")
                if model_name:
                    return DiscoveredModel(
                        name=model_name,
                        endpoint=endpoint,
                        is_available=True,
                    )
        elif isinstance(data, list) and len(data) > 0:
            # Some servers return a plain list
            model_name = data[0].get("id", "")
            if model_name:
                return DiscoveredModel(
                    name=model_name,
                    endpoint=endpoint,
                    is_available=True,
                )
    except Exception:
        pass
    return None


def _try_ollama_tags(endpoint: str) -> DiscoveredModel | None:
    """Try to get model info from Ollama /api/tags endpoint."""
    try:
        url = f"http://{endpoint}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())

        # Ollama returns { "models": [ { "name": "...", ... }, ... ] }
        if isinstance(data, dict) and "models" in data:
            models = data["models"]
            if models and len(models) > 0:
                # Get first model name
                model_name = models[0].get("name", "")
                if model_name:
                    return DiscoveredModel(
                        name=model_name,
                        endpoint=endpoint,
                        is_available=True,
                    )
    except Exception:
        pass
    return None


def discover_llm(endpoint: str | None = None) -> DiscoveredModel:
    """Discover what LLM is running at the given endpoint.

    Tries OpenAI-compatible /v1/models first, then Ollama /api/tags.

    Args:
        endpoint: The endpoint to check (host:port). Defaults to localhost:11434.

    Returns:
        DiscoveredModel with name and availability status.
    """
    if endpoint is None:
        endpoint = DEFAULT_ENDPOINT

    endpoint = _normalize_endpoint(endpoint)

    # Try OpenAI-compatible endpoint first (works for LM Studio, vLLM, etc.)
    result = _try_openai_models(endpoint)
    if result:
        logger.info("Discovered OpenAI-compatible model: %s", result.name)
        return result

    # Try Ollama native endpoint
    result = _try_ollama_tags(endpoint)
    if result:
        logger.info("Discovered Ollama model: %s", result.name)
        return result

    # Nothing found
    logger.debug("No LLM found at %s", endpoint)
    return DiscoveredModel(
        name="",
        endpoint=endpoint,
        is_available=False,
    )


def get_display_name(endpoint: str | None = None) -> str:
    """Get a human-readable display name for the LLM at the endpoint.

    Returns:
        String like "qwen3-coder:30b via localhost:11434" or
        "No local model found" if nothing is running.
    """
    result = discover_llm(endpoint)

    if not result.is_available:
        return "No local model found"

    return f"{result.name} via {result.endpoint}"
