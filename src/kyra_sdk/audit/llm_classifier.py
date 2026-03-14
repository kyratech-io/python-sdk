from __future__ import annotations

import json
from typing import Iterable, List, Optional


# Default provider host patterns used for LLM call detection.
LLM_PROVIDER_PATTERNS: List[str] = [
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.com",
    "api.mistral.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.fireworks.ai",
    "inference.ai.azure.com",          # Azure OpenAI
    "bedrock-runtime.",                # AWS Bedrock (bedrock-runtime.*.amazonaws.com)
]


def _iter_patterns(additional: Optional[Iterable[str]] = None) -> List[str]:
    patterns = list(LLM_PROVIDER_PATTERNS)
    if additional:
        patterns.extend(additional)
    return patterns


def is_llm_call(url: str, additional: Optional[Iterable[str]] = None) -> bool:
    """
    Return True if the given URL looks like an LLM provider endpoint.

    Detection is a simple substring match against known provider host
    patterns plus any additional endpoints supplied by the caller.
    """
    if not url:
        return False
    url_lower = url.lower()
    for pattern in _iter_patterns(additional):
        if pattern.lower() in url_lower:
            return True
    return False


def extract_model_from_request(body: bytes | str | None) -> str:
    """
    Extract the `model` field from a JSON request body, if present.

    This is the only field we parse SDK-side; all other parsing is
    delegated to the server.
    """
    if body is None:
        return "unknown"
    try:
        if isinstance(body, bytes):
            text = body.decode("utf-8", errors="ignore")
        else:
            text = body
        if not text:
            return "unknown"
        data = json.loads(text)
        if isinstance(data, dict):
            model = data.get("model")
            if isinstance(model, str) and model:
                return model
    except Exception:
        return "unknown"
    return "unknown"

