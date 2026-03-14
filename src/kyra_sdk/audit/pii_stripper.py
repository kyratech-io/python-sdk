from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


REDACTED = "[REDACTED]"


def _redact_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    redacted: List[Any] = []
    for msg in messages:
        if not isinstance(msg, dict):
            redacted.append(msg)
            continue
        role = msg.get("role")
        # Copy to avoid mutating caller data if they reuse the same dict
        msg_copy: Dict[str, Any] = dict(msg)
        if role == "user":
            # Only redact the content field; preserve role and other structure.
            if "content" in msg_copy:
                msg_copy["content"] = REDACTED
        redacted.append(msg_copy)
    return redacted


def _redact_top_level(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Redact potentially sensitive free-form text fields.
    if "prompt" in obj:
        obj["prompt"] = REDACTED
    if "input" in obj:
        obj["input"] = REDACTED

    # Preserve identity/structural fields exactly: user, userId, user_id,
    # messages[*].role, tool_calls, model, system, tokens, latency, status codes, etc.
    if "messages" in obj:
        obj["messages"] = _redact_messages(obj["messages"])

    return obj


def extract_user_id(body: bytes | str | None) -> Optional[str]:
    """
    Extract the first non-empty string from user, userId, or user_id in the JSON body.
    Call on raw body before pii_strip so identity is available for audit.
    """
    if body is None:
        return None
    try:
        if isinstance(body, bytes):
            text = body.decode("utf-8", errors="ignore")
        else:
            text = body
        if not text:
            return None
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for field in ("user", "userId", "user_id"):
        val = data.get(field)
        if isinstance(val, str) and val:
            return val
    return None


def pii_strip(body: bytes | str | None) -> Dict[str, Any]:
    """
    Strip PII-like content from an LLM request/response body.

    - Redacts:
      - messages[*].content where role == "user"
      - prompt
      - input
    - Preserves:
      - user, userId, user_id
      - messages[*].role
      - tool_calls
      - model
      - system
      - all token counts, latency values, and status codes

    If JSON parsing fails, returns an empty object {} instead of raw bytes.
    """
    if body is None:
        return {}
    try:
        if isinstance(body, bytes):
            text = body.decode("utf-8", errors="ignore")
        else:
            text = body
        if not text:
            return {}
        data = json.loads(text)
    except Exception:
        return {}

    # Only operate on JSON objects; if the payload is a list or primitive,
    # return {} rather than attempting to infer PII structure.
    if not isinstance(data, dict):
        return {}

    return _redact_top_level(dict(data))

