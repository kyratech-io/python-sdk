"""
HTTP interceptor for propagating Kyra governance headers and capturing audit
telemetry for LLM and memory calls.

Call ``activate()`` once per agent session after setting a ``GovernanceContext``.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import httpx

from ..governance_context import get_context
from ..audit import get_audit_queue, is_llm_call, extract_model_from_request, pii_strip, extract_user_id

try:  # optional dependencies
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore


_original_send = httpx.Client.send
_original_async_send = httpx.AsyncClient.send
_requests_original_send = getattr(requests, "Session.send", None) if requests else None
_aiohttp_original_request = getattr(aiohttp.ClientSession, "_request", None) if aiohttp else None
_active = False

_LLM_ENDPOINTS: List[str] = []
_MEMORY_ENDPOINTS: List[str] = []


def configure_endpoints(
    llm_endpoints: Optional[List[str]] = None,
    memory_endpoints: Optional[List[str]] = None,
) -> None:
    """
    Configure additional LLM and memory endpoints for classification.

    Called from ``KyraGovernor`` so that HTTP interception can use the same
    endpoint registry as the governor evaluation path.
    """
    global _LLM_ENDPOINTS, _MEMORY_ENDPOINTS
    _LLM_ENDPOINTS = list(llm_endpoints or [])
    _MEMORY_ENDPOINTS = list(memory_endpoints or [])


def activate() -> None:
    """
    Monkey-patch HTTP clients to inject Kyra governance headers into all
    outgoing requests and capture lightweight audit telemetry.
    """
    global _active
    if _active:
        return
    _active = True

    def patched_send(self, request, **kwargs):
        _inject_headers(request)
        start = time.perf_counter()
        response = _original_send(self, request, **kwargs)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        _classify_and_audit_httpx(request, response, elapsed_ms)
        return response

    async def patched_async_send(self, request, **kwargs):
        _inject_headers(request)
        start = time.perf_counter()
        response = await _original_async_send(self, request, **kwargs)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        _classify_and_audit_httpx(request, response, elapsed_ms)
        return response

    httpx.Client.send = patched_send  # type: ignore[assignment]
    httpx.AsyncClient.send = patched_async_send  # type: ignore[assignment]

    if requests is not None and _requests_original_send is not None:

        def requests_patched_send(self, request, **kwargs):  # type: ignore[override]
            start = time.perf_counter()
            response = _requests_original_send(self, request, **kwargs)  # type: ignore[misc]
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            try:
                url = getattr(request, "url", "")
                _classify_and_audit_generic(
                    url=str(url),
                    method=getattr(request, "method", "GET"),
                    request_body=getattr(request, "body", None),
                    response_body=getattr(response, "content", None),
                    elapsed_ms=elapsed_ms,
                )
            except Exception:
                pass
            return response

        requests.Session.send = requests_patched_send  # type: ignore[assignment]

    if aiohttp is not None and _aiohttp_original_request is not None:

        async def aiohttp_patched_request(self, method, str_or_url, *args, **kwargs):  # type: ignore[override]
            start = time.perf_counter()
            response = await _aiohttp_original_request(self, method, str_or_url, *args, **kwargs)  # type: ignore[misc]
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            try:
                url = str(response.url)
                body = kwargs.get("data") or kwargs.get("json")
                text = await response.text()
                _classify_and_audit_generic(
                    url=url,
                    method=method,
                    request_body=body,
                    response_body=text,
                    elapsed_ms=elapsed_ms,
                )
            except Exception:
                pass
            return response

        aiohttp.ClientSession._request = aiohttp_patched_request  # type: ignore[assignment]


def deactivate() -> None:
    """Restore original HTTP client behavior."""
    global _active
    if not _active:
        return
    _active = False
    httpx.Client.send = _original_send  # type: ignore[assignment]
    httpx.AsyncClient.send = _original_async_send  # type: ignore[assignment]
    if requests is not None and _requests_original_send is not None:
        requests.Session.send = _requests_original_send  # type: ignore[assignment]
    if aiohttp is not None and _aiohttp_original_request is not None:
        aiohttp.ClientSession._request = _aiohttp_original_request  # type: ignore[assignment]


def _inject_headers(request: httpx.Request) -> None:
    ctx = get_context()
    if ctx is None:
        return
    headers = ctx.to_headers()
    for k, v in headers.items():
        request.headers[k] = v


def _is_memory_call(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    for pattern in _MEMORY_ENDPOINTS:
        if pattern.lower() in url_lower:
            return True
    return False


def _memory_event_type(method: str) -> str:
    m = (method or "GET").upper()
    if m == "GET":
        return "MEMORY_READ"
    if m in ("POST", "PUT"):
        return "MEMORY_WRITE"
    if m == "PATCH":
        return "MEMORY_UPDATE"
    if m == "DELETE":
        return "MEMORY_DELETE"
    return "MEMORY_READ"


def _classify_and_audit_httpx(request: httpx.Request, response: httpx.Response, elapsed_ms: int) -> None:
    try:
        url = str(request.url)
        method = request.method
        body_bytes = bytes(request.content or b"")
        resp_bytes = bytes(response.content or b"")
        _classify_and_audit_generic(
            url=url,
            method=method,
            request_body=body_bytes,
            response_body=resp_bytes,
            elapsed_ms=elapsed_ms,
        )
    except Exception:
        # Audit must never interfere with HTTP success/failure.
        pass


def _classify_and_audit_generic(
    url: str,
    method: str,
    request_body: Any,
    response_body: Any,
    elapsed_ms: int,
) -> None:
    ctx = get_context()
    session_id = ctx.session_id if ctx else None
    agent_id = ctx.root_agent_id if ctx else None

    # LLM calls
    if is_llm_call(url, _LLM_ENDPOINTS):
        user_id = extract_user_id(request_body)
        stripped_req = pii_strip(request_body)
        stripped_res = pii_strip(response_body)
        model = extract_model_from_request(request_body)
        payload = {
            "agentId": agent_id,
            "sessionId": session_id,
            "userId": user_id,
            "url": url,
            "requestBody": stripped_req,
            "responseBody": stripped_res,
            "latencyMs": elapsed_ms,
            "model": model,
        }
        try:
            queue = get_audit_queue()
            queue.enqueue_llm_raw(payload)
        except Exception:
            pass

    # Memory calls
    if _is_memory_call(url):
        event_type = _memory_event_type(method)
        payload = {
            "agentId": agent_id,
            "sessionId": session_id,
            "eventType": event_type,
            "sourceEndpoint": url,
            "latencyMs": elapsed_ms,
        }
        try:
            queue = get_audit_queue()
            queue.enqueue_memory_event(payload)
        except Exception:
            pass

