# Patches httpx globally within an agent run to auto-inject X-Kyra-Trace headers.
# Call activate() once after creating GovernanceContext for the session.
# This propagates trace context to any HTTP calls the agent makes downstream.

import httpx
from typing import Optional
from ..governance_context import get_context


_original_send = httpx.Client.send
_original_async_send = httpx.AsyncClient.send
_active = False


def activate():
    """
    Monkey-patch httpx to inject Kyra governance headers into all outgoing requests.
    Call once per agent session after setting GovernanceContext.
    """
    global _active
    if _active:
        return
    _active = True

    def patched_send(self, request, **kwargs):
        _inject_headers(request)
        return _original_send(self, request, **kwargs)

    async def patched_async_send(self, request, **kwargs):
        _inject_headers(request)
        return await _original_async_send(self, request, **kwargs)

    httpx.Client.send = patched_send
    httpx.AsyncClient.send = patched_async_send


def deactivate():
    """Restore original httpx behavior."""
    global _active
    if not _active:
        return
    _active = False
    httpx.Client.send = _original_send
    httpx.AsyncClient.send = _original_async_send


def _inject_headers(request: httpx.Request):
    ctx = get_context()
    if ctx is None:
        return
    headers = ctx.to_headers()
    for k, v in headers.items():
        request.headers[k] = v
