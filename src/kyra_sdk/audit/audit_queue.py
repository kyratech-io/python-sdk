from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Optional

import httpx


class AuditQueue:
    """
    Fire-and-forget audit queue.

    A single background worker drains an in-memory queue and POSTs payloads
    to Kyra's audit endpoints. Errors are swallowed and items are never
    retried to avoid impacting the agent execution path.
    """

    def __init__(self, server_url: str, maxsize: int = 5000) -> None:
        self._server_url = server_url.rstrip("/")
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=maxsize)
        self._client = httpx.Client(timeout=3.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue_llm_raw(self, payload: Dict[str, Any]) -> None:
        """
        Enqueue a payload for POST /api/v1/audit/llm-raw.
        """
        payload = dict(payload)
        payload["_endpoint"] = "/api/v1/audit/llm-raw"
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            # Drop silently — never block agent execution.
            pass

    def enqueue_memory_event(self, payload: Dict[str, Any]) -> None:
        """
        Enqueue a payload for POST /api/v1/audit/memory-event.
        """
        payload = dict(payload)
        payload["_endpoint"] = "/api/v1/audit/memory-event"
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            pass

    def enqueue_session_event(self, payload: Dict[str, Any]) -> None:
        """
        Enqueue a payload for POST /api/v1/audit/session-event.
        """
        payload = dict(payload)
        payload["_endpoint"] = "/api/v1/audit/session-event"
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            pass

    def enqueue_tool_result(self, payload: Dict[str, Any]) -> None:
        """
        Enqueue a payload for POST /api/v1/audit/tool-result.
        """
        payload = dict(payload)
        payload["_endpoint"] = "/api/v1/audit/tool-result"
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                endpoint = item.pop("_endpoint", None)
                if not endpoint:
                    continue
                self._post(endpoint, item)
            except Exception:
                # Swallow all errors — audit failures must not affect agents.
                continue

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> None:
        try:
            self._client.post(f"{self._server_url}{endpoint}", json=payload)
        except Exception:
            # Intentionally swallow all errors; no retries.
            pass


_default_queue: Optional[AuditQueue] = None
_default_server_url: str = "https://api.kyratech.io"
_lock = threading.Lock()


def configure(server_url: str) -> None:
    """
    Configure the base server URL used for audit POSTs.

    This should be called once from KyraGovernor so that audit traffic
    is routed to the same Kyra deployment as evaluate().
    """
    global _default_server_url
    _default_server_url = server_url.rstrip("/") if server_url else _default_server_url


def get_audit_queue() -> AuditQueue:
    """
    Return the process-wide AuditQueue instance, creating it lazily if needed.
    """
    global _default_queue
    if _default_queue is not None:
        return _default_queue
    with _lock:
        if _default_queue is None:
            _default_queue = AuditQueue(_default_server_url)
        return _default_queue

