from __future__ import annotations

import threading
from typing import Optional

from .models import AgentContext


_agent_ctx_local = threading.local()


def set_agent_context(ctx: Optional[AgentContext]) -> None:
    """
    Set the current AgentContext for this thread.

    When present, the AgentContext is attached to ActionRequest as
    `agentContext` and allows the server to emit AGENT_DECISION events
    without parsing LLM output.
    """
    _agent_ctx_local.ctx = ctx


def get_agent_context() -> Optional[AgentContext]:
    return getattr(_agent_ctx_local, "ctx", None)

