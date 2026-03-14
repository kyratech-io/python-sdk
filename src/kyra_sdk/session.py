from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from .governor import KyraGovernor
from .governance_context import GovernanceContext, get_context, set_context, clear_context


@contextmanager
def session(governor: KyraGovernor, session_id: str) -> Generator[str, None, None]:
    """
    Session context manager that emits SESSION_STARTED and SESSION_COMPLETED
    audit events around a block of work.

    Example:
        with session(governor, "sess_123") as sid:
            result = my_agent.run(...)
    """
    # JS-style semantics: create a NEW GovernanceContext with session_id set
    # and make it the active context for the duration of this block.
    prev = get_context()
    ctx = GovernanceContext()
    ctx.session_id = session_id
    set_context(ctx)
    # Prevent lazy SESSION_STARTED in governor from double-emitting.
    try:
        if hasattr(governor, "_started_sessions"):
            governor._started_sessions.add(session_id)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        governor._emit_session_event("SESSION_STARTED", session_id)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        yield session_id
    finally:
        try:
            governor._emit_session_event("SESSION_COMPLETED", session_id)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Restore previous context (or clear) after the session finishes.
        try:
            if prev is None:
                clear_context()
            else:
                set_context(prev)
        except Exception:
            pass

