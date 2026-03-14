"""
Generic adapter: wraps any duck-typed tool with .name, .description, and .run(**kwargs) or .invoke(**kwargs).
All evaluation goes through governor._evaluate_before_call with framework GENERIC.
"""
import time
from typing import Any, Optional

from ..models import ErrGovernanceBlock


class GenericWrappedTool:
    """Wraps any object with name, description, and run or invoke method."""

    def __init__(self, tool: Any, governor: Any) -> None:
        self._tool = tool
        self._governor = governor

    @property
    def name(self) -> str:
        return getattr(self._tool, "name", "unknown")

    @property
    def description(self) -> str:
        return getattr(self._tool, "description", "")

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        ok, block_reason = self._governor._evaluate_before_call(
            tool_name=self.name,
            tool_description=self.description,
            parameters=kwargs,
            framework_override="GENERIC",
        )
        if not ok:
            raise ErrGovernanceBlock(block_reason)
        start = time.perf_counter()
        try:
            fn = getattr(self._tool, "invoke", None) or getattr(self._tool, "run", None)
            if fn is None:
                raise AttributeError("Tool must have .invoke or .run method")
            if callable(fn):
                result = fn(**kwargs) if not args else fn(*args, **kwargs)
            else:
                raise AttributeError("Tool must have .invoke or .run callable")
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            self._governor._tracer.record_tool_result(
                self.name,
                str(result),
                execution_time_ms=execution_time_ms,
                success=True,
                sequence_number=self._governor._tracer.next_sequence(),
            )
            # Emit tool-result audit event if kyraEventId is available.
            try:
                self._governor._emit_tool_result(
                    tool_name=self.name,
                    execution_time_ms=execution_time_ms,
                    success=True,
                    error_message=None,
                )
            except Exception:
                pass
            return result
        except Exception:
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            self._governor._tracer.record_tool_result(
                self.name,
                "",
                execution_time_ms=execution_time_ms,
                success=False,
                sequence_number=self._governor._tracer.next_sequence(),
            )
            try:
                self._governor._emit_tool_result(
                    tool_name=self.name,
                    execution_time_ms=execution_time_ms,
                    success=False,
                    error_message="",
                )
            except Exception:
                pass
            raise
