"""
LangChain callback that captures LLM reasoning and tool results for governance.
Add to AgentExecutor: AgentExecutor(..., callbacks=[KyraLangChainCallback(governor)])
"""
import time
from typing import Any, Dict, List, Optional, Union

from ..governance_context import get_context

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = None  # type: ignore


class KyraLangChainCallback(BaseCallbackHandler if BaseCallbackHandler else object):
    """
    LangChain callback that captures full LLM reasoning for governance.
    Add to any AgentExecutor: AgentExecutor(..., callbacks=[KyraLangChainCallback(governor)])
    """

    def __init__(self, governor: Any) -> None:
        if BaseCallbackHandler is not None:
            super().__init__()
        self._governor = governor
        self._llm_start_time: Optional[float] = None
        self._tool_start_time: Optional[float] = None
        self._current_tool_name: Optional[str] = None
        self._tools: List[Any] = []

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any
    ) -> None:
        """Hook 1 — LLM input captured before LLM call."""
        try:
            self._llm_start_time = time.time()
            flat_messages = []
            for group in messages:
                for msg in group:
                    role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    flat_messages.append({"role": role, "content": str(content)})

            model = (
                serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model")
                or "unknown"
            )
            tools_offered = [t.name for t in getattr(self, "_tools", [])]
            temp = serialized.get("kwargs", {}).get("temperature")
            max_tokens = serialized.get("kwargs", {}).get("max_tokens")

            self._governor._tracer.track_llm_input(
                model=model,
                messages=flat_messages,
                tools_offered=tools_offered,
                temperature=temp,
                max_tokens=max_tokens,
            )
        except Exception:
            pass

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Hook 1 fallback for non-chat completion models."""
        try:
            self._llm_start_time = time.time()
            messages = [{"role": "user", "content": p} for p in prompts]
            model = serialized.get("kwargs", {}).get("model_name", "unknown")
            self._governor._tracer.track_llm_input(
                model=model,
                messages=messages,
                tools_offered=[],
                temperature=None,
                max_tokens=None,
            )
        except Exception:
            pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Hook 2 — LLM output captured after LLM responds."""
        try:
            latency_ms = int(
                (time.time() - (self._llm_start_time or time.time())) * 1000
            )
            generations = getattr(response, "generations", [[]])
            first_gen = generations[0][0] if generations and generations[0] else None

            response_text = ""
            thinking_text = ""
            finish_reason = "stop"
            input_tokens = output_tokens = thinking_tokens = 0

            if first_gen:
                response_text = getattr(first_gen, "text", "") or ""
                msg = getattr(first_gen, "message", None)
                if msg:
                    raw_tcs = (
                        getattr(msg, "tool_calls", None)
                        or getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
                    )
                    for _ in raw_tcs or []:
                        pass  # tool_calls_chosen not used in track_llm_output
                    thinking_text = (
                        getattr(msg, "thinking", None)
                        or getattr(msg, "additional_kwargs", {}).get("thinking", "")
                        or ""
                    )
                    finish_reason = (
                        getattr(msg, "response_metadata", {}).get(
                            "finish_reason", "stop"
                        )
                        or "stop"
                    )

            llm_output = getattr(response, "llm_output", {}) or {}
            usage = llm_output.get("token_usage", {}) or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            thinking_tokens = (
                usage.get("reasoning_tokens", 0)
                or usage.get("completion_tokens_details", {}).get(
                    "reasoning_tokens", 0
                )
                or 0
            )

            self._governor._tracer.track_llm_output(
                response_text=response_text,
                thinking_text=thinking_text,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                latency_ms=latency_ms,
            )
        except Exception:
            pass

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Hook 3 extension — LLM's stated rationale for this tool call."""
        try:
            rationale = getattr(action, "log", "") or ""
            self._governor._tracer.set_chosen_tool_rationale(rationale.strip())
        except Exception:
            pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Track tool start time for execution duration."""
        try:
            self._tool_start_time = time.time()
            self._current_tool_name = (
                serialized.get("name") or kwargs.get("name", "unknown")
            )
        except Exception:
            pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Hook 4 — tool output captured after tool executes."""
        try:
            exec_ms = int(
                (time.time() - (self._tool_start_time or time.time())) * 1000
            )
            ctx = get_context()
            seq = ctx.aggregate_action_count if ctx else 0
            self._governor._tracer.record_tool_result(
                tool_name=self._current_tool_name or "unknown",
                output=str(output),
                execution_time_ms=exec_ms,
                success=True,
                sequence_number=seq,
            )
        except Exception:
            pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Hook 4 error path."""
        try:
            exec_ms = int(
                (time.time() - (self._tool_start_time or time.time())) * 1000
            )
            ctx = get_context()
            seq = ctx.aggregate_action_count if ctx else 0
            self._governor._tracer.record_tool_result(
                tool_name=self._current_tool_name or "unknown",
                output="",
                execution_time_ms=exec_ms,
                success=False,
                sequence_number=seq,
            )
        except Exception:
            pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Record LLM error — clear pending trace."""
        try:
            self._governor._tracer.clear_current_llm_call()
        except Exception:
            pass

    def set_tools(self, tools: List[Any]) -> None:
        """Provide the tool list so on_chat_model_start can include toolsOffered."""
        self._tools = tools
