"""
Session-scoped tracer for LLM calls and tool results, attached to evaluate as agentTrace.
"""
import hashlib
import threading
from typing import Any, Dict, List, Optional

MAX_PRIOR_TOOL_RESULTS = 5
MAX_MESSAGE_CONTENT_LEN = 2000


def _truncate_message(role: str, content: str) -> Dict[str, str]:
    if len(content) > MAX_MESSAGE_CONTENT_LEN:
        content = content[:MAX_MESSAGE_CONTENT_LEN] + "...[truncated]"
    return {"role": role, "content": content}


def _filter_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system = []
    others = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        truncated = _truncate_message(role, content)
        if role == "system":
            system.append(truncated)
        else:
            others.append(truncated)
    if len(others) > 3:
        others = others[-3:]
    return system + others


class SessionTracer:
    """Holds session-scoped LLM call data and last 5 tool results for agentTrace."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence_counter = 0
        self._current_llm_call: Optional[Dict[str, Any]] = None
        self._tool_results: List[Dict[str, Any]] = []

    def _next_seq(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    def next_sequence(self) -> int:
        """Increment and return next sequence number (for tool results)."""
        with self._lock:
            self._sequence_counter += 1
            return self._sequence_counter

    def track_llm_input(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools_offered: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        try:
            with self._lock:
                self._next_seq()
                filtered = _filter_messages(messages)
                self._current_llm_call = {
                    "model": model,
                    "inputMessages": filtered,
                    "toolsOffered": tools_offered or [],
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                }
        except Exception:
            pass

    def track_llm_output(
        self,
        response_text: str = "",
        thinking_text: str = "",
        finish_reason: str = "stop",
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        latency_ms: int = 0,
    ) -> None:
        try:
            with self._lock:
                if self._current_llm_call is None:
                    self._current_llm_call = {}
                self._next_seq()
                self._current_llm_call.update(
                    {
                        "responseText": response_text,
                        "thinkingText": thinking_text,
                        "finishReason": finish_reason,
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                        "thinkingTokens": thinking_tokens,
                        "latencyMs": latency_ms,
                    }
                )
        except Exception:
            pass

    def set_chosen_tool_rationale(self, rationale: str) -> None:
        try:
            with self._lock:
                if self._current_llm_call is not None:
                    self._current_llm_call["chosenToolRationale"] = rationale
        except Exception:
            pass

    def build_agent_trace(self) -> Optional[Dict[str, Any]]:
        """Build the agentTrace dict to attach to evaluate request. Omit if empty."""
        try:
            with self._lock:
                seq = self._next_seq()
                trace = {"sequenceNumber": seq}
                if self._current_llm_call:
                    trace["llmCall"] = self._current_llm_call
                if self._tool_results:
                    trace["priorToolResults"] = self._tool_results
                # Omit entirely if no meaningful data (optional per spec)
                if len(trace) == 1 and "priorToolResults" not in trace:
                    return None
                return trace
        except Exception:
            return None

    def record_tool_result(
        self,
        tool_name: str,
        output: str,
        execution_time_ms: int = 0,
        success: bool = True,
        sequence_number: Optional[int] = None,
    ) -> None:
        try:
            with self._lock:
                summary = output[:500] if len(output) > 500 else output
                output_hash = "sha256:" + hashlib.sha256(output.encode("utf-8")).hexdigest()
                result = {
                    "toolName": tool_name,
                    "outputSummary": summary,
                    "outputHash": output_hash,
                    "executionTimeMs": execution_time_ms,
                    "success": success,
                    "sequenceNumber": sequence_number or self._next_seq(),
                }
                self._tool_results.append(result)
                if len(self._tool_results) > MAX_PRIOR_TOOL_RESULTS:
                    self._tool_results = self._tool_results[-MAX_PRIOR_TOOL_RESULTS:]
        except Exception:
            pass

    def clear_current_llm_call(self) -> None:
        try:
            with self._lock:
                self._current_llm_call = None
        except Exception:
            pass
