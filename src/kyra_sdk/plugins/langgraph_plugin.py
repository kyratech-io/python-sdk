from typing import Any

from ..models import ErrGovernanceBlock


class KyraToolNode:
    """
    Wraps a LangGraph ToolNode with Kyra pre-execution enforcement.
    Usage:
        tool_node = governor.wrap_tool_node(ToolNode(tools))
    """

    def __init__(self, tool_node: Any, governor: Any):
        self.tool_node = tool_node
        self.governor = governor
        # Expose same interface as ToolNode
        self.name = getattr(tool_node, "name", "tools")

    def __call__(self, state: dict) -> dict:
        """Called by LangGraph for each tool invocation in the graph"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        # Extract tool calls from last AIMessage
        if last_message and hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                ok, block_reason = self.governor._evaluate_before_call(
                    tool_name=tool_call["name"],
                    tool_description="",
                    parameters=tool_call.get("args", {}),
                    framework_override="LANGGRAPH",
                )
                if not ok:
                    raise ErrGovernanceBlock(block_reason)

        # If all passed, delegate to original ToolNode
        result = self.tool_node(state)
        # Hook 4: record each tool result for agentTrace and emit tool-result audit
        try:
            messages = result.get("messages", [])
            for msg in messages:
                name = getattr(msg, "name", None)
                content = getattr(msg, "content", "") or ""
                if isinstance(name, str) and name:
                    self.governor._tracer.record_tool_result(
                        tool_name=name,
                        output=str(content),
                        execution_time_ms=0,
                        success=True,
                        sequence_number=self.governor._tracer.next_sequence(),
                    )
                    try:
                        self.governor._emit_tool_result(
                            tool_name=name,
                            execution_time_ms=0,
                            success=True,
                            error_message=None,
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        return result
