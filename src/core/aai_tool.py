import time
from langchain_core.tools import BaseTool
from typing import Any, Optional, Type
from pydantic import BaseModel, ConfigDict

from ..models import ErrGovernanceBlock

class KyraWrappedTool(BaseTool):
    """
    Wraps any LangChain/CrewAI BaseTool with Kyra pre-execution enforcement.
    Fully transparent — same name, description, and schema as the wrapped tool.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped_tool: Any  # BaseTool
    governor: Any      # KyraGovernor
    framework_wire: Optional[str] = None  # e.g. LANGCHAIN, CREWAI; sent to server when set

    @property
    def name(self) -> str:
        return self.wrapped_tool.name

    @property
    def description(self) -> str:
        return self.wrapped_tool.description

    @property
    def args_schema(self) -> Optional[Type[BaseModel]]:
        return getattr(self.wrapped_tool, "args_schema", None)

    def _run(self, *args, **kwargs) -> Any:
        requested_tier = getattr(self.wrapped_tool, "requested_tier", None)
        ok, block_reason = self.governor._evaluate_before_call(
            tool_name=self.wrapped_tool.name,
            tool_description=self.wrapped_tool.description,
            parameters=kwargs,
            requested_tier=requested_tier,
            framework_override=self.framework_wire,
        )
        if not ok:
            raise ErrGovernanceBlock(block_reason)
        start = time.perf_counter()
        try:
            result = self.wrapped_tool._run(*args, **kwargs)
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            self.governor._tracer.record_tool_result(
                self.wrapped_tool.name,
                str(result),
                execution_time_ms=execution_time_ms,
                success=True,
                sequence_number=self.governor._tracer.next_sequence(),
            )
            return result
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            self.governor._tracer.record_tool_result(
                self.wrapped_tool.name,
                "",
                execution_time_ms=execution_time_ms,
                success=False,
                sequence_number=self.governor._tracer.next_sequence(),
            )
            raise

    async def _arun(self, *args, **kwargs) -> Any:
        requested_tier = getattr(self.wrapped_tool, "requested_tier", None)
        ok, block_reason = await self.governor._evaluate_before_call_async(
            tool_name=self.wrapped_tool.name,
            tool_description=self.wrapped_tool.description,
            parameters=kwargs,
            requested_tier=requested_tier,
            framework_override=self.framework_wire,
        )
        if not ok:
            raise ErrGovernanceBlock(block_reason)
        start = time.perf_counter()
        try:
            result = await self.wrapped_tool._arun(*args, **kwargs)
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            self.governor._tracer.record_tool_result(
                self.wrapped_tool.name,
                str(result),
                execution_time_ms=execution_time_ms,
                success=True,
                sequence_number=self.governor._tracer.next_sequence(),
            )
            return result
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            self.governor._tracer.record_tool_result(
                self.wrapped_tool.name,
                "",
                execution_time_ms=execution_time_ms,
                success=False,
                sequence_number=self.governor._tracer.next_sequence(),
            )
            raise
