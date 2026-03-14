import httpx
import time
import asyncio
import logging
import threading
from typing import List, Optional, Any, Tuple, Union

from .governance_context import GovernanceContext, get_context
from .session_tracer import SessionTracer
from .models import (
    ActionRequest, EvaluationDecision,
    KyraServerUnavailableException,
    PolicyDocument,
)

logger = logging.getLogger("kyra_sdk")
SDK_VERSION = "1.0.0"


def _normalize_mode(mode: Optional[str]) -> Optional[str]:
    """Map config mode ('enforce'|'shadow') to wire format ('ENFORCE'|'SHADOW'). Empty/unknown returns None."""
    if not mode:
        return None
    m = mode.strip().lower()
    if m == "enforce":
        return "ENFORCE"
    if m == "shadow":
        return "SHADOW"
    return None


WrapFramework = str  # "langchain" | "langgraph" | "crewai" | "alchemyst" | "generic"

FRAMEWORK_WIRE = {
    "langchain": "LANGCHAIN",
    "langgraph": "LANGGRAPH",
    "crewai": "CREWAI",
    "alchemyst": "ALCHEMYST",
    "generic": "GENERIC",
}


def normalize_framework(f: Optional[str]) -> str:
    """Map framework option to wire format (e.g. LANGCHAIN, GENERIC)."""
    if not f:
        return "LANGCHAIN"
    key = f.strip().lower()
    return FRAMEWORK_WIRE.get(key, key.upper().replace("-", "_"))


class KyraGovernor:
    """
    Main entry point for Kyra governance.

    Usage:
        governor = KyraGovernor(api_key="kyra_sk_...")
        tools = governor.wrap(tools)
        agent = create_react_agent(llm, tools, prompt)
    """

    def __init__(
        self,
        api_key: str,
        server_url: str = "https://api.kyratech.io",
        timeout_ms: int = 5000,
        fail_open: bool = True,
        mode: Optional[str] = None,  # "enforce" | "shadow" — sent as ENFORCE | SHADOW
        agent_id: Optional[str] = None,
        session_intent: Optional[str] = None,
        framework: str = "LANGCHAIN",
        additional_llm_endpoints: Optional[List[str]] = None,
        memory_endpoints: Optional[List[str]] = None,
    ):
        self.api_key = api_key
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout_ms / 1000
        self.fail_open = fail_open
        self.mode = mode
        self.agent_id = agent_id
        self.session_intent = session_intent
        self.framework = framework
        self._llm_endpoints: List[str] = additional_llm_endpoints or []
        self._memory_endpoints: List[str] = memory_endpoints or []
        self._registered_agent_id: Optional[str] = None
        self._prompt_hash: Optional[str] = None
        self._tracer = SessionTracer()
        # Track sessions we've already emitted SESSION_STARTED for (avoid duplicates).
        self._started_sessions: set[str] = set()

        self._client = httpx.Client(
            headers={"X-Kyra-Key": api_key, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        self._async_client = httpx.AsyncClient(
            headers={"X-Kyra-Key": api_key, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        self._poller_stop = threading.Event()
        self._start_escalation_poller()

        # Configure audit queue and HTTP interceptor defaults so that
        # HTTP calls made within this process can emit audit telemetry
        # without impacting the main evaluation path.
        try:
            from .audit import audit_queue  # type: ignore
            from .core import http_interceptor  # type: ignore

            audit_queue.configure(self.server_url)
            http_interceptor.configure_endpoints(
                llm_endpoints=self._llm_endpoints,
                memory_endpoints=self._memory_endpoints,
            )
        except Exception:
            # Never fail governor construction due to optional audit wiring.
            pass

    def wrap(
        self,
        tools_or_node: Union[List[Any], Any],
        framework: str = "generic",
    ) -> Union[List[Any], Tuple[List[Any], Any], Any]:
        """
        Unified wrap: supports all frameworks via framework=.

        - framework="langchain" => (wrapped_tools, callback)
        - framework="langgraph" => pass a single ToolNode as first arg, returns KyraToolNode
        - framework="crewai" | "alchemyst" | "generic" (default) => list of wrapped tools
        """
        framework = (framework or "generic").strip().lower()
        wire = normalize_framework(framework)

        if framework == "langgraph":
            from .plugins.langgraph_plugin import KyraToolNode
            tool_node = tools_or_node[0] if isinstance(tools_or_node, list) else tools_or_node
            return KyraToolNode(tool_node=tool_node, governor=self)

        tools = tools_or_node if isinstance(tools_or_node, list) else [tools_or_node]
        if framework == "langchain":
            from .core.kyra_tool import KyraWrappedTool
            from .plugins.langchain_callback import KyraLangChainCallback
            callback = KyraLangChainCallback(self)
            callback.set_tools(tools)
            wrapped = [KyraWrappedTool(wrapped_tool=t, governor=self, framework_wire=wire) for t in tools]
            return (wrapped, callback)

        if framework == "generic":
            from .core.generic_tool import GenericWrappedTool
            return [GenericWrappedTool(tool=t, governor=self) for t in tools]

        from .core.kyra_tool import KyraWrappedTool
        return [KyraWrappedTool(wrapped_tool=t, governor=self, framework_wire=wire) for t in tools]

    def wrap_tool_node(self, tool_node: Any) -> Any:
        """Deprecated: use governor.wrap(tool_node, framework="langgraph") instead."""
        return self.wrap(tool_node, framework="langgraph")

    def _evaluate_before_call(
        self,
        tool_name: str,
        tool_description: str,
        parameters: dict,
        framework_override: Optional[str] = None,
        agent_context: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Single internal evaluation path — all adapters use this.
        agent_context: optional override; when None, uses get_agent_context() (thread-local).
        """
        ctx = get_context()
        # Lazy SESSION_STARTED: emit once per session before first evaluate in that session.
        try:
            sid = ctx.session_id if ctx else None
            if sid and sid not in self._started_sessions:
                self._started_sessions.add(sid)
                self._emit_session_event("SESSION_STARTED", sid)
        except Exception:
            pass
        framework = normalize_framework(framework_override) if framework_override else self.framework
        if agent_context is None:
            try:
                from .agent_context import get_agent_context  # type: ignore
                agent_ctx = get_agent_context()
            except Exception:
                agent_ctx = None
        else:
            agent_ctx = agent_context
        req = ActionRequest(
            tool_name=tool_name,
            tool_description=tool_description,
            parameters=parameters,
            agent_id=self._registered_agent_id or self.agent_id,
            governance_context=ctx.to_dto() if ctx else self._empty_ctx(),
            session_intent=self.session_intent,
            framework=framework,
            sdk_version=SDK_VERSION,
            prompt_hash=self._prompt_hash,
            agent_trace=self._tracer.build_agent_trace(),
            mode=self._normalize_mode(self.mode),
            session_id=ctx.session_id if ctx else None,
            trace_id=trace_id or (ctx.trace_id if ctx else None),
            agent_context=agent_ctx,
        )
        try:
            resp = self._client.post(
                f"{self.server_url}/v1/evaluate",
                json=req.to_dict(),
            )
            resp.raise_for_status()
            decision = EvaluationDecision.from_dict(resp.json())
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(f"Kyra server unreachable: {e}. fail_open={self.fail_open}")
            if self.fail_open:
                return (True, "")
            raise KyraServerUnavailableException(str(e))
        # Persist kyraEventId on the governance context for tool-result audit.
        if ctx is not None:
            ctx.last_kyra_event_id = decision.kyra_event_id
        ok, block_reason = self._handle_decision(decision)
        if not ok and decision.outcome == "ESCALATE" and decision.escalation_id:
            threading.Thread(
                target=self._post_escalation_async,
                args=(tool_name, tool_description, parameters, "", decision),
                daemon=True,
            ).start()
        if ok and ctx is not None:
            ctx.aggregate_action_count += 1
        return (ok, block_reason)

    def evaluate(
        self,
        tool_name: str,
        tool_description: str,
        parameters: dict,
        framework_override: Optional[str] = None,
        agent_context: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Returns (ok, block_reason). ok=True for ALLOW, False for BLOCK/ESCALATE/server error.
        On ESCALATE returns (False, reason) and fires async POST to /v1/escalations.
        agent_context: optional; when provided, used for this call instead of get_agent_context().
        trace_id: optional; user-provided or from context; if not set, server generates one.
        """
        return self._evaluate_before_call(
            tool_name, tool_description, parameters, framework_override, agent_context, trace_id
        )

    async def _evaluate_before_call_async(
        self,
        tool_name: str,
        tool_description: str,
        parameters: dict,
        framework_override: Optional[str] = None,
        agent_context: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Async single internal evaluation path. agent_context: optional override; else get_agent_context(). trace_id: optional; if not set, server generates."""
        ctx = get_context()
        # Lazy SESSION_STARTED: emit once per session before first evaluate in that session.
        try:
            sid = ctx.session_id if ctx else None
            if sid and sid not in self._started_sessions:
                self._started_sessions.add(sid)
                self._emit_session_event("SESSION_STARTED", sid)
        except Exception:
            pass
        framework = normalize_framework(framework_override) if framework_override else self.framework
        if agent_context is None:
            try:
                from .agent_context import get_agent_context  # type: ignore
                agent_ctx = get_agent_context()
            except Exception:
                agent_ctx = None
        else:
            agent_ctx = agent_context
        req = ActionRequest(
            tool_name=tool_name,
            tool_description=tool_description,
            parameters=parameters,
            agent_id=self._registered_agent_id or self.agent_id,
            governance_context=ctx.to_dto() if ctx else self._empty_ctx(),
            session_intent=self.session_intent,
            framework=framework,
            sdk_version=SDK_VERSION,
            prompt_hash=self._prompt_hash,
            agent_trace=self._tracer.build_agent_trace(),
            mode=self._normalize_mode(self.mode),
            session_id=ctx.session_id if ctx else None,
            trace_id=trace_id or (ctx.trace_id if ctx else None),
            agent_context=agent_ctx,
        )
        try:
            resp = await self._async_client.post(
                f"{self.server_url}/v1/evaluate",
                json=req.to_dict(),
            )
            resp.raise_for_status()
            decision = EvaluationDecision.from_dict(resp.json())
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(f"Kyra server unreachable (async): {e}. fail_open={self.fail_open}")
            if self.fail_open:
                return (True, "")
            raise KyraServerUnavailableException(str(e))
        if ctx is not None:
            ctx.last_kyra_event_id = decision.kyra_event_id
        ok, block_reason = self._handle_decision(decision)
        if not ok and decision.outcome == "ESCALATE" and decision.escalation_id:
            threading.Thread(
                target=self._post_escalation_async,
                args=(tool_name, tool_description, parameters, "", decision),
                daemon=True,
            ).start()
        if ok and ctx is not None:
            ctx.aggregate_action_count += 1
        return (ok, block_reason)

    async def evaluate_async(
        self,
        tool_name: str,
        tool_description: str,
        parameters: dict,
        framework_override: Optional[str] = None,
        agent_context: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Async version of evaluate(); returns (ok, block_reason)."""
        return await self._evaluate_before_call_async(
            tool_name, tool_description, parameters, framework_override, agent_context, trace_id
        )

    def _normalize_mode(self, mode: Optional[str]) -> Optional[str]:
        """Map config mode to wire format."""
        return _normalize_mode(mode)

    def _handle_decision(self, decision: EvaluationDecision) -> Tuple[bool, str]:
        """Return (ok, block_reason). ok=True for ALLOW, False otherwise."""
        if decision.outcome == "ALLOW":
            return (True, "")
        if decision.outcome == "BLOCK":
            return (False, decision.block_reason or "Action blocked by Kyra policy")
        if decision.outcome == "ESCALATE":
            return (False, decision.block_reason or "escalation required")
        if decision.outcome == "RETURN_TO_USER":
            return (False, f"missing parameters: {decision.missing_parameters}")
        return (False, decision.outcome)

    def _post_escalation_async(
        self,
        tool_name: str,
        tool_description: str,
        parameters: dict,
        tier: str,
        decision: EvaluationDecision,
    ) -> None:
        try:
            self._client.post(
                f"{self.server_url}/v1/escalations",
                json={
                    "toolName": tool_name,
                    "toolDescription": tool_description,
                    "parameters": parameters,
                    "tier": tier,
                    "blockReason": decision.block_reason,
                    "traceId": decision.trace_id,
                    "sessionId": decision.evaluation_id,
                    "escalationId": decision.escalation_id,
                },
            )
        except Exception as e:
            logger.warning(f"post_escalation_async failed: {e}")

    def _start_escalation_poller(self) -> None:
        def poll() -> None:
            while not self._poller_stop.wait(timeout=30.0):
                agent_id = self._registered_agent_id or self.agent_id
                if not agent_id:
                    continue
                try:
                    resp = self._client.get(
                        f"{self.server_url}/v1/escalations",
                        params={"status": "approved", "agentId": agent_id},
                    )
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get("escalations", [])
                    for es in items:
                        if es.get("status") != "APPROVED":
                            continue
                        self._client.post(f"{self.server_url}/v1/agents/{agent_id}/graph/rules", json={})
                        self._client.patch(
                            f"{self.server_url}/v1/escalations/{es.get('escalationId', '')}",
                            json={"status": "processed"},
                        )
                except Exception:
                    pass

        t = threading.Thread(target=poll, daemon=True)
        t.start()

    def _emit_tool_result(
        self,
        tool_name: str,
        execution_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Emit a lightweight tool result audit event when kyraEventId is available.
        """
        try:
            from .governance_context import get_context  # type: ignore
            from .audit import get_audit_queue  # type: ignore
        except Exception:
            return
        ctx = get_context()
        if not ctx or not getattr(ctx, "last_kyra_event_id", None):
            return
        status = "SUCCESS" if success else "FAILURE"
        payload = {
            "agentId": ctx.root_agent_id,
            "sessionId": ctx.session_id,
            "kyraEventId": ctx.last_kyra_event_id,
            "status": status,
            "durationMs": execution_time_ms,
        }
        if not success and error_message:
            payload["errorMessage"] = error_message
        try:
            queue = get_audit_queue()
            queue.enqueue_tool_result(payload)
        except Exception:
            pass

    def _empty_ctx(self):
        from .models import GovernanceContextDto
        return GovernanceContextDto()

    def _emit_session_event(self, event_type: str, session_id: str) -> None:
        """
        Emit a SESSION_STARTED or SESSION_COMPLETED audit event for the given session.
        """
        try:
            from .audit import get_audit_queue  # type: ignore
        except Exception:
            return
        payload = {
            "agentId": self._registered_agent_id or self.agent_id,
            "sessionId": session_id,
            "eventType": event_type,
            "sdkVersion": SDK_VERSION,
            "framework": self.framework,
            "mode": self._normalize_mode(self.mode),
        }
        try:
            queue = get_audit_queue()
            queue.enqueue_session_event(payload)
        except Exception:
            pass

    def register_agent(self, agent_name: str, system_prompt: str,
                        tools: List[Any], policies: Optional[List[PolicyDocument]] = None) -> str:
        """
        Register agent with Kyra server. Called automatically on first wrap().
        Returns agentId to use in subsequent evaluate() calls.
        """
        tool_defs = []
        for t in tools:
            d = {
                "name": t.name,
                "description": t.description,
                "parametersSchema": {
                    k: str(v) for k, v in (t.args_schema.schema().get("properties", {}) or {}).items()
                } if hasattr(t, "args_schema") and t.args_schema else {},
            }
            requested_tier = getattr(t, "requested_tier", None)
            if requested_tier:
                d["requestedTier"] = requested_tier
            tool_defs.append(d)
        import hashlib, json
        tool_defs.sort(key=lambda x: x["name"])
        source_hash = "sha256:" + hashlib.sha256(
            (system_prompt + json.dumps(tool_defs, sort_keys=True)).encode()
        ).hexdigest()

        try:
            resp = self._client.post(
                f"{self.server_url}/v1/agents/register",
                json={
                    "agentName": agent_name,
                    "systemPrompt": system_prompt,
                    "tools": tool_defs,
                    "framework": self.framework,
                    "sdkVersion": SDK_VERSION,
                    "sourceHash": source_hash,
                    "policies": [p.to_dict() for p in policies] if policies else None,
                },
            )
            resp.raise_for_status()
            self._registered_agent_id = resp.json()["agentId"]
            # Block 3: store prompt hash for evaluate payload (used in all subsequent evaluate calls)
            if system_prompt:
                self._prompt_hash = "sha256:" + hashlib.sha256(
                    system_prompt.encode("utf-8")
                ).hexdigest()
            return self._registered_agent_id
        except Exception as e:
            logger.warning(f"Agent registration failed: {e}. Continuing without agentId.")
            return ""

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
