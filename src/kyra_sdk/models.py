from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class GovernanceContextDto:
    trace_id: str = ""
    root_agent_id: str = ""
    original_intent_verbatim: str = ""
    original_intent_hash: str = ""
    chain_depth: int = 0
    aggregate_rows_affected: int = 0
    aggregate_action_count: int = 0
    session_id: str = ""
    parent_trace_id: Optional[str] = None
    parent_agent_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "traceId": self.trace_id,
            "rootAgentId": self.root_agent_id,
            "originalIntentVerbatim": self.original_intent_verbatim,
            "originalIntentHash": self.original_intent_hash,
            "chainDepth": self.chain_depth,
            "aggregateRowsAffected": self.aggregate_rows_affected,
        }
        if self.aggregate_action_count != 0:
            d["aggregateActionCount"] = self.aggregate_action_count
        if self.session_id:
            d["sessionId"] = self.session_id
        if self.parent_trace_id:
            d["parentTraceId"] = self.parent_trace_id
        if self.parent_agent_id:
            d["parentAgentId"] = self.parent_agent_id
        return d


@dataclass
class AgentContext:
    reasoning: Optional[str] = None
    chosen_action: Optional[str] = None
    confidence: Optional[float] = None
    memory_ids_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {}
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        if self.chosen_action is not None:
            d["chosenAction"] = self.chosen_action
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.memory_ids_used:
            d["memoryIdsUsed"] = list(self.memory_ids_used)
        return d


@dataclass
class ActionRequest:
    """Wire format sent to POST /v1/evaluate"""
    tool_name: str
    tool_description: str
    parameters: Dict[str, Any]
    governance_context: GovernanceContextDto
    agent_id: Optional[str] = None
    session_intent: Optional[str] = None
    framework: str = "LANGCHAIN"
    sdk_version: str = "1.0.0"
    prompt_hash: Optional[str] = None
    agent_trace: Optional[dict] = None
    mode: Optional[str] = None  # ENFORCE | SHADOW — request server to use this mode
    # Top-level sessionId for easier indexing on the server; duplicated
    # from GovernanceContextDto.session_id when present.
    session_id: Optional[str] = None
    # Top-level traceId: user-provided or from context; None = server generates.
    trace_id: Optional[str] = None
    agent_context: Optional[AgentContext] = None

    def to_dict(self) -> dict:
        d = {
            "toolName": self.tool_name,
            "toolDescription": self.tool_description,
            "parameters": self.parameters,
            "agentId": self.agent_id,
            "governanceContext": self.governance_context.to_dict(),
            "sessionIntent": self.session_intent,
            "framework": self.framework,
            "sdkVersion": self.sdk_version,
        }
        if self.prompt_hash:
            d["promptHash"] = self.prompt_hash
        if self.agent_trace:
            d["agentTrace"] = self.agent_trace
        if self.mode:
            d["mode"] = self.mode
        if self.session_id:
            d["sessionId"] = self.session_id
        if self.trace_id is not None:
            d["traceId"] = self.trace_id
        if self.agent_context:
            d["agentContext"] = self.agent_context.to_dict()
        return d

@dataclass
class PolicyDocument:
    policy_id: str
    description: str
    applies_to_tools: List[str]
    condition: str
    action: str  # "BLOCK" | "ESCALATE"
    tier: str    # "T0" - "T4"
    def to_dict(self) -> dict:
        return {
            "policyId": self.policy_id,
            "description": self.description,
            "appliesToTools": self.applies_to_tools,
            "condition": self.condition,
            "action": self.action,
            "tier": self.tier,
        }

@dataclass
class GateResultDto:
    gate: str
    passed: bool
    failure_reason: Optional[str]
    duration_ms: int


@dataclass
class EvaluationDecision:
    """Wire format returned from POST /v1/evaluate"""
    trace_id: str
    evaluation_id: str
    org_id: str
    outcome: str              # ALLOW | BLOCK | ESCALATE | RETURN_TO_USER
    mode: str                 # SHADOW | ENFORCE
    shadow_outcome: Optional[str]
    tier: str                 # T0 | T1 | T2 | T3 | T4
    block_reason: Optional[str] = None
    missing_parameters: Optional[List[str]] = None
    escalation_id: Optional[str] = None
    evaluation_ms: int = 0
    gate_results: List[GateResultDto] = field(default_factory=list)
    kyra_event_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationDecision":
        gate_results = [
            GateResultDto(
                gate=g.get("gate", ""),
                passed=g.get("passed", False),
                failure_reason=g.get("failureReason"),
                duration_ms=g.get("durationMs", 0),
            )
            for g in d.get("gateResults", [])
        ]
        return cls(
            trace_id=d.get("traceId", ""),
            evaluation_id=d.get("evaluationId", ""),
            org_id=d.get("orgId", ""),
            outcome=d.get("outcome", "ALLOW"),
            mode=d.get("mode", "SHADOW"),
            shadow_outcome=d.get("shadowOutcome"),
            tier=d.get("tier", "T0"),
            block_reason=d.get("blockReason"),
            missing_parameters=d.get("missingParameters"),
            escalation_id=d.get("escalationId"),
            evaluation_ms=d.get("evaluationMs", 0),
            gate_results=gate_results,
            kyra_event_id=d.get("kyraEventId"),
        )


# Exception hierarchy
class KyraException(Exception):
    def __init__(self, message: str, decision: Optional[EvaluationDecision] = None):
        super().__init__(message)
        self.decision = decision


class KyraBlockedException(KyraException):
    """Raised when Kyra blocks an agent action (enforce mode only)"""
    pass


class KyraEscalationDeniedException(KyraException):
    """Raised when a human approver denies an escalated action"""
    pass


class KyraReturnToUserException(KyraException):
    """Raised when required parameters are missing — return to user for clarification"""
    @property
    def missing_parameters(self) -> List[str]:
        if not self.decision:
            return []
        return self.decision.missing_parameters or []


class KyraServerUnavailableException(KyraException):
    """Raised when server unreachable and fail_open=False"""
    pass


class ErrGovernanceBlock(Exception):
    """Single exported governance error. Raised by wrapped tools when the action is blocked (BLOCK, ESCALATE, server error)."""
    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(msg if msg else "kyra: blocked")
