import hashlib
import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional, List
from .models import GovernanceContextDto


@dataclass
class GovernanceContext:
    """
    Immutable record propagated across all agent hops in a trace.
    Set once at session start (from user's message).
    aggregate_action_count and highest_tier_in_chain may be updated by the governor after each evaluate.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_agent_id: str = ""
    original_intent_verbatim: str = ""
    original_intent_hash: str = ""
    chain_depth: int = 0
    aggregate_rows_affected: int = 0
    aggregate_action_count: int = 0
    highest_tier_in_chain: str = ""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hops: List[dict] = field(default_factory=list)
    parent_trace_id: Optional[str] = None
    parent_agent_id: Optional[str] = None

    @classmethod
    def from_human_message(cls, message: str, root_agent_id: str = "") -> "GovernanceContext":
        intent = message.strip()
        return cls(
            trace_id=str(uuid.uuid4()),
            root_agent_id=root_agent_id,
            original_intent_verbatim=intent,
            original_intent_hash=hashlib.sha256(intent.encode("utf-8")).hexdigest(),
            session_id=str(uuid.uuid4()),
            aggregate_action_count=0,
            highest_tier_in_chain="",
        )

    @classmethod
    def from_agent_spawn(cls, parent: "GovernanceContext", child_agent_id: str) -> "GovernanceContext":
        return cls(
            trace_id=str(uuid.uuid4()),
            root_agent_id=parent.root_agent_id,
            original_intent_verbatim=parent.original_intent_verbatim,
            original_intent_hash=parent.original_intent_hash,
            chain_depth=parent.chain_depth,
            aggregate_rows_affected=parent.aggregate_rows_affected,
            aggregate_action_count=0,
            highest_tier_in_chain="T0",
            session_id=parent.session_id,
            hops=list(parent.hops),
            parent_trace_id=parent.trace_id,
            parent_agent_id=child_agent_id,
        )

    def increment_depth(self) -> "GovernanceContext":
        """Return new context with incremented depth — do not mutate original"""
        import dataclasses
        return dataclasses.replace(self, chain_depth=self.chain_depth + 1)

    def add_rows(self, count: int) -> "GovernanceContext":
        import dataclasses
        return dataclasses.replace(self,
            aggregate_rows_affected=self.aggregate_rows_affected + count)

    def to_dto(self) -> GovernanceContextDto:
        return GovernanceContextDto(
            trace_id=self.trace_id,
            root_agent_id=self.root_agent_id,
            original_intent_verbatim=self.original_intent_verbatim,
            original_intent_hash=self.original_intent_hash,
            chain_depth=self.chain_depth,
            aggregate_rows_affected=self.aggregate_rows_affected,
            aggregate_action_count=self.aggregate_action_count,
            highest_tier_in_chain=self.highest_tier_in_chain,
            session_id=self.session_id,
            parent_trace_id=self.parent_trace_id,
            parent_agent_id=self.parent_agent_id,
        )

    def to_headers(self) -> dict:
        """HTTP headers for propagation across service calls"""
        import json, base64
        payload = {
            "traceId": self.trace_id,
            "rootAgentId": self.root_agent_id,
            "originalIntentHash": self.original_intent_hash,
            "chainDepth": self.chain_depth,
        }
        headers = {
            "X-Kyra-Trace": self.trace_id,
            "X-Kyra-Governance": base64.b64encode(
                json.dumps(payload).encode()).decode(),
        }
        if self.session_id:
            headers["X-Kyra-Session"] = self.session_id
        return headers


# Thread-local context storage
_ctx_local = threading.local()


def get_context() -> Optional[GovernanceContext]:
    return getattr(_ctx_local, "ctx", None)


def set_context(ctx: Optional[GovernanceContext]) -> None:
    _ctx_local.ctx = ctx


def clear_context() -> None:
    if hasattr(_ctx_local, "ctx"):
        del _ctx_local.ctx
