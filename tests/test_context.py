import pytest
from langchain_kyra import GovernanceContext, get_context, set_context


def test_from_human_message():
    ctx = GovernanceContext.from_human_message("  Delete user 123  ", root_agent_id="agent-1")
    assert ctx.original_intent_verbatim == "Delete user 123"
    assert ctx.root_agent_id == "agent-1"
    assert len(ctx.trace_id) == 36
    assert len(ctx.original_intent_hash) == 64


def test_to_dto():
    ctx = GovernanceContext.from_human_message("hello")
    dto = ctx.to_dto()
    assert dto.trace_id == ctx.trace_id
    assert dto.original_intent_hash == ctx.original_intent_hash
    assert dto.chain_depth == 0


def test_from_agent_spawn_inherits_and_links():
    parent = GovernanceContext.from_human_message("Delete user 123", root_agent_id="parent-agent")
    child = GovernanceContext.from_agent_spawn(parent, child_agent_id="child-agent")

    assert child.parent_trace_id == parent.trace_id
    assert child.session_id == parent.session_id
    assert child.original_intent_verbatim == parent.original_intent_verbatim
    assert child.trace_id != parent.trace_id
    assert child.parent_agent_id == "child-agent"


def test_to_headers():
    ctx = GovernanceContext.from_human_message("hello")
    headers = ctx.to_headers()
    assert "X-Kyra-Trace" in headers
    assert "X-Kyra-Governance" in headers
    assert headers["X-Kyra-Trace"] == ctx.trace_id


def test_increment_depth():
    ctx = GovernanceContext.from_human_message("hello")
    ctx2 = ctx.increment_depth()
    assert ctx2.chain_depth == 1
    assert ctx.chain_depth == 0


def test_get_set_context():
    set_context(None)
    assert get_context() is None
    ctx = GovernanceContext.from_human_message("test")
    set_context(ctx)
    assert get_context() is ctx
    set_context(None)
