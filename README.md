## Kyra Python SDK Integration – Function-by-function Guide (`langchain-kyra`)

This guide explains the Kyra governance SDK for Python in terms of the **functions and classes you call from your agent**, using a consistent structure:

- **Function name**
- **How and where to use**
- **Code snippet / example**
- **Other relevant information**

It targets LangChain, CrewAI, and LangGraph agents built in Python.

---

## High-level flow (what you do per agent)

For a typical Python agent using Kyra:

1. Create a `KyraGovernor` with your API key and configuration.
2. Wrap your tools with `governor.wrap(tools, framework=...)`.
3. (Optional but recommended) Use `GovernanceContext` and `set_context` to propagate governance metadata across your run.
4. Catch `ErrGovernanceBlock` (or Kyra-specific exceptions) when actions are blocked, escalated, or missing parameters.

The sections below walk the main APIs in that structure.

---

## Core configuration and setup

### `KyraGovernor`

- **Function name**: `KyraGovernor(api_key: str, **config)`
- **How and where to use**:
  - Instantiate once when your application or agent process starts.
  - Reuse the same governor object across runs of the same logical agent.
- **Code snippet / example**:

```python
from langchain_kyra import KyraGovernor

governor = KyraGovernor(
    api_key="kyra_sk_...",        # required
    server_url="https://api.kyratech.io",  # optional, default may be set in the SDK
    fail_open=True,               # allow when Kyra is unreachable; set False to fail closed
    mode="enforce",               # "enforce" | "shadow"
    agent_id="my-agent-v1",       # optional but recommended
)
```

- **Other relevant information**:
  - `mode="enforce"` means tool calls are blocked on policy violation.
  - `mode="shadow"` means Kyra evaluates but does not block (violations are logged only).
  - `fail_open=True` allows the action if the Kyra server is unreachable; `False` raises a server-unavailable-style error instead.

---

## Wrapping tools

### `governor.wrap(tools, framework=None)`

- **Function name**: `governor.wrap(tools, framework: str | None = None)`
- **How and where to use**:
  - Call after you build your tools list (LangChain Tools, CrewAI tools, etc.).
  - Use the **returned tools** instead of the raw ones in your agent.
- **Code snippet / example (generic / minimal)**:

```python
from langchain_kyra import KyraGovernor

governor = KyraGovernor(api_key="kyra_sk_...")
governed_tools = governor.wrap(tools)  # returns a list — use directly as tools
```

- **Code snippet / example (LangChain with telemetry)**:

```python
from langchain_kyra import KyraGovernor
from langchain.agents import create_react_agent

governor = KyraGovernor(api_key="kyra_sk_...")
governed_tools, callback = governor.wrap(tools, framework="langchain")

agent = create_react_agent(llm=llm, tools=governed_tools, prompt=prompt)
# When using AgentExecutor: AgentExecutor(..., callbacks=[callback])
```

- **Other relevant information**:
  - With `framework="langchain"`, `wrap` returns `(governed_tools, callback)`:
    - `governed_tools`: tools that perform Kyra evaluation before execution.
    - `callback`: a `KyraLangChainCallback` that records LLM inputs/outputs, tool choices, and results.
  - For other frameworks (CrewAI, LangGraph) you pass `framework="crewai"` or `framework="langgraph"`; the function still returns governed tools compatible with those frameworks.

---

## Governance context (per run)

### `GovernanceContext.from_human_message`

- **Function name**: `GovernanceContext.from_human_message(intent: str, root_agent_id: str)`
- **How and where to use**:
  - Call at the start of a run, using the **first user message** and the root agent ID.
  - Then attach it via `set_context` so all Kyra evaluations in this run share the same governance metadata.
- **Code snippet / example**:

```python
from langchain_kyra import GovernanceContext, set_context

ctx = GovernanceContext.from_human_message(
    "Refund order 123 if policy allows",
    root_agent_id="refund-agent-v1",
)

set_context(ctx)
# Run your agent here; Kyra will see governanceContext (traceId, intent, etc.)
```

- **Other relevant information**:
  - `GovernanceContext` tracks:
    - `trace_id`, `session_id`
    - `root_agent_id`
    - `original_intent_verbatim` (first user message)
    - `aggregate_action_count`, `aggregate_rows_affected`
    - `highest_tier_in_chain` (T0–T4)

---

### `GovernanceContext.from_agent_spawn`

- **Function name**: `GovernanceContext.from_agent_spawn(parent: GovernanceContext, child_agent_id: str)`
- **How and where to use**:
  - Use when a top-level/orchestrator agent spawns a child agent and you want Kyra to track the parent/child relationship.
- **Code snippet / example**:

```python
from langchain_kyra import GovernanceContext, set_context

parent = GovernanceContext.from_human_message(
    "Refund order 123",
    root_agent_id="orchestrator-agent",
)

child = GovernanceContext.from_agent_spawn(
    parent,
    child_agent_id="refund-agent-v1",
)

set_context(child)
# Run the child agent here; Kyra will send parentTraceId/parentAgentId
```

- **Other relevant information**:
  - This allows Kyra to reason over full chains of agents when applying policies or analysis.

---

### `set_context` / `get_context`

- **Function names**:
  - `set_context(ctx: GovernanceContext) -> None`
  - `get_context() -> GovernanceContext | None`
- **How and where to use**:
  - `set_context` is called once per run (after creating the appropriate context).
  - `get_context` is generally only needed in advanced scenarios; most users rely on the SDK attaching context automatically for evaluations.
- **Code snippet / example**:

```python
from langchain_kyra import GovernanceContext, set_context, get_context

ctx = GovernanceContext.from_human_message("...", root_agent_id="my-agent-v1")
set_context(ctx)

# Later, inside your code:
current_ctx = get_context()
```

- **Other relevant information**:
  - Under the hood, the SDK uses this context when building the `ActionRequest` sent to Kyra.

---

## Error handling and outcomes

### `ErrGovernanceBlock`

- **Function name**: `class ErrGovernanceBlock(Exception)`
- **How and where to use**:
  - Raised by **wrapped tools** when Kyra decides the action must not proceed (BLOCK, ESCALATE, or fail-closed server errors).
  - Catch it around your tool invocations or inside your agent’s error handling layer.
- **Code snippet / example**:

```python
from langchain_kyra import ErrGovernanceBlock

try:
    result = await some_governed_tool.ainvoke({"param": "value"})
except ErrGovernanceBlock as e:
    print("Governance block:", e.msg)
    # Decide whether to ask the user for more details, surface an escalation message,
    # or stop the run.
```

- **Other relevant information**:
  - The single exported error type for “this call should not proceed”.
  - Its `msg` field is safe to log or present to users.

---

### Kyra exception hierarchy (optional advanced usage)

- **Function / class names**:
  - `KyraException`
  - `KyraBlockedException`
  - `KyraEscalationDeniedException`
  - `KyraReturnToUserException`
  - `KyraServerUnavailableException`
- **How and where to use**:
  - Useful if you work directly with `EvaluationDecision` or advanced flows where you differentiate between “blocked”, “escalation denied”, and “return to user”.
- **Code snippet / example**:

```python
from langchain_kyra import KyraReturnToUserException

try:
    result = await some_governed_tool.ainvoke(params)
except KyraReturnToUserException as e:
    print("Missing parameters:", e.missing_parameters)
    # Ask the user to supply the missing fields
```

- **Other relevant information**:
  - All these classes carry an optional `EvaluationDecision` for more detailed inspection (gates, tier, outcome, etc.).

---

## Policies, tiers, and evaluation model

### `PolicyDocument` (wire model)

- **Function name**: `class PolicyDocument(dataclass)`
- **How and where to use**:
  - Represents explicit policies the server enforces; typically used when you construct registration payloads or call lower-level APIs.
- **Code snippet / example**:

```python
from langchain_kyra.models import PolicyDocument

policies = [
    PolicyDocument(
        policy_id="refundco-cash-only-v1",
        description="Refunds must be cash only",
        applies_to_tools=["issue_refund", "issue_chargeback_response"],  # multiple tools example
        condition="params.type != cash",
        action="BLOCK",
        tier="T2",
    ),
]
```

- **Other relevant information**:
  - Field meanings:
    - `policy_id`: unique ID for the policy.
    - `description`: human-readable summary.
    - `applies_to_tools`: one or more tool names this policy should apply to.
    - `condition`: expression evaluated on the request (params, governance context, etc.).
    - `action`: `"BLOCK"` or `"ESCALATE"`.
    - `tier`: governance tier for this policy (`"T0"`–`"T4"`), see below.

---

### Tiers (`requested_tier`, `tier`, and evaluation tier)

- **Function / field names**:
  - `requested_tier` on `ActionRequest`
  - `tier` on `PolicyDocument`
  - `tier` on `EvaluationDecision`
- **How and where to use**:
  - Use `requested_tier` to **floor** the risk level of a tool call (server uses `max(llmClassifiedTier, requested_tier)`).
  - Use `tier` on `PolicyDocument` to describe how sensitive the policy is.
- **Other relevant information**:
  - Typical meanings:
    - `T0`: read-only / informational actions (e.g. lookups).
    - `T1`: low-risk writes (e.g. updating non-critical metadata).
    - `T2`: medium-risk actions (e.g. reversible financial operations).
    - `T3`: high-risk actions (e.g. irreversible payments, chargebacks).
    - `T4`: very high-risk / critical controls (e.g. admin-only or production-wide changes).

---

## Install, build, and publish (for context)

These are not functions, but are useful when wiring the SDK into your environment.

### Install

- **How and where to use**:
  - Install the Python SDK from PyPI.
- **Code snippet / example**:

```bash
pip install langchain-kyra

# With optional framework extras:
pip install langchain-kyra[langchain]
pip install langchain-kyra[crewai]
pip install langchain-kyra[langgraph]
pip install langchain-kyra[all]
```

---

### Build (local)

- **How and where to use**:
  - Build the package locally when developing or testing changes to the SDK.
- **Code snippet / example**:

```bash
pip install build
python -m build
pip install dist/langchain_kyra-1.0.0-py3-none-any.whl  # local test
```

---

### Publish (to PyPI)

- **How and where to use**:
  - Upload built distributions to PyPI using `twine`.
- **Code snippet / example**:

```bash
twine upload dist/*
```

