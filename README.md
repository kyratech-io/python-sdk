# Kyra Python SDK

Runtime governance for AI agent actions in Python (LangChain, CrewAI, LangGraph). Every tool call is evaluated by Kyra before execution; blocked actions raise `ErrGovernanceBlock` so your agent can respond safely.

---

## Install

```bash
pip install kyra-sdk
```

The base package works with any tool (generic). To use Kyra’s wrappers for a specific framework, install the matching extra so those dependencies are available:

| Extra | Adds | Use when |
|-------|------|----------|
| `[langchain]` | langchain-core | Wrapping LangChain tools / AgentExecutor |
| `[langgraph]` | langgraph | Wrapping a LangGraph tool node |
| `[crewai]` | crewai | Wrapping CrewAI tools |
| `[all]` | all of the above | Supporting multiple frameworks |

```bash
pip install kyra-sdk[langchain]
pip install kyra-sdk[langgraph]
pip install kyra-sdk[crewai]
pip install kyra-sdk[all]
```

---

## Quick start

```python
from kyra_sdk import KyraGovernor, GovernanceContext, set_context, ErrGovernanceBlock

# 1. Create governor (once per process or per agent)
governor = KyraGovernor(
    api_key="kyra_sk_...",
    agent_id="my-agent-v1",
    fail_open=False,
)

# 2. Build your tools (LangChain, CrewAI, or generic) and wrap them
tools = [search_tool, refund_tool]
governed_tools = governor.wrap(tools)

# 3. For each run: create governance context from the user message and set it
ctx = GovernanceContext.from_human_message(
    "Refund order 123 if policy allows",
    root_agent_id="my-agent-v1",
)
set_context(ctx)

# 4. Run your agent with governed_tools; tool invocations are evaluated by Kyra
try:
    result = await agent.ainvoke({"input": "Refund order 123"})
except ErrGovernanceBlock as e:
    print("Blocked by policy:", e.msg)
```

---

## Configuration

Create the governor with your API key and optional settings:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `api_key` | Yes | — | Your Kyra API key (`kyra_sk_...`) |
| `timeout_ms` | No | 5000 | Request timeout in milliseconds |
| `fail_open` | No | `True` | If True, allow tool execution when Kyra is unreachable |
| `mode` | No | `None` | `"enforce"` or `"shadow"` — sent to server as ENFORCE / SHADOW |
| `agent_id` | No | `None` | Agent identifier for evaluate requests; recommended for multi-agent |
| `session_intent` | No | `None` | Optional session-level intent hint |
| `framework` | No | `"LANGCHAIN"` | Reported in wire format |

**Example (fail-closed, enforce mode):**

```python
governor = KyraGovernor(
    api_key=os.environ["KYRA_API_KEY"],
    agent_id="refund-agent-v1",
    fail_open=False,
    mode="enforce",
)
```

---

## Wrapping tools

Call `governor.wrap(tools)` after building your tool list. Use the **returned** tools in your agent so every invocation goes through Kyra first.

### Generic / CrewAI / Alchemyst

```python
governed_tools = governor.wrap(tools)
# Use governed_tools in your agent
```

### LangChain (with telemetry callback)

When using LangChain, pass `framework="langchain"` to get governed tools plus a callback that records LLM and tool activity (including agent trace with thinking when the model returns it) for Kyra:

```python
governed_tools, callback = governor.wrap(tools, framework="langchain")
agent = create_react_agent(llm=llm, tools=governed_tools, prompt=prompt)
# When using AgentExecutor:
executor = AgentExecutor(agent=agent, tools=governed_tools, callbacks=[callback])
```

### LangGraph

```python
# Wrap the tool node so all tool calls are governed
governed_tool_node = governor.wrap(tool_node, framework="langgraph")
# Use governed_tool_node in your graph
```

---

## Governance context (per run)

Set a governance context at the start of each run so all Kyra evaluations in that run share the same trace, session, and intent. **Trace ID** can be user-provided (e.g. from your context) or omitted so Kyra auto-generates it; the decision returned from each evaluate always includes the trace ID used.

### From user message

```python
from kyra_sdk import GovernanceContext, set_context, get_context

ctx = GovernanceContext.from_human_message(
    "Refund order 123 if policy allows",
    root_agent_id="refund-agent-v1",
)
set_context(ctx)

# Run your agent; Kyra will use this context for every tool evaluate
result = await agent.ainvoke({"input": "..."})

# Optional: read current context
current = get_context()
if current:
    print("Trace ID:", current.trace_id, "Session ID:", current.session_id)
```

### From agent spawn (multi-agent)

When an orchestrator delegates to a sub-agent, create a child context so Kyra can link the sub-agent’s trace to the parent:

```python
parent = GovernanceContext.from_human_message(
    "Refund order 123",
    root_agent_id="orchestrator-agent",
)
child = GovernanceContext.from_agent_spawn(parent, child_agent_id="refund-agent-v1")
set_context(child)

# Run the sub-agent; Kyra will receive parentTraceId / parentAgentId
result = await refund_agent.ainvoke(...)
```

---

## Agent registration (optional)

You can register your agent with Kyra so the server associates evaluate requests with a known agent (and optional policies):

```python
agent_id = governor.register_agent(
    agent_name="Refund Agent",
    system_prompt=system_prompt,
    tools=tools,
)
# agent_id is stored on the governor and used for subsequent evaluate calls
```

Optional: pass a list of `PolicyDocument` for server-side policies (see Policies section below).

---

## Error handling

When Kyra blocks an action (BLOCK, ESCALATE, or server unreachable with `fail_open=False`), wrapped tools raise **`ErrGovernanceBlock`**. Catch it to inform the user or trigger escalation:

```python
from kyra_sdk import ErrGovernanceBlock

try:
    result = await governed_tool.ainvoke({"order_id": "123", "amount": 50.0})
except ErrGovernanceBlock as e:
    print("Governance block:", e.msg)
    # e.msg is safe to log or show to users
```

### Exception hierarchy (optional)

For finer handling you can catch Kyra-specific exceptions:

```python
from kyra_sdk import (
    KyraBlockedException,
    KyraEscalationDeniedException,
    KyraReturnToUserException,
    KyraServerUnavailableException,
)

try:
    result = await governed_tool.ainvoke(params)
except KyraReturnToUserException as e:
    print("Missing parameters:", e.missing_parameters)
    # Ask the user to supply the missing fields
except KyraBlockedException:
    print("Action blocked by policy")
except KyraEscalationDeniedException:
    print("Escalation was denied")
except KyraServerUnavailableException:
    print("Kyra server unreachable (and fail_open=False)")
```

---

## Policies (reference)

**PolicyDocument** is used when registering agents or configuring server-side policies (policy_id, description, applies_to_tools, condition, action, and optional fields per server API). See the SDK source and Kyra server docs for full policy semantics.

---

## Summary

| Step | What to do |
|------|------------|
| 1 | Create `KyraGovernor(api_key=..., agent_id=..., fail_open=...)` |
| 2 | Wrap tools with `governor.wrap(tools)` or `governor.wrap(tools, framework="langchain")` |
| 3 | For each run: `ctx = GovernanceContext.from_human_message(msg, root_agent_id)` and `set_context(ctx)` |
| 4 | Run your agent with the governed tools |
| 5 | Catch `ErrGovernanceBlock` when an action is blocked |

Optional: call `register_agent()` at startup, and use `GovernanceContext.from_agent_spawn(parent, child_agent_id)` when delegating from an orchestrator to a sub-agent.
