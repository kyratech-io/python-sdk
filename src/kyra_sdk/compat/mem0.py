"""
Mem0 compatibility helpers.
Mem0 (memory layer) + Kyra (authorization layer) are complementary.
This module provides a joint quickstart helper.
"""

from typing import Any, List, Optional
from ..governor import KyraGovernor


def create_governed_agent(
    tools: List[Any],
    api_key: str,
    mem0_client: Optional[Any] = None,
    **governor_kwargs,
) -> tuple:
    """
    Wrap tools with Kyra governance. Optionally pass a Mem0 client.
    Returns (governed_tools, governor).

    Stack this provides:
      Mem0  → persistent memory across conversations
      Kyra   → what the agent is allowed to DO with that memory

    Example:
        from mem0 import MemoryClient
        from kyra_sdk.compat.mem0 import create_governed_agent

        governed_tools, governor = create_governed_agent(
            tools=my_tools,
            api_key="kyra_sk_...",
            mem0_client=MemoryClient(api_key="m0-..."),
        )
        # Build your agent with governed_tools and mem0_client
    """
    governor = KyraGovernor(api_key=api_key, **governor_kwargs)
    governed_tools = governor.wrap(tools)
    return governed_tools, governor
