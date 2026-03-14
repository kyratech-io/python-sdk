import pytest
from kyra_sdk import KyraGovernor


def test_wrap_returns_same_count():
    governor = KyraGovernor(api_key="kyra_sk_test")
    tools = [
        type("Tool", (), {"name": "t1", "description": "d1", "args_schema": None})(),
        type("Tool", (), {"name": "t2", "description": "d2", "args_schema": None})(),
    ]
    wrapped = governor.wrap(tools)
    assert len(wrapped) == 2


def test_wrap_generic_returns_list():
    governor = KyraGovernor(api_key="kyra_sk_test")
    tools = [
        type("Tool", (), {"name": "g1", "description": "Generic", "run": lambda **kw: "ok", "invoke": None})(),
    ]
    wrapped = governor.wrap(tools, framework="generic")
    assert isinstance(wrapped, list)
    assert len(wrapped) == 1
    assert wrapped[0].name == "g1"
