import pytest
import respx
import httpx
from langchain_kyra import KyraGovernor


@respx.mock
def test_allow_passes_through():
    respx.post("https://api.kyra.dev/v1/evaluate").mock(return_value=httpx.Response(
        200, json={"outcome": "ALLOW", "traceId": "t1", "evaluationId": "e1",
                   "orgId": "org1", "mode": "ENFORCE", "tier": "T0", "evaluationMs": 5,
                   "gateResults": []}
    ))
    governor = KyraGovernor(api_key="kyra_sk_test")
    ok, block_reason = governor.evaluate("delete_user", "Deletes a user", {"user_id": "123"})
    assert ok is True
    assert block_reason == ""


@respx.mock
def test_block_returns_ok_false():
    respx.post("https://api.kyra.dev/v1/evaluate").mock(return_value=httpx.Response(
        200, json={"outcome": "BLOCK", "blockReason": "Exceeds row limit",
                   "traceId": "t1", "evaluationId": "e1", "orgId": "org1",
                   "mode": "ENFORCE", "tier": "T3", "evaluationMs": 10, "gateResults": []}
    ))
    governor = KyraGovernor(api_key="kyra_sk_test")
    ok, block_reason = governor.evaluate("bulk_delete", "Deletes many records", {"table": "orders"})
    assert ok is False
    assert "Exceeds row limit" in block_reason


def test_fail_oKyra_on_timeout():
    governor = AAIGovernor(api_key="kyra_sk_test",
                           server_url="http://doesnotexist.invalid",
                           timeout_ms=100, fail_open=True)
    ok, _ = governor.evaluate("any_tool", "desc", {})
    assert ok is True


def test_shadow_mode_returns_allow():
    with respx.mock:
        respx.post("https://api.kyra.dev/v1/evaluate").mock(return_value=httpx.Response(
            200, json={"outcome": "ALLOW", "shadowOutcome": "BLOCK", "mode": "SHADOW",
                       "traceId": "t1", "evaluationId": "e1", "orgId": "org1",
                       "tier": "T3", "evaluationMs": 5, "gateResults": []}
        ))
        governor = KyraGovernor(api_key="kyra_sk_test")
        ok, _ = governor.evaluate("bulk_delete", "desc", {})
        assert ok is True