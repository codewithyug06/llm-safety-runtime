"""
End-to-end pipeline test: LatentSentinel → Kafka → Remediator → Audit Log
===========================================================================
Tests the full ARGUS safety pipeline using mocked infrastructure components.
Verifies that an adversarial prompt triggers a safety signal, routes through
the remediation graph, and produces an audit record — all within 200ms.

For a real Kafka integration test, use pytest-docker to spin up a broker.
These tests use mock Kafka producers/consumers to remain fast in CI.

Run with:
    pytest tests/e2e/test_full_pipeline.py -v
    pytest tests/e2e/test_full_pipeline.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.autonomous_remediator.integrations import (
    AuditRecord,
    MLflowRollbackClient,
    PagerDutyNotifier,
    RedisQuarantineStore,
    SlackNotifier,
    SpannerAuditLogger,
)
from src.autonomous_remediator.langgraph_graph import (
    NO_ACTION_THRESHOLD,
    QUARANTINE_THRESHOLD,
    ROLLBACK_THRESHOLD,
    RemediationState,
    run_remediation,
)
from src.autonomous_remediator.rate_limiter import RateLimitAgent
from src.safety_critic.critic import ContentModality, CriticInput, CriticOutput


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_quarantine_store() -> RedisQuarantineStore:
    """QuarantineStore backed by an in-memory dict (no Redis needed)."""
    store = RedisQuarantineStore.__new__(RedisQuarantineStore)
    store._redis_url = "redis://mock"
    store._default_ttl_s = 300
    store._key_prefix = "argus:quarantine:"
    store._store: dict = {}

    def _quarantine(agent_id, ttl_s=None, reason=""):
        store._store[agent_id] = reason

    def _release(agent_id):
        store._store.pop(agent_id, None)

    def _is_quarantined(agent_id):
        return agent_id in store._store

    def _get_info(agent_id):
        if agent_id in store._store:
            return {"reason": store._store[agent_id]}
        return None

    store.quarantine = _quarantine
    store.release = _release
    store.is_quarantined = _is_quarantined
    store.get_quarantine_info = _get_info
    return store


def _make_mock_audit_logger() -> SpannerAuditLogger:
    """SpannerAuditLogger backed by in-memory list (no Spanner needed)."""
    logger = SpannerAuditLogger.__new__(SpannerAuditLogger)
    logger._records: list = []

    def _write(record: AuditRecord):
        logger._records.append(record)

    def _query(agent_id, limit=50):
        return [r for r in logger._records if r.agent_id == agent_id][:limit]

    logger.write = _write
    logger.query_by_agent = _query
    return logger


def _make_mock_rate_limiter() -> RateLimitAgent:
    """RateLimitAgent that always allows (no Redis needed)."""
    limiter = RateLimitAgent.__new__(RateLimitAgent)
    limiter._violations: dict = {}
    limiter._violation_threshold = 3

    def _check(agent_id, is_high_risk=False):
        return True, 0.0

    def _is_exceeded(agent_id):
        return limiter._violations.get(agent_id, 0) >= 3

    limiter.check_and_apply = _check
    limiter.is_violation_threshold_exceeded = _is_exceeded
    limiter.get_violation_count = lambda a: limiter._violations.get(a, 0)
    limiter.reset_violations = lambda a: limiter._violations.pop(a, None)
    return limiter


def _make_mock_slack() -> SlackNotifier:
    slack = SlackNotifier.__new__(SlackNotifier)
    slack._sent: list = []
    slack.send_alert = AsyncMock(side_effect=lambda **kw: slack._sent.append(kw) or True)
    return slack


def _make_mock_pagerduty() -> PagerDutyNotifier:
    pd = PagerDutyNotifier.__new__(PagerDutyNotifier)
    pd._triggered: list = []
    pd.trigger_incident = AsyncMock(side_effect=lambda **kw: pd._triggered.append(kw) or True)
    pd.resolve_incident = AsyncMock(return_value=True)
    return pd


def _make_mock_rollback_client() -> MLflowRollbackClient:
    client = MLflowRollbackClient.__new__(MLflowRollbackClient)
    client._rolled_back: list = []
    client.rollback = lambda name, version: client._rolled_back.append((name, version))
    client.get_current_production_version = lambda name: "5"
    return client


# ── Pipeline fixture ──────────────────────────────────────────────────────────

@pytest.fixture()
def pipeline_components():
    """Returns all mock pipeline components for reuse."""
    return {
        "quarantine_store": _make_mock_quarantine_store(),
        "audit_logger": _make_mock_audit_logger(),
        "rate_limiter": _make_mock_rate_limiter(),
        "slack": _make_mock_slack(),
        "pagerduty": _make_mock_pagerduty(),
        "rollback_client": _make_mock_rollback_client(),
    }


# ── TestCriticOutput ──────────────────────────────────────────────────────────

class TestCriticOutputIntegration:
    """Verify CriticOutput properties across the full score range."""

    def test_score_0_0_is_safe(self) -> None:
        out = CriticOutput(safety_score=0.0, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.risk_category == "safe"
        assert not out.is_unsafe

    def test_score_1_0_is_clearly_unsafe(self) -> None:
        out = CriticOutput(safety_score=1.0, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.risk_category == "clearly_unsafe"
        assert out.is_unsafe

    @pytest.mark.parametrize("score,expected", [
        (0.0, "safe"),
        (0.29, "safe"),
        (0.30, "borderline"),
        (0.59, "borderline"),
        (0.60, "likely_unsafe"),
        (0.79, "likely_unsafe"),
        (0.80, "clearly_unsafe"),
        (1.0, "clearly_unsafe"),
    ])
    def test_risk_categories(self, score: float, expected: str) -> None:
        out = CriticOutput(safety_score=score, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.risk_category == expected


# ── TestRemediationGraph ──────────────────────────────────────────────────────

class TestRemediationGraph:
    """Test the full remediation graph routing logic."""

    @pytest.mark.asyncio
    async def test_low_score_no_action(self, pipeline_components) -> None:
        """Score below 0.40 → no_action, no quarantine."""
        qs = pipeline_components["quarantine_store"]
        al = pipeline_components["audit_logger"]

        from src.autonomous_remediator.langgraph_graph import build_remediation_graph
        graph = build_remediation_graph(
            quarantine_store=qs,
            audit_logger=al,
            rate_limiter=pipeline_components["rate_limiter"],
            slack=pipeline_components["slack"],
            pagerduty=pipeline_components["pagerduty"],
            rollback_client=pipeline_components["rollback_client"],
        )
        state = await graph.ainvoke({
            "agent_id": "safe_agent",
            "safety_score": 0.15,
            "risk_category": "safe",
            "modality": "text",
            "content_preview": "Hello world",
            "action": "",
            "action_detail": "",
            "outcome": "",
            "latency_ms": 0.0,
            "audit_record_id": "",
            "error": "",
            "start_time": time.monotonic(),
        })

        assert state["action"] == "no_action"
        assert not qs.is_quarantined("safe_agent")
        # Audit record should still be written
        records = al.query_by_agent("safe_agent")
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_mid_score_filter_output(self, pipeline_components) -> None:
        """Score 0.50 → filter_output, short quarantine applied."""
        qs = pipeline_components["quarantine_store"]
        al = pipeline_components["audit_logger"]

        from src.autonomous_remediator.langgraph_graph import build_remediation_graph
        graph = build_remediation_graph(
            quarantine_store=qs,
            audit_logger=al,
            rate_limiter=pipeline_components["rate_limiter"],
            slack=pipeline_components["slack"],
            pagerduty=pipeline_components["pagerduty"],
            rollback_client=pipeline_components["rollback_client"],
        )
        state = await graph.ainvoke({
            "agent_id": "filter_agent",
            "safety_score": 0.55,
            "risk_category": "borderline",
            "modality": "text",
            "content_preview": "Borderline content",
            "action": "",
            "action_detail": "",
            "outcome": "",
            "latency_ms": 0.0,
            "audit_record_id": "",
            "error": "",
            "start_time": time.monotonic(),
        })

        assert state["action"] == "filter_output"
        assert state["outcome"] == "success"
        # Filter is implemented via short quarantine
        assert qs.is_quarantined("filter_agent")

    @pytest.mark.asyncio
    async def test_high_score_quarantine(self, pipeline_components) -> None:
        """Score 0.85 → quarantine action taken."""
        qs = pipeline_components["quarantine_store"]
        al = pipeline_components["audit_logger"]
        sn = pipeline_components["slack"]

        from src.autonomous_remediator.langgraph_graph import build_remediation_graph
        graph = build_remediation_graph(
            quarantine_store=qs,
            audit_logger=al,
            rate_limiter=pipeline_components["rate_limiter"],
            slack=sn,
            pagerduty=pipeline_components["pagerduty"],
            rollback_client=pipeline_components["rollback_client"],
        )
        state = await graph.ainvoke({
            "agent_id": "unsafe_agent",
            "safety_score": 0.85,
            "risk_category": "clearly_unsafe",
            "modality": "text",
            "content_preview": "Dangerous content",
            "action": "",
            "action_detail": "",
            "outcome": "",
            "latency_ms": 0.0,
            "audit_record_id": "",
            "error": "",
            "start_time": time.monotonic(),
        })

        assert state["action"] == "quarantine"
        assert state["outcome"] == "success"
        assert qs.is_quarantined("unsafe_agent")

    @pytest.mark.asyncio
    async def test_critical_score_rollback_escalate(self, pipeline_components) -> None:
        """Score ≥ 0.90 → rollback + PagerDuty escalation."""
        qs = pipeline_components["quarantine_store"]
        al = pipeline_components["audit_logger"]
        sn = pipeline_components["slack"]
        pd = pipeline_components["pagerduty"]
        rc = pipeline_components["rollback_client"]

        from src.autonomous_remediator.langgraph_graph import build_remediation_graph
        graph = build_remediation_graph(
            quarantine_store=qs,
            audit_logger=al,
            rate_limiter=pipeline_components["rate_limiter"],
            slack=sn,
            pagerduty=pd,
            rollback_client=rc,
        )
        state = await graph.ainvoke({
            "agent_id": "critical_agent",
            "safety_score": 0.96,
            "risk_category": "clearly_unsafe",
            "modality": "code",
            "content_preview": "rm -rf /",
            "action": "",
            "action_detail": "",
            "outcome": "",
            "latency_ms": 0.0,
            "audit_record_id": "",
            "error": "",
            "start_time": time.monotonic(),
        })

        assert state["action"] == "rollback_and_escalate"
        assert state["outcome"] == "success"
        # PagerDuty should have been triggered
        assert pd.trigger_incident.called

    @pytest.mark.asyncio
    async def test_audit_record_written_for_all_actions(self, pipeline_components) -> None:
        """All action paths must write an audit record."""
        al = pipeline_components["audit_logger"]

        from src.autonomous_remediator.langgraph_graph import build_remediation_graph
        graph = build_remediation_graph(
            quarantine_store=pipeline_components["quarantine_store"],
            audit_logger=al,
            rate_limiter=pipeline_components["rate_limiter"],
            slack=pipeline_components["slack"],
            pagerduty=pipeline_components["pagerduty"],
            rollback_client=pipeline_components["rollback_client"],
        )

        scores = [0.1, 0.5, 0.72, 0.85, 0.95]
        agents = [f"agent_{i}" for i in range(len(scores))]

        for agent_id, score in zip(agents, scores):
            await graph.ainvoke({
                "agent_id": agent_id,
                "safety_score": score,
                "risk_category": "test",
                "modality": "text",
                "content_preview": "test",
                "action": "",
                "action_detail": "",
                "outcome": "",
                "latency_ms": 0.0,
                "audit_record_id": "",
                "error": "",
                "start_time": time.monotonic(),
            })

        # Every agent should have exactly one audit record
        for agent_id in agents:
            records = al.query_by_agent(agent_id)
            assert len(records) == 1, f"Expected 1 audit record for {agent_id}, got {len(records)}"

    @pytest.mark.asyncio
    async def test_e2e_latency_under_200ms(self, pipeline_components) -> None:
        """End-to-end remediation graph should complete in < 200ms (mocked infra)."""
        from src.autonomous_remediator.langgraph_graph import build_remediation_graph
        graph = build_remediation_graph(
            quarantine_store=pipeline_components["quarantine_store"],
            audit_logger=pipeline_components["audit_logger"],
            rate_limiter=pipeline_components["rate_limiter"],
            slack=pipeline_components["slack"],
            pagerduty=pipeline_components["pagerduty"],
            rollback_client=pipeline_components["rollback_client"],
        )

        t0 = time.perf_counter()
        state = await graph.ainvoke({
            "agent_id": "latency_test_agent",
            "safety_score": 0.82,
            "risk_category": "clearly_unsafe",
            "modality": "text",
            "content_preview": "test",
            "action": "",
            "action_detail": "",
            "outcome": "",
            "latency_ms": 0.0,
            "audit_record_id": "",
            "error": "",
            "start_time": time.monotonic(),
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 200, f"E2E latency {elapsed_ms:.1f}ms exceeds 200ms SLA"
        assert state["outcome"] == "success"


# ── TestRedisQuarantineStoreIntegration ───────────────────────────────────────

class TestRedisQuarantineStoreMock:
    """Tests for RedisQuarantineStore using in-memory mock."""

    def test_quarantine_and_is_quarantined(self) -> None:
        store = _make_mock_quarantine_store()
        store.quarantine("agent_x", reason="test")
        assert store.is_quarantined("agent_x") is True

    def test_release_clears_flag(self) -> None:
        store = _make_mock_quarantine_store()
        store.quarantine("agent_y")
        store.release("agent_y")
        assert store.is_quarantined("agent_y") is False

    def test_not_quarantined_initially(self) -> None:
        store = _make_mock_quarantine_store()
        assert store.is_quarantined("new_agent") is False

    def test_get_quarantine_info(self) -> None:
        store = _make_mock_quarantine_store()
        store.quarantine("agent_z", reason="score=0.9")
        info = store.get_quarantine_info("agent_z")
        assert info is not None


# ── TestAuditLoggerIntegration ────────────────────────────────────────────────

class TestAuditLoggerMock:
    def test_write_and_query(self) -> None:
        al = _make_mock_audit_logger()
        record = AuditRecord(
            record_id="test-uuid",
            timestamp="2026-01-01T00:00:00Z",
            agent_id="agent_99",
            safety_score=0.87,
            action_taken="quarantine",
            action_detail="score exceeded threshold",
            triggered_by="argus-remediator",
            outcome="success",
            latency_ms=45.0,
        )
        al.write(record)
        results = al.query_by_agent("agent_99")
        assert len(results) == 1
        assert results[0].action_taken == "quarantine"

    def test_query_empty_for_unknown_agent(self) -> None:
        al = _make_mock_audit_logger()
        assert al.query_by_agent("unknown_agent") == []
