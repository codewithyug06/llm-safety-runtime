"""
MOD-06: AutonomousRemediator LangGraph StateGraph
==================================================
Defines the full multi-agent remediation graph using LangGraph's StateGraph.

Graph topology:
    triage → route → [filter_output | quarantine | rollback | rate_limit | escalate]
                   → audit_log

Routing logic (from configs/autonomous_remediator.yaml):
    score < 0.40  → no_action (pass-through)
    0.40–0.65     → filter_output (soft filtering)
    0.65–0.80     → rate_limit   (slow down agent)
    0.80–0.90     → quarantine   (block agent temporarily)
    ≥ 0.90        → rollback + escalate (hard reset + human alert)

Each node is an async function that receives the RemediationState TypedDict
and returns updated state. The conditional edge after 'route' reads the
'action' field to dispatch to the correct node.

Integration points:
    - RedisQuarantineStore → quarantine node
    - MLflowRollbackClient → rollback node
    - RateLimitAgent       → rate_limit node
    - SlackNotifier        → escalate node
    - PagerDutyNotifier    → escalate node (critical only)
    - SpannerAuditLogger   → audit_log node (all paths)
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, TypedDict

import structlog

from src.autonomous_remediator.integrations import (
    AuditRecord,
    MLflowRollbackClient,
    PagerDutyNotifier,
    RedisQuarantineStore,
    SlackNotifier,
    SpannerAuditLogger,
)
from src.autonomous_remediator.rate_limiter import RateLimitAgent
from src.exceptions import RemediationError

logger = structlog.get_logger(__name__)

# ── State TypedDict ───────────────────────────────────────────────────────────

class RemediationState(TypedDict):
    """Shared state flowing through the remediation graph.

    Fields:
        agent_id: ID of the agent being remediated.
        safety_score: Latest safety score [0.0, 1.0].
        risk_category: Human-readable risk label.
        modality: Content modality (text/code/image/structured).
        content_preview: First 200 chars of flagged content.
        action: Remediation action decided by route node.
        action_detail: Extra context about the action.
        outcome: 'success' or 'failure'.
        latency_ms: Elapsed time since graph started (ms).
        audit_record_id: UUID of the written audit record.
        error: Error message if any node failed.
        start_time: monotonic clock at graph entry.
    """

    agent_id: str
    safety_score: float
    risk_category: str
    modality: str
    content_preview: str
    action: str
    action_detail: str
    outcome: str
    latency_ms: float
    audit_record_id: str
    error: str
    start_time: float


# ── Thresholds ────────────────────────────────────────────────────────────────

NO_ACTION_THRESHOLD: float = 0.40
FILTER_THRESHOLD: float = 0.65
QUARANTINE_THRESHOLD: float = 0.80
ROLLBACK_THRESHOLD: float = 0.90


# ── Node implementations ──────────────────────────────────────────────────────

async def triage_node(state: RemediationState) -> RemediationState:
    """Log receipt of a remediation request and set start time.

    Args:
        state: Current remediation state.

    Returns:
        Updated state with start_time set.
    """
    logger.info(
        "triage_start",
        agent_id=state["agent_id"],
        safety_score=f"{state['safety_score']:.3f}",
        modality=state["modality"],
    )
    return {**state, "start_time": time.monotonic()}


async def route_node(state: RemediationState) -> RemediationState:
    """Decide remediation action based on safety score thresholds.

    Args:
        state: Current remediation state.

    Returns:
        Updated state with 'action' field set.
    """
    score = state["safety_score"]

    if score < NO_ACTION_THRESHOLD:
        action = "no_action"
    elif score < FILTER_THRESHOLD:
        action = "filter_output"
    elif score < QUARANTINE_THRESHOLD:
        action = "rate_limit"
    elif score < ROLLBACK_THRESHOLD:
        action = "quarantine"
    else:
        action = "rollback_and_escalate"

    logger.info(
        "route_decision",
        agent_id=state["agent_id"],
        safety_score=f"{score:.3f}",
        action=action,
    )
    return {**state, "action": action}


async def filter_output_node(state: RemediationState, quarantine_store: RedisQuarantineStore) -> RemediationState:
    """Apply soft output filtering — flag agent for downstream filtering.

    Args:
        state: Current remediation state.
        quarantine_store: Redis store to set filter flag.

    Returns:
        Updated state with outcome.
    """
    agent_id = state["agent_id"]
    try:
        # Set a short-lived filter flag (60s) — softer than full quarantine
        quarantine_store.quarantine(
            agent_id,
            ttl_s=60,
            reason=f"Soft filter: score={state['safety_score']:.3f}",
        )
        logger.info("filter_applied", agent_id=agent_id, score=state["safety_score"])
        return {
            **state,
            "outcome": "success",
            "action_detail": f"Output filter applied for 60s (score={state['safety_score']:.3f})",
        }
    except Exception as exc:
        logger.error("filter_failed", agent_id=agent_id, error=str(exc))
        return {**state, "outcome": "failure", "error": str(exc)}


async def rate_limit_node(
    state: RemediationState,
    rate_limiter: RateLimitAgent,
) -> RemediationState:
    """Apply rate limiting to the agent.

    Args:
        state: Current remediation state.
        rate_limiter: RateLimitAgent instance.

    Returns:
        Updated state with outcome.
    """
    agent_id = state["agent_id"]
    allowed, wait_ms = rate_limiter.check_and_apply(agent_id, is_high_risk=True)
    exceeded = rate_limiter.is_violation_threshold_exceeded(agent_id)

    logger.info(
        "rate_limit_applied",
        agent_id=agent_id,
        allowed=allowed,
        wait_ms=wait_ms,
        violations_exceeded=exceeded,
    )
    return {
        **state,
        "outcome": "success",
        "action_detail": f"Rate limited: allowed={allowed}, wait={wait_ms:.0f}ms, "
                         f"threshold_exceeded={exceeded}",
    }


async def quarantine_node(
    state: RemediationState,
    quarantine_store: RedisQuarantineStore,
    slack: SlackNotifier,
) -> RemediationState:
    """Quarantine agent — block output for default TTL (300s).

    Args:
        state: Current remediation state.
        quarantine_store: Redis quarantine store.
        slack: Slack notifier for alert dispatch.

    Returns:
        Updated state.
    """
    agent_id = state["agent_id"]
    try:
        quarantine_store.quarantine(
            agent_id,
            reason=f"Safety score {state['safety_score']:.3f} ≥ {QUARANTINE_THRESHOLD}",
        )
        await slack.send_alert(
            message=f"Agent `{agent_id}` quarantined (score={state['safety_score']:.3f})",
            severity="warning",
            agent_id=agent_id,
            extra_fields={"modality": state["modality"], "risk": state["risk_category"]},
        )
        logger.info("agent_quarantined", agent_id=agent_id, score=state["safety_score"])
        return {**state, "outcome": "success", "action_detail": "Agent quarantined for 300s"}
    except Exception as exc:
        logger.error("quarantine_node_failed", agent_id=agent_id, error=str(exc))
        return {**state, "outcome": "failure", "error": str(exc)}


async def rollback_and_escalate_node(
    state: RemediationState,
    rollback_client: MLflowRollbackClient,
    quarantine_store: RedisQuarantineStore,
    slack: SlackNotifier,
    pagerduty: PagerDutyNotifier,
    model_name: str = "argus-safety-critic",
    rollback_version: int = 1,
) -> RemediationState:
    """Full rollback + human escalation for critical safety violations.

    Args:
        state: Current remediation state.
        rollback_client: MLflow rollback client.
        quarantine_store: Redis quarantine store.
        slack: Slack notifier.
        pagerduty: PagerDuty notifier.
        model_name: Model to roll back.
        rollback_version: Version to roll back to.

    Returns:
        Updated state.
    """
    agent_id = state["agent_id"]
    details = []

    # 1. Hard quarantine (600s)
    try:
        quarantine_store.quarantine(
            agent_id,
            ttl_s=600,
            reason=f"CRITICAL: score={state['safety_score']:.3f} ≥ {ROLLBACK_THRESHOLD}",
        )
        details.append("quarantined(600s)")
    except Exception as exc:
        logger.error("rollback_quarantine_failed", error=str(exc))

    # 2. Model rollback
    try:
        rollback_client.rollback(model_name, rollback_version)
        details.append(f"model_rolled_back(v{rollback_version})")
    except Exception as exc:
        logger.error("model_rollback_failed", error=str(exc))
        details.append(f"rollback_failed:{exc}")

    # 3. Slack + PagerDuty escalation
    msg = (
        f":rotating_light: *CRITICAL*: Agent `{agent_id}` safety score "
        f"{state['safety_score']:.3f} exceeds rollback threshold. "
        f"Actions: {', '.join(details)}"
    )
    await slack.send_alert(
        message=msg,
        severity="critical",
        agent_id=agent_id,
        extra_fields={"score": str(state["safety_score"]), "modality": state["modality"]},
    )
    await pagerduty.trigger_incident(
        agent_id=agent_id,
        summary=f"ARGUS CRITICAL: Agent {agent_id} safety={state['safety_score']:.3f}",
        severity="critical",
        details={"safety_score": state["safety_score"], "actions": details},
    )

    logger.warning(
        "rollback_and_escalate_complete",
        agent_id=agent_id,
        score=state["safety_score"],
        actions=details,
    )

    return {
        **state,
        "outcome": "success",
        "action_detail": f"Rollback+escalate: {', '.join(details)}",
    }


async def audit_log_node(
    state: RemediationState,
    audit_logger: SpannerAuditLogger,
) -> RemediationState:
    """Write audit record to Spanner.

    Args:
        state: Final remediation state.
        audit_logger: Spanner audit logger.

    Returns:
        Updated state with audit_record_id set.
    """
    elapsed_ms = (time.monotonic() - state.get("start_time", time.monotonic())) * 1000
    record_id = str(uuid.uuid4())

    record = AuditRecord(
        record_id=record_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_id=state["agent_id"],
        safety_score=state["safety_score"],
        action_taken=state["action"],
        action_detail=state.get("action_detail", ""),
        triggered_by="argus-remediator",
        outcome=state.get("outcome", "unknown"),
        latency_ms=elapsed_ms,
    )

    try:
        audit_logger.write(record)
    except Exception as exc:
        logger.error("audit_log_failed", error=str(exc), record_id=record_id)

    return {**state, "audit_record_id": record_id, "latency_ms": elapsed_ms}


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_remediation_graph(
    quarantine_store: Optional[RedisQuarantineStore] = None,
    rollback_client: Optional[MLflowRollbackClient] = None,
    rate_limiter: Optional[RateLimitAgent] = None,
    slack: Optional[SlackNotifier] = None,
    pagerduty: Optional[PagerDutyNotifier] = None,
    audit_logger: Optional[SpannerAuditLogger] = None,
    model_name: str = "argus-safety-critic",
    rollback_version: int = 1,
) -> Any:
    """Build and compile the LangGraph remediation StateGraph.

    Args:
        quarantine_store: Redis quarantine store (created with defaults if None).
        rollback_client: MLflow rollback client (defaults if None).
        rate_limiter: Rate limit agent (defaults if None).
        slack: Slack notifier (no-op if None).
        pagerduty: PagerDuty notifier (no-op if None).
        audit_logger: Spanner audit logger (no-op if None).
        model_name: Model name for rollback.
        rollback_version: Version to roll back to.

    Returns:
        Compiled LangGraph runnable.

    Raises:
        ImportError: If langgraph is not installed.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        raise ImportError("Run: pip install langgraph>=0.1.0")

    # Use defaults if not provided
    qs = quarantine_store or RedisQuarantineStore()
    rc = rollback_client or MLflowRollbackClient()
    rl = rate_limiter or RateLimitAgent()
    sn = slack or SlackNotifier()
    pd = pagerduty or PagerDutyNotifier()
    al = audit_logger or SpannerAuditLogger()

    # Build graph
    graph = StateGraph(RemediationState)

    # Add nodes — use async closures so dependency-injected async nodes are properly awaited.
    graph.add_node("triage", triage_node)
    graph.add_node("route", route_node)

    async def _filter_output(s: RemediationState) -> RemediationState:
        return await filter_output_node(s, qs)

    async def _rate_limit(s: RemediationState) -> RemediationState:
        return await rate_limit_node(s, rl)

    async def _quarantine(s: RemediationState) -> RemediationState:
        return await quarantine_node(s, qs, sn)

    async def _rollback_and_escalate(s: RemediationState) -> RemediationState:
        return await rollback_and_escalate_node(s, rc, qs, sn, pd, model_name, rollback_version)

    async def _audit_log(s: RemediationState) -> RemediationState:
        return await audit_log_node(s, al)

    graph.add_node("filter_output", _filter_output)
    graph.add_node("rate_limit", _rate_limit)
    graph.add_node("quarantine", _quarantine)
    graph.add_node("rollback_and_escalate", _rollback_and_escalate)
    graph.add_node("audit_log", _audit_log)

    # Edges
    graph.set_entry_point("triage")
    graph.add_edge("triage", "route")

    # Conditional routing based on action field
    graph.add_conditional_edges(
        "route",
        lambda s: s["action"],
        {
            "no_action": "audit_log",
            "filter_output": "filter_output",
            "rate_limit": "rate_limit",
            "quarantine": "quarantine",
            "rollback_and_escalate": "rollback_and_escalate",
        },
    )

    # All action nodes flow to audit_log
    for node in ["filter_output", "rate_limit", "quarantine", "rollback_and_escalate"]:
        graph.add_edge(node, "audit_log")

    graph.add_edge("audit_log", END)

    return graph.compile()


async def run_remediation(
    agent_id: str,
    safety_score: float,
    risk_category: str = "borderline",
    modality: str = "text",
    content_preview: str = "",
    graph: Optional[Any] = None,
) -> RemediationState:
    """Convenience function to run remediation for a single safety signal.

    Args:
        agent_id: Agent to remediate.
        safety_score: Safety score [0.0, 1.0].
        risk_category: Human-readable risk category.
        modality: Content modality.
        content_preview: First 200 chars of flagged content.
        graph: Pre-compiled graph (builds a default one if None).

    Returns:
        Final RemediationState after graph execution.

    Example:
        state = await run_remediation("agent_007", safety_score=0.85)
        print(state["action"], state["outcome"], state["latency_ms"])
    """
    if graph is None:
        graph = build_remediation_graph()

    initial_state: RemediationState = {
        "agent_id": agent_id,
        "safety_score": safety_score,
        "risk_category": risk_category,
        "modality": modality,
        "content_preview": content_preview[:200],
        "action": "",
        "action_detail": "",
        "outcome": "",
        "latency_ms": 0.0,
        "audit_record_id": "",
        "error": "",
        "start_time": time.monotonic(),
    }

    final_state = await graph.ainvoke(initial_state)

    logger.info(
        "remediation_complete",
        agent_id=agent_id,
        action=final_state["action"],
        outcome=final_state["outcome"],
        latency_ms=f"{final_state['latency_ms']:.1f}",
    )

    return final_state
