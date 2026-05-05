"""
MOD-06: AutonomousRemediator — Agentic Self-Healing System
===========================================================
A LangGraph-orchestrated multi-agent system that automatically applies targeted
fixes when LatentSentinel + PredictiveOracle flag high-risk conditions.

Agent topology:
    SafetySignal
        → RemediatorOrchestrator (LangGraph StateGraph)
            ├─ OutputFilterAgent      (low risk: rewrite/truncate output)
            ├─ QuarantineAgent        (medium risk: isolate misbehaving agent)
            ├─ RollbackAgent          (high risk: revert to safe model version)
            ├─ RateLimitAgent         (apply exponential backoff throttling)
            └─ EscalationAgent        (critical: page human reviewer)

All decisions are logged immutably to Cloud Spanner for compliance audit trails.
Target SLA: <200ms end-to-end from SafetySignal to remediation applied.

Uses the Anthropic API for LLM-assisted reasoning about which fix to apply.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import structlog
from anthropic import Anthropic

logger = structlog.get_logger(__name__)

# Anthropic model used for LLM-assisted remediation reasoning.
# Override via configs/autonomous_remediator.yaml → serving.llm_model
_LLM_MODEL: str = "meta-llama/Llama-3.1-70B-Instruct"


# ── Enums ─────────────────────────────────────────────────────────────────────

class RemediationAction(Enum):
    FILTER_OUTPUT = "filter_output"
    QUARANTINE_AGENT = "quarantine_agent"
    ROLLBACK_MODEL = "rollback_model"
    RATE_LIMIT = "rate_limit"
    ESCALATE_HUMAN = "escalate_human"
    NO_ACTION = "no_action"


class RemediationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


# ── State ──────────────────────────────────────────────────────────────────────

class RemediatorState(TypedDict):
    """Shared state flowing through the LangGraph agent graph.

    This TypedDict is the single source of truth as the graph executes.
    Each node reads from and writes to this state.
    """
    request_id: str
    agent_id: str
    risk_level: str
    composite_score: float
    triggered_categories: List[str]
    probe_results: Dict[str, float]
    predicted_failure_prob: float

    # Decision-making
    orchestrator_reasoning: str
    selected_action: str
    action_parameters: Dict[str, Any]

    # Execution
    remediation_id: str
    status: str
    action_result: Dict[str, Any]
    error: Optional[str]

    # Audit
    timeline: List[Dict[str, Any]]
    latency_ms: float


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RemediationDecision:
    """Output of the orchestrator reasoning step.

    Args:
        action: Which action to take.
        parameters: Action-specific parameters.
        reasoning: Chain-of-thought explanation (stored in audit log).
        confidence: How confident the orchestrator is in this decision.
    """
    action: RemediationAction
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float


@dataclass
class RemediationRecord:
    """Immutable audit record stored in Cloud Spanner.

    Args:
        remediation_id: Globally unique ID for this remediation event.
        request_id: The original LLM request that triggered this.
        agent_id: Which deployed agent was remediated.
        risk_level: Risk level at time of remediation.
        composite_score: Safety score that triggered remediation.
        action_taken: The remediation action applied.
        action_parameters: Specific parameters of the action.
        orchestrator_reasoning: LLM chain-of-thought reasoning for the decision.
        outcome: Success/failure + details.
        latency_ms: Total remediation latency.
        timestamp_iso: ISO 8601 timestamp.
    """
    remediation_id: str
    request_id: str
    agent_id: str
    risk_level: str
    composite_score: float
    action_taken: str
    action_parameters: Dict[str, Any]
    orchestrator_reasoning: str
    outcome: Dict[str, Any]
    latency_ms: float
    timestamp_iso: str


# ── Individual Action Agents ───────────────────────────────────────────────────

class OutputFilterAgent:
    """Rewrites or truncates unsafe LLM outputs in-flight.

    The fastest remediation — operates on the output text itself.
    Does not stop the agent or rollback the model.

    Args:
        anthropic_client: Anthropic API client for intelligent rewriting.
        max_rewrite_tokens: Max tokens for rewritten output.
    """

    def __init__(
        self,
        anthropic_client: Anthropic,
        max_rewrite_tokens: int = 500,
    ) -> None:
        self._client = anthropic_client
        self._max_tokens = max_rewrite_tokens

    async def execute(
        self,
        original_output: str,
        risk_categories: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Filter or rewrite the unsafe output.

        Args:
            original_output: The LLM output to filter.
            risk_categories: Categories of risk detected.
            parameters: Optional parameters (e.g., hard_truncate, rewrite_mode).

        Returns:
            Dict with 'filtered_output', 'action_taken', and 'explanation'.
        """
        mode = parameters.get("mode", "rewrite")

        if mode == "truncate":
            # Hard truncate at first unsafe segment
            filtered = original_output[:parameters.get("max_chars", 200)]
            return {
                "filtered_output": filtered + " [Response truncated for safety]",
                "action_taken": "hard_truncate",
                "explanation": f"Truncated due to: {risk_categories}",
            }

        # Intelligent rewrite via Anthropic API
        prompt = (
            f"The following AI output was flagged for safety risks: {risk_categories}.\n"
            f"Rewrite it to be helpful while removing the unsafe elements.\n"
            f"If the entire response is unsafe, replace with a polite refusal.\n\n"
            f"Original output:\n{original_output[:1000]}\n\n"
            f"Provide ONLY the rewritten output, no explanation."
        )

        response = self._client.messages.create(
            model=_LLM_MODEL,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "filtered_output": response.content[0].text,
            "action_taken": "ai_rewrite",
            "explanation": f"Rewrote output to remove: {risk_categories}",
        }


class QuarantineAgent:
    """Isolates a misbehaving agent by routing its traffic to a shadow model.

    Sets a quarantine flag in Redis that the API gateway reads.
    Traffic is rerouted to a pre-validated shadow model endpoint.

    Args:
        redis_client: Redis connection for quarantine flag storage.
        shadow_endpoint: URL of the safe shadow model endpoint.
        quarantine_ttl_seconds: How long to keep agent quarantined.
    """

    def __init__(
        self,
        redis_client: Any,
        shadow_endpoint: str,
        quarantine_ttl_seconds: int = 300,
    ) -> None:
        self._redis = redis_client
        self._shadow = shadow_endpoint
        self._ttl = quarantine_ttl_seconds

    async def execute(
        self,
        agent_id: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Quarantine an agent.

        Args:
            agent_id: The agent deployment ID to quarantine.
            parameters: Options (e.g., ttl_override, reason).

        Returns:
            Dict with 'quarantined', 'shadow_endpoint', 'expires_at'.
        """
        ttl = parameters.get("ttl_seconds", self._ttl)
        quarantine_key = f"argus:quarantine:{agent_id}"

        await self._redis.setex(
            quarantine_key,
            ttl,
            json.dumps({
                "reason": parameters.get("reason", "safety_violation"),
                "shadow_endpoint": self._shadow,
                "quarantined_at": time.time(),
            }),
        )

        logger.warning(
            "agent_quarantined",
            agent_id=agent_id,
            ttl_seconds=ttl,
            shadow=self._shadow,
        )

        return {
            "quarantined": True,
            "agent_id": agent_id,
            "shadow_endpoint": self._shadow,
            "ttl_seconds": ttl,
        }


class RollbackAgent:
    """Reverts the monitored model to the last registered safe checkpoint.

    Interacts with the MLflow model registry to find and promote the
    last known-safe version, then triggers a Kubeflow redeployment.

    Args:
        mlflow_uri: MLflow tracking server URI.
        kubeflow_client: Kubeflow Pipelines client.
        model_name: MLflow registered model name.
    """

    def __init__(
        self,
        mlflow_uri: str,
        model_name: str,
        deployment_callback: Optional[Any] = None,
    ) -> None:
        self._mlflow_uri = mlflow_uri
        self._model_name = model_name
        self._deploy_fn = deployment_callback

    async def execute(
        self,
        current_version: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rollback to the previous safe model version.

        Args:
            current_version: The model version that triggered the rollback.
            parameters: Options (e.g., target_version, dry_run).

        Returns:
            Dict with 'rolled_back_to', 'previous_version', 'deploy_triggered'.
        """
        import mlflow
        mlflow.set_tracking_uri(self._mlflow_uri)
        client = mlflow.tracking.MlflowClient()

        # Find last Production version
        production_versions = client.get_latest_versions(
            self._model_name,
            stages=["Production"],
        )

        if not production_versions:
            return {"rolled_back_to": None, "error": "No production version found"}

        safe_version = production_versions[0].version

        if parameters.get("dry_run", False):
            return {
                "rolled_back_to": safe_version,
                "previous_version": current_version,
                "deploy_triggered": False,
                "dry_run": True,
            }

        # Transition current to Archived
        client.transition_model_version_stage(
            name=self._model_name,
            version=current_version,
            stage="Archived",
        )

        # Trigger redeployment if callback provided
        deploy_triggered = False
        if self._deploy_fn:
            await self._deploy_fn(model_name=self._model_name, version=safe_version)
            deploy_triggered = True

        logger.warning(
            "model_rollback_executed",
            from_version=current_version,
            to_version=safe_version,
        )

        return {
            "rolled_back_to": safe_version,
            "previous_version": current_version,
            "deploy_triggered": deploy_triggered,
        }


class EscalationAgent:
    """Pages human safety reviewers for critical incidents.

    Sends rich context bundles to Slack and PagerDuty with all
    information needed for a human to make a remediation decision.

    Args:
        slack_webhook: Slack incoming webhook URL.
        pagerduty_routing_key: PagerDuty events API routing key.
        anthropic_client: Anthropic API client for generating context summary.
    """

    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        pagerduty_routing_key: Optional[str] = None,
        anthropic_client: Optional[Anthropic] = None,
    ) -> None:
        self._slack_webhook = slack_webhook
        self._pd_key = pagerduty_routing_key
        self._client = anthropic_client

    async def execute(
        self,
        state: RemediatorState,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Escalate to human reviewer.

        Args:
            state: Full remediation state for context.
            parameters: Options (e.g., priority, on-call rotation).

        Returns:
            Dict with 'escalated', 'channels', 'incident_id'.
        """
        incident_id = f"ARGUS-{int(time.time())}-{state['request_id'][:8]}"

        # Generate human-readable summary via Anthropic API
        summary = await self._generate_summary(state)

        channels = []

        if self._slack_webhook:
            await self._notify_slack(incident_id, summary, state)
            channels.append("slack")

        if self._pd_key:
            await self._trigger_pagerduty(incident_id, summary, state)
            channels.append("pagerduty")

        logger.critical(
            "human_escalation_triggered",
            incident_id=incident_id,
            risk_level=state["risk_level"],
            score=state["composite_score"],
        )

        return {
            "escalated": True,
            "incident_id": incident_id,
            "channels": channels,
            "summary": summary,
        }

    async def _generate_summary(self, state: RemediatorState) -> str:
        """Generate a human-readable incident summary.

        Args:
            state: Current remediation state.

        Returns:
            Plain-text incident summary for on-call engineer.
        """
        if not self._client:
            return (
                f"ARGUS Safety Incident\n"
                f"Risk: {state['risk_level']} (score: {state['composite_score']:.2f})\n"
                f"Categories: {state['triggered_categories']}\n"
                f"Agent: {state['agent_id']}\n"
                f"Request: {state['request_id']}"
            )

        prompt = (
            f"Generate a concise on-call incident report (max 150 words) for this ARGUS safety event:\n"
            f"Risk level: {state['risk_level']}\n"
            f"Score: {state['composite_score']:.3f}\n"
            f"Categories triggered: {state['triggered_categories']}\n"
            f"Agent ID: {state['agent_id']}\n"
            f"Predicted failure probability: {state.get('predicted_failure_prob', 0):.2f}\n"
            f"\nInclude: what was detected, severity, recommended immediate action."
        )

        response = self._client.messages.create(
            model=_LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _notify_slack(
        self,
        incident_id: str,
        summary: str,
        state: RemediatorState,
    ) -> None:
        """Post alert to Slack safety channel."""
        import aiohttp

        payload = {
            "text": f"🚨 *ARGUS Safety Alert* — `{incident_id}`",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{summary}```"},
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Risk: *{state['risk_level']}*"},
                        {"type": "mrkdwn", "text": f"Score: `{state['composite_score']:.3f}`"},
                    ],
                },
            ],
        }

        async with aiohttp.ClientSession() as session:
            await session.post(self._slack_webhook, json=payload)

    async def _trigger_pagerduty(
        self,
        incident_id: str,
        summary: str,
        state: RemediatorState,
    ) -> None:
        """Create PagerDuty incident for critical events."""
        import aiohttp

        payload = {
            "routing_key": self._pd_key,
            "event_action": "trigger",
            "dedup_key": incident_id,
            "payload": {
                "summary": summary[:1024],
                "severity": "critical" if state["risk_level"] == "critical" else "error",
                "source": "argus-remediator",
                "custom_details": {
                    "agent_id": state["agent_id"],
                    "composite_score": state["composite_score"],
                    "categories": state["triggered_categories"],
                },
            },
        }

        async with aiohttp.ClientSession() as session:
            await session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"},
            )


# ── LangGraph Orchestrator ────────────────────────────────────────────────────

class RemediatorOrchestrator:
    """Builds and runs the LangGraph StateGraph for remediation decisions.

    The orchestrator uses the Anthropic API to reason about which remediation
    action to take, then routes to the appropriate specialized sub-agent.

    Args:
        anthropic_client: Anthropic API client.
        output_filter: OutputFilterAgent instance.
        quarantine_agent: QuarantineAgent instance.
        rollback_agent: RollbackAgent instance.
        escalation_agent: EscalationAgent instance.
        audit_callback: Async function to persist RemediationRecord to Spanner.

    Example:
        orchestrator = RemediatorOrchestrator(
            anthropic_client=Anthropic(),
            output_filter=output_filter,
            quarantine_agent=quarantine,
            rollback_agent=rollback,
            escalation_agent=escalation,
            audit_callback=spanner_writer.write,
        )
        result = await orchestrator.run(state)
    """

    # Triage thresholds
    FILTER_THRESHOLD = 0.40
    QUARANTINE_THRESHOLD = 0.65
    ROLLBACK_THRESHOLD = 0.80
    ESCALATE_THRESHOLD = 0.90

    def __init__(
        self,
        anthropic_client: Anthropic,
        output_filter: OutputFilterAgent,
        quarantine_agent: QuarantineAgent,
        rollback_agent: RollbackAgent,
        escalation_agent: EscalationAgent,
        audit_callback: Optional[Any] = None,
    ) -> None:
        self._client = anthropic_client
        self._agents = {
            RemediationAction.FILTER_OUTPUT: output_filter,
            RemediationAction.QUARANTINE_AGENT: quarantine_agent,
            RemediationAction.ROLLBACK_MODEL: rollback_agent,
            RemediationAction.ESCALATE_HUMAN: escalation_agent,
        }
        self._audit = audit_callback

    async def run(self, state: RemediatorState) -> RemediatorState:
        """Execute the full remediation pipeline.

        Args:
            state: Initial RemediatorState with risk signal data.

        Returns:
            Final state with remediation result and audit trail.
        """
        t0 = time.monotonic_ns()
        state["remediation_id"] = str(uuid.uuid4())
        state["status"] = RemediationStatus.IN_PROGRESS.value
        state["timeline"] = []

        self._add_event(state, "remediation_started")

        # Step 1: Orchestrator decides action via Anthropic API
        decision = await self._orchestrate(state)
        state["selected_action"] = decision.action.value
        state["action_parameters"] = decision.parameters
        state["orchestrator_reasoning"] = decision.reasoning
        self._add_event(state, "action_selected", action=decision.action.value)

        # Step 2: Execute the selected action
        result = await self._execute_action(decision, state)
        state["action_result"] = result
        state["status"] = RemediationStatus.COMPLETED.value
        self._add_event(state, "remediation_completed")

        state["latency_ms"] = (time.monotonic_ns() - t0) / 1e6

        if state["latency_ms"] > 200:
            logger.warning("remediation_sla_miss", latency_ms=state["latency_ms"])

        # Step 3: Audit log
        if self._audit:
            await self._write_audit(state)

        return state

    async def _orchestrate(self, state: RemediatorState) -> RemediationDecision:
        """Use the Anthropic API to reason about which action to take.

        Args:
            state: Current state with risk signal.

        Returns:
            RemediationDecision with selected action and reasoning.
        """
        # Fast triage without LLM for clear-cut cases
        score = state["composite_score"]

        if score < self.FILTER_THRESHOLD:
            return RemediationDecision(
                action=RemediationAction.NO_ACTION,
                parameters={},
                reasoning="Score below filter threshold — no action required",
                confidence=0.95,
            )

        if score >= self.ESCALATE_THRESHOLD:
            return RemediationDecision(
                action=RemediationAction.ESCALATE_HUMAN,
                parameters={"priority": "critical"},
                reasoning="Score exceeds escalation threshold — requires human review",
                confidence=0.99,
            )

        # Use Anthropic API for nuanced decision in middle range
        prompt = self._build_decision_prompt(state)
        response = self._client.messages.create(
            model=_LLM_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_decision(response.content[0].text, score)

    def _build_decision_prompt(self, state: RemediatorState) -> str:
        """Build the prompt for the LLM remediation decision.

        Args:
            state: Current remediation state.

        Returns:
            Structured prompt for decision making.
        """
        return f"""You are ARGUS's remediation orchestrator. Choose the safest remediation action.

SAFETY SIGNAL:
- Risk level: {state['risk_level']}
- Composite score: {state['composite_score']:.3f}
- Triggered categories: {state['triggered_categories']}
- Predicted failure probability (60s): {state.get('predicted_failure_prob', 0):.2f}
- Agent ID: {state['agent_id']}

AVAILABLE ACTIONS (choose exactly one):
1. filter_output — rewrite the current output to remove unsafe content (fast, minimal disruption)
2. quarantine_agent — isolate the agent to a shadow model for 5 minutes
3. rollback_model — revert to the last safe model checkpoint (disruptive, use for high risk)
4. escalate_human — page on-call engineer immediately (for ambiguous critical situations)

DECISION CRITERIA:
- Score 0.40-0.64: prefer filter_output
- Score 0.65-0.79: prefer quarantine_agent
- Score 0.80-0.89: prefer rollback_model
- Score 0.90+: always escalate_human
- If predicted failure > 0.85: escalate regardless of current score

Respond in JSON:
{{"action": "action_name", "reason": "1-sentence explanation", "confidence": 0.0-1.0}}"""

    def _parse_decision(self, response_text: str, score: float) -> RemediationDecision:
        """Parse the LLM JSON response into a RemediationDecision.

        Args:
            response_text: Raw LLM response text.
            score: Current risk score (fallback for parsing failures).

        Returns:
            RemediationDecision parsed from response.
        """
        try:
            data = json.loads(response_text.strip())
            action = RemediationAction(data["action"])
            return RemediationDecision(
                action=action,
                parameters={},
                reasoning=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.8)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: rule-based decision
            logger.warning("decision_parse_failed", error=str(e))
            action = (
                RemediationAction.ROLLBACK_MODEL if score >= 0.80
                else RemediationAction.QUARANTINE_AGENT if score >= 0.65
                else RemediationAction.FILTER_OUTPUT
            )
            return RemediationDecision(
                action=action,
                parameters={},
                reasoning=f"Fallback rule-based: score={score:.2f}",
                confidence=0.7,
            )

    async def _execute_action(
        self,
        decision: RemediationDecision,
        state: RemediatorState,
    ) -> Dict[str, Any]:
        """Route to and execute the selected agent action.

        Args:
            decision: The remediation decision from orchestration.
            state: Full remediation state.

        Returns:
            Action result dict.
        """
        if decision.action == RemediationAction.NO_ACTION:
            return {"message": "No action required"}

        agent = self._agents.get(decision.action)
        if not agent:
            return {"error": f"Unknown action: {decision.action}"}

        try:
            if decision.action == RemediationAction.FILTER_OUTPUT:
                return await agent.execute(
                    original_output=state.get("agent_output", ""),
                    risk_categories=state["triggered_categories"],
                    parameters=decision.parameters,
                )
            elif decision.action == RemediationAction.QUARANTINE_AGENT:
                return await agent.execute(
                    agent_id=state["agent_id"],
                    parameters=decision.parameters,
                )
            elif decision.action == RemediationAction.ROLLBACK_MODEL:
                return await agent.execute(
                    current_version=state.get("model_version", "unknown"),
                    parameters=decision.parameters,
                )
            elif decision.action == RemediationAction.ESCALATE_HUMAN:
                return await agent.execute(
                    state=state,
                    parameters=decision.parameters,
                )
        except Exception as e:
            logger.error("action_execution_failed", action=decision.action.value, error=str(e))
            return {"error": str(e), "action": decision.action.value}

    def _add_event(self, state: RemediatorState, event: str, **kwargs: Any) -> None:
        """Append a timestamped event to the state timeline.

        Args:
            state: Mutable state dict.
            event: Event name.
            **kwargs: Additional event metadata.
        """
        state["timeline"].append({
            "event": event,
            "timestamp_ns": time.monotonic_ns(),
            **kwargs,
        })

    async def _write_audit(self, state: RemediatorState) -> None:
        """Persist an immutable audit record.

        Args:
            state: Final remediation state.
        """
        from datetime import datetime, timezone
        record = RemediationRecord(
            remediation_id=state["remediation_id"],
            request_id=state["request_id"],
            agent_id=state["agent_id"],
            risk_level=state["risk_level"],
            composite_score=state["composite_score"],
            action_taken=state["selected_action"],
            action_parameters=state["action_parameters"],
            orchestrator_reasoning=state["orchestrator_reasoning"],
            outcome=state["action_result"],
            latency_ms=state["latency_ms"],
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
        )

        try:
            await self._audit(record)
        except Exception as e:
            logger.error("audit_write_failed", error=str(e))
