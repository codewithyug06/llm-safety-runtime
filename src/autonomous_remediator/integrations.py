"""
MOD-06: AutonomousRemediator External Integrations
====================================================
All external client wrappers used by the remediation action agents:
  - RedisQuarantineStore     — quarantine flag management via Redis
  - MLflowRollbackClient     — model version rollback via MLflow
  - SlackNotifier            — alert dispatch via Slack webhooks
  - PagerDutyNotifier        — on-call escalation via PagerDuty Events API v2
  - SpannerAuditLogger       — immutable audit log writes to Cloud Spanner

Design principles:
- All integrations fail-safe: log errors but never crash the remediation loop
- Async-first where I/O is network-bound (Slack, PagerDuty)
- Quarantine TTL configurable (default 300s)
- Spanner rows are INSERT-only (no UPDATE/DELETE) for immutability
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from src.exceptions import AuditLogError, QuarantineError, RollbackError

logger = structlog.get_logger(__name__)


# ── Redis Quarantine Store ────────────────────────────────────────────────────

class RedisQuarantineStore:
    """Manages agent quarantine flags in Redis with TTL-based expiry.

    A quarantined agent has its output suppressed by the filter layer.
    Quarantine flags auto-expire after `default_ttl_s` seconds so
    agents automatically recover without manual intervention.

    Args:
        redis_url: Redis connection URL (redis://host:port/db).
        default_ttl_s: Default quarantine TTL in seconds.
        key_prefix: Redis key prefix (avoids collisions).

    Example:
        store = RedisQuarantineStore("redis://localhost:6379/0")
        store.quarantine("agent_123", ttl_s=300)
        assert store.is_quarantined("agent_123")
        store.release("agent_123")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl_s: int = 300,
        key_prefix: str = "argus:quarantine:",
    ) -> None:
        self._redis_url = redis_url
        self._default_ttl_s = default_ttl_s
        self._key_prefix = key_prefix
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazily create Redis client.

        Returns:
            Redis client instance.

        Raises:
            ImportError: If redis-py is not installed.
        """
        if self._client is None:
            try:
                import redis
            except ImportError:
                raise ImportError("Run: pip install redis>=5.0.0")
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _key(self, agent_id: str) -> str:
        return f"{self._key_prefix}{agent_id}"

    def quarantine(self, agent_id: str, ttl_s: Optional[int] = None, reason: str = "") -> None:
        """Set quarantine flag for an agent.

        Args:
            agent_id: Agent to quarantine.
            ttl_s: TTL in seconds (uses default if None).
            reason: Human-readable reason for logging.

        Raises:
            QuarantineError: If Redis write fails.
        """
        ttl = ttl_s or self._default_ttl_s
        payload = json.dumps({
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "ttl_s": ttl,
        })
        try:
            client = self._get_client()
            client.setex(self._key(agent_id), ttl, payload)
            logger.info(
                "agent_quarantined",
                agent_id=agent_id,
                ttl_s=ttl,
                reason=reason,
            )
        except Exception as exc:
            raise QuarantineError(f"Failed to quarantine {agent_id}: {exc}") from exc

    def release(self, agent_id: str) -> None:
        """Remove quarantine flag immediately.

        Args:
            agent_id: Agent to release.
        """
        try:
            client = self._get_client()
            client.delete(self._key(agent_id))
            logger.info("agent_released", agent_id=agent_id)
        except Exception as exc:
            logger.error("quarantine_release_failed", agent_id=agent_id, error=str(exc))

    def is_quarantined(self, agent_id: str) -> bool:
        """Return True if agent is currently quarantined.

        Args:
            agent_id: Agent to check.

        Returns:
            True if quarantine flag is set and not expired.
        """
        try:
            client = self._get_client()
            return client.exists(self._key(agent_id)) > 0
        except Exception as exc:
            logger.error("quarantine_check_failed", agent_id=agent_id, error=str(exc))
            return False

    def get_quarantine_info(self, agent_id: str) -> Optional[Dict]:
        """Get quarantine metadata for an agent.

        Args:
            agent_id: Agent to query.

        Returns:
            Dict with quarantined_at, reason, ttl_s — or None if not quarantined.
        """
        try:
            client = self._get_client()
            val = client.get(self._key(agent_id))
            if val:
                return json.loads(val)
        except Exception as exc:
            logger.error("quarantine_get_failed", agent_id=agent_id, error=str(exc))
        return None


# ── MLflow Rollback Client ────────────────────────────────────────────────────

class MLflowRollbackClient:
    """Rolls back a model in MLflow registry to a specific version.

    Transitions the target version to Production and demotes the current
    Production version to Archived.

    Args:
        tracking_uri: MLflow tracking server URI.

    Example:
        client = MLflowRollbackClient("http://mlflow:5000")
        client.rollback("argus-safety-critic", version=3)
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000") -> None:
        self._tracking_uri = tracking_uri
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazily create MLflow MlflowClient."""
        if self._client is None:
            try:
                import mlflow
                mlflow.set_tracking_uri(self._tracking_uri)
                self._client = mlflow.MlflowClient()
            except ImportError:
                raise ImportError("Run: pip install mlflow>=2.10.0")
        return self._client

    def get_current_production_version(self, model_name: str) -> Optional[str]:
        """Get the currently Production model version.

        Args:
            model_name: Registered model name.

        Returns:
            Version string (e.g. "4") or None if no Production version.
        """
        try:
            client = self._get_client()
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return versions[0].version
        except Exception as exc:
            logger.error("get_production_version_failed", model=model_name, error=str(exc))
        return None

    def rollback(self, model_name: str, version: int) -> None:
        """Roll back model to specified version.

        Transitions target version to Production and current Production to Archived.

        Args:
            model_name: Registered model name in MLflow.
            version: Version number to roll back to.

        Raises:
            RollbackError: If the transition fails.
        """
        try:
            client = self._get_client()

            # Archive current production version
            current = self.get_current_production_version(model_name)
            if current and current != str(version):
                client.transition_model_version_stage(
                    name=model_name,
                    version=current,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                logger.info(
                    "previous_version_archived",
                    model=model_name,
                    version=current,
                )

            # Promote target version to Production
            client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage="Production",
                archive_existing_versions=True,
            )

            logger.info(
                "model_rolled_back",
                model=model_name,
                target_version=version,
                previous_version=current,
            )

        except Exception as exc:
            raise RollbackError(
                message=f"Rollback failed for {model_name} v{version}: {exc}",
                model_name=model_name,
                target_version=version,
            ) from exc

    def list_available_versions(self, model_name: str) -> List[Dict]:
        """List all registered versions for a model.

        Args:
            model_name: Registered model name.

        Returns:
            List of dicts with version, stage, run_id, creation_time.
        """
        try:
            client = self._get_client()
            versions = client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_time": v.creation_timestamp,
                }
                for v in versions
            ]
        except Exception as exc:
            logger.error("list_versions_failed", model=model_name, error=str(exc))
            return []


# ── Slack Notifier ────────────────────────────────────────────────────────────

class SlackNotifier:
    """Sends alert messages to Slack via Incoming Webhook.

    Args:
        webhook_url: Slack Incoming Webhook URL.
        channel: Channel override (uses webhook default if empty).
        username: Bot display name.
        icon_emoji: Bot emoji icon.

    Example:
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/...")
        await notifier.send_alert("Agent X quarantined: safety score 0.93")
    """

    def __init__(
        self,
        webhook_url: str = "",
        channel: str = "#argus-alerts",
        username: str = "ARGUS Remediator",
        icon_emoji: str = ":shield:",
    ) -> None:
        self._webhook_url = webhook_url
        self._channel = channel
        self._username = username
        self._icon_emoji = icon_emoji

    async def send_alert(
        self,
        message: str,
        severity: str = "warning",
        agent_id: str = "",
        extra_fields: Optional[Dict] = None,
    ) -> bool:
        """Send an alert to Slack.

        Args:
            message: Human-readable alert message.
            severity: One of 'info', 'warning', 'critical'.
            agent_id: Agent associated with this alert.
            extra_fields: Additional key-value pairs to include.

        Returns:
            True if message was sent successfully.
        """
        if not self._webhook_url:
            logger.warning("slack_webhook_not_configured", message=message[:50])
            return False

        color_map = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}
        color = color_map.get(severity, "#888888")

        fields = [{"title": "Severity", "value": severity.upper(), "short": True}]
        if agent_id:
            fields.append({"title": "Agent ID", "value": agent_id, "short": True})
        if extra_fields:
            fields.extend([
                {"title": k, "value": str(v), "short": True}
                for k, v in extra_fields.items()
            ])

        payload = {
            "username": self._username,
            "icon_emoji": self._icon_emoji,
            "attachments": [{
                "color": color,
                "text": message,
                "fields": fields,
                "ts": int(time.time()),
            }],
        }
        if self._channel:
            payload["channel"] = self._channel

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status == 200:
                        logger.info("slack_alert_sent", severity=severity, agent_id=agent_id)
                        return True
                    else:
                        logger.warning("slack_alert_failed", status=resp.status)
                        return False
        except Exception as exc:
            logger.error("slack_send_error", error=str(exc))
            return False


# ── PagerDuty Notifier ────────────────────────────────────────────────────────

class PagerDutyNotifier:
    """Triggers PagerDuty incidents via Events API v2.

    Args:
        integration_key: PagerDuty Events API v2 routing key.
        service_name: Service name shown in PagerDuty.

    Example:
        notifier = PagerDutyNotifier(integration_key="abc123...")
        await notifier.trigger_incident("agent_42", "Safety score 0.96", severity="critical")
    """

    PAGERDUTY_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        integration_key: str = "",
        service_name: str = "ARGUS Safety System",
    ) -> None:
        self._integration_key = integration_key
        self._service_name = service_name

    async def trigger_incident(
        self,
        agent_id: str,
        summary: str,
        severity: str = "critical",
        details: Optional[Dict] = None,
    ) -> bool:
        """Trigger a PagerDuty incident.

        Args:
            agent_id: Dedup key (one open incident per agent at a time).
            summary: Short summary string (shown in alert).
            severity: One of 'critical', 'error', 'warning', 'info'.
            details: Additional context dict.

        Returns:
            True if incident was created successfully.
        """
        if not self._integration_key:
            logger.warning("pagerduty_key_not_configured", summary=summary[:50])
            return False

        payload = {
            "routing_key": self._integration_key,
            "event_action": "trigger",
            "dedup_key": f"argus-{agent_id}",
            "payload": {
                "summary": f"[{self._service_name}] {summary}",
                "severity": severity,
                "source": agent_id,
                "component": "argus-remediator",
                "custom_details": details or {},
            },
        }

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.PAGERDUTY_EVENTS_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status in (200, 202):
                        logger.info(
                            "pagerduty_incident_triggered",
                            agent_id=agent_id,
                            severity=severity,
                        )
                        return True
                    else:
                        body = await resp.text()
                        logger.warning("pagerduty_trigger_failed", status=resp.status, body=body[:100])
                        return False
        except Exception as exc:
            logger.error("pagerduty_send_error", error=str(exc))
            return False

    async def resolve_incident(self, agent_id: str) -> bool:
        """Resolve an open PagerDuty incident.

        Args:
            agent_id: Dedup key to resolve.

        Returns:
            True if resolved successfully.
        """
        if not self._integration_key:
            return False

        payload = {
            "routing_key": self._integration_key,
            "event_action": "resolve",
            "dedup_key": f"argus-{agent_id}",
        }
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.PAGERDUTY_EVENTS_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    return resp.status in (200, 202)
        except Exception as exc:
            logger.error("pagerduty_resolve_error", error=str(exc))
            return False


# ── Cloud Spanner Audit Logger ────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """A single immutable remediation audit record.

    Args:
        record_id: Unique record UUID.
        timestamp: ISO-8601 event timestamp.
        agent_id: Agent that triggered remediation.
        safety_score: Safety score at time of action.
        action_taken: Remediation action string.
        action_detail: Additional action context.
        triggered_by: Source system or human.
        outcome: 'success' or 'failure'.
        latency_ms: End-to-end remediation latency.
    """

    record_id: str
    timestamp: str
    agent_id: str
    safety_score: float
    action_taken: str
    action_detail: str
    triggered_by: str
    outcome: str
    latency_ms: float


class SpannerAuditLogger:
    """Writes immutable audit records to Cloud Spanner.

    Records are INSERT-only — never updated or deleted — providing
    a tamper-evident audit trail for compliance and forensics.

    Args:
        instance_id: Cloud Spanner instance ID.
        database_id: Cloud Spanner database ID.
        table_name: Spanner table for audit records.

    Example:
        logger_client = SpannerAuditLogger("argus-instance", "argus-db")
        logger_client.write(AuditRecord(...))
    """

    def __init__(
        self,
        instance_id: str = "argus-instance",
        database_id: str = "argus-db",
        table_name: str = "RemediationAuditLog",
    ) -> None:
        self._instance_id = instance_id
        self._database_id = database_id
        self._table_name = table_name
        self._database: Optional[Any] = None

    def _get_database(self) -> Any:
        """Lazily create Cloud Spanner database client.

        Returns:
            Cloud Spanner Database object.

        Raises:
            ImportError: If google-cloud-spanner is not installed.
        """
        if self._database is None:
            try:
                from google.cloud import spanner
            except ImportError:
                raise ImportError("Run: pip install google-cloud-spanner>=3.40.0")

            client = spanner.Client()
            instance = client.instance(self._instance_id)
            self._database = instance.database(self._database_id)
        return self._database

    def write(self, record: AuditRecord) -> None:
        """Write an audit record to Spanner (INSERT only).

        Args:
            record: Audit record to persist.

        Raises:
            AuditLogError: If the write fails.
        """
        try:
            db = self._get_database()
            with db.batch() as batch:
                batch.insert(
                    table=self._table_name,
                    columns=[
                        "RecordId", "Timestamp", "AgentId", "SafetyScore",
                        "ActionTaken", "ActionDetail", "TriggeredBy",
                        "Outcome", "LatencyMs",
                    ],
                    values=[[
                        record.record_id,
                        record.timestamp,
                        record.agent_id,
                        record.safety_score,
                        record.action_taken,
                        record.action_detail,
                        record.triggered_by,
                        record.outcome,
                        record.latency_ms,
                    ]],
                )
            logger.info(
                "audit_record_written",
                record_id=record.record_id,
                agent_id=record.agent_id,
                action=record.action_taken,
            )
        except Exception as exc:
            raise AuditLogError(
                f"Spanner write failed for record {record.record_id}: {exc}"
            ) from exc

    def query_by_agent(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Query audit records for a specific agent.

        Args:
            agent_id: Agent ID to query.
            limit: Maximum records to return.

        Returns:
            List of audit record dicts, newest first.
        """
        try:
            db = self._get_database()
            with db.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    f"SELECT * FROM {self._table_name} "
                    f"WHERE AgentId = @agent_id "
                    f"ORDER BY Timestamp DESC LIMIT @limit",
                    params={"agent_id": agent_id, "limit": limit},
                    param_types={
                        "agent_id": __import__(
                            "google.cloud.spanner_v1", fromlist=["param_types"]
                        ).param_types.STRING,
                        "limit": __import__(
                            "google.cloud.spanner_v1", fromlist=["param_types"]
                        ).param_types.INT64,
                    },
                )
                return [
                    {
                        "record_id": row[0],
                        "timestamp": row[1],
                        "agent_id": row[2],
                        "safety_score": row[3],
                        "action_taken": row[4],
                        "action_detail": row[5],
                        "triggered_by": row[6],
                        "outcome": row[7],
                        "latency_ms": row[8],
                    }
                    for row in results
                ]
        except Exception as exc:
            logger.error("audit_query_failed", agent_id=agent_id, error=str(exc))
            return []
