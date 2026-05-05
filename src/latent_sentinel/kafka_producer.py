"""
MOD-01: SafetySignal Kafka Producer
=====================================
Serializes SafetySignal outputs from LatentSentinel and publishes them
to the argus.safety.signals Kafka topic for downstream consumers
(PredictiveOracle, AutonomousRemediator).

Design decisions:
- acks="1" (leader only): safety signals are time-critical; we accept
  the small durability trade-off for sub-millisecond publish latency.
- linger_ms=0: no batching delay — every signal is published immediately.
- async produce with on_delivery callback for error logging.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import structlog

from src.exceptions import KafkaConnectionError
from src.latent_sentinel.sentinel import ProbeCategory, RiskLevel, SafetySignal

logger = structlog.get_logger(__name__)


def _signal_to_dict(signal: SafetySignal) -> Dict[str, Any]:
    """Convert a SafetySignal to a JSON-serializable dictionary.

    Args:
        signal: The SafetySignal to serialize.

    Returns:
        Dictionary safe for json.dumps().
    """
    probe_scores = {
        cat.name: result.risk_score
        for cat, result in signal.probe_results.items()
    }
    return {
        "request_id": signal.request_id,
        "composite_risk_score": signal.composite_score,
        "risk_level": signal.risk_level.value,
        "probe_scores": probe_scores,
        "latency_ms": signal.total_latency_ms,
        "triggered_early": signal.triggered_early,
        "alert_tokens_ahead": signal.alert_tokens_ahead,
    }


class SafetySignalProducer:
    """Publishes SafetySignal events to Kafka.

    Wraps the confluent-kafka Producer with ARGUS-specific serialization
    and error handling.

    Args:
        bootstrap_servers: Comma-separated Kafka broker addresses.
        topic: Kafka topic to publish to.
        producer_config: Additional confluent-kafka producer settings.
        on_error: Optional callback invoked on delivery failure.

    Example:
        producer = SafetySignalProducer(
            bootstrap_servers="kafka:9092",
            topic="argus.safety.signals",
        )
        producer.publish(signal, agent_id="agent-42")
        producer.flush()
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str = "argus.safety.signals",
        producer_config: Optional[Dict[str, Any]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self._topic = topic
        self._on_error = on_error
        self._producer = self._build_producer(bootstrap_servers, producer_config or {})
        logger.info(
            "kafka_producer_initialized",
            topic=topic,
            bootstrap_servers=bootstrap_servers,
        )

    def _build_producer(
        self,
        bootstrap_servers: str,
        extra_config: Dict[str, Any],
    ) -> Any:
        """Create and return a confluent-kafka Producer instance.

        Args:
            bootstrap_servers: Kafka broker address string.
            extra_config: Additional producer configuration keys.

        Returns:
            Configured Producer instance.

        Raises:
            KafkaConnectionError: If the producer cannot be initialized.
        """
        try:
            from confluent_kafka import Producer
        except ImportError:
            raise ImportError("Run: pip install confluent-kafka")

        config = {
            "bootstrap.servers": bootstrap_servers,
            "acks": "1",
            "linger.ms": 0,
            "compression.type": "lz4",
            **extra_config,
        }
        try:
            return Producer(config)
        except Exception as exc:
            raise KafkaConnectionError(
                f"Failed to initialize Kafka producer: {exc}"
            ) from exc

    def _on_delivery(self, err: Any, msg: Any) -> None:
        """Confluent-kafka delivery callback.

        Args:
            err: Delivery error (None on success).
            msg: The delivered message object.
        """
        if err is not None:
            logger.error(
                "kafka_delivery_failed",
                topic=msg.topic(),
                partition=msg.partition(),
                error=str(err),
            )
            if self._on_error:
                self._on_error(RuntimeError(str(err)))
        else:
            logger.debug(
                "kafka_delivery_success",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

    def publish(self, signal: SafetySignal, agent_id: str = "") -> None:
        """Serialize and publish a SafetySignal to Kafka.

        Only publishes signals with risk_level above SAFE to reduce noise.

        Args:
            signal: The SafetySignal from LatentSentinel.
            agent_id: Optional identifier for the monitored agent.

        Raises:
            KafkaConnectionError: If the producer queue is full (buffer overflow).
        """
        if signal.risk_level == RiskLevel.SAFE:
            return  # Don't flood Kafka with safe signals

        payload = _signal_to_dict(signal)
        if agent_id:
            payload["agent_id"] = agent_id

        try:
            self._producer.produce(
                topic=self._topic,
                key=agent_id.encode() if agent_id else None,
                value=json.dumps(payload).encode("utf-8"),
                on_delivery=self._on_delivery,
            )
            # Non-blocking poll to handle delivery callbacks
            self._producer.poll(0)
        except BufferError as exc:
            raise KafkaConnectionError(
                f"Kafka producer queue full — increase queue.buffering.max.messages: {exc}"
            ) from exc

    def publish_batch(self, signals: list[SafetySignal], agent_id: str = "") -> None:
        """Publish multiple SafetySignals in one batch.

        Args:
            signals: List of SafetySignals to publish.
            agent_id: Shared agent identifier for all signals.
        """
        for signal in signals:
            self.publish(signal, agent_id=agent_id)

    def flush(self, timeout: float = 5.0) -> int:
        """Wait for all outstanding produce requests to complete.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            Number of messages still in the local queue (0 on full success).
        """
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning("kafka_flush_incomplete", remaining=remaining)
        return remaining

    def close(self) -> None:
        """Flush and close the producer."""
        self.flush()
        logger.info("kafka_producer_closed", topic=self._topic)
