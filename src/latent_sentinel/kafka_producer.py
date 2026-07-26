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
        if signal.risk_level == RiskLevel.SAFE:
            return 

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
            self._producer.poll(0)
        except BufferError as exc:
            raise KafkaConnectionError(
                f"Kafka producer queue full — increase queue.buffering.max.messages: {exc}"
            ) from exc

    def publish_batch(self, signals: list[SafetySignal], agent_id: str = "") -> None:

        for signal in signals:
            self.publish(signal, agent_id=agent_id)

    def flush(self, timeout: float = 5.0) -> int:

        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning("kafka_flush_incomplete", remaining=remaining)
        return remaining

    def close(self) -> None:
        self.flush()
        logger.info("kafka_producer_closed", topic=self._topic)
