"""
MOD-05: Telemetry Kafka Consumer → PredictiveOracle Pipeline
==============================================================
Consumes telemetry events from the `argus.telemetry` Kafka topic,
maintains a rolling TelemetryWindow per monitored agent, and calls
PredictiveOracle.predict() every inference_interval_s seconds.

Publishes OraclePrediction results to `argus.risk.predictions` topic
for the AutonomousRemediator to consume.

Design decisions:
- One TelemetryWindow deque per agent_id (max seq_len=60 steps)
- Inference runs async to not block the Kafka consumer poll loop
- Dead agents cleaned up after TTL_SECONDS of inactivity
- JSON serialization for cross-service compatibility

Run standalone:
    python -m src.predictive_oracle.telemetry_consumer
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np
import structlog

from src.exceptions import InsufficientTelemetryError, OraclePredictionError

logger = structlog.get_logger(__name__)

# How long (seconds) to keep a silent agent's window before evicting
AGENT_TTL_SECONDS: float = 300.0


@dataclass
class TelemetryEvent:
    """A single telemetry snapshot from a monitored agent.

    Args:
        agent_id: Identifier of the producing agent.
        timestamp: ISO-8601 event timestamp.
        request_rate: Requests per second.
        error_rate: Fraction of requests with errors [0,1].
        p95_latency_ms: 95th percentile response latency (ms).
        token_throughput: Tokens generated per second.
        safety_score_avg: Rolling average safety score [0,1].
        memory_utilization: GPU/CPU memory fraction [0,1].
        queue_depth: Pending request queue depth.
        cpu_utilization: CPU utilization fraction [0,1].
        context_window_fill: Fraction of context window used [0,1].
    """

    agent_id: str
    timestamp: str
    request_rate: float = 0.0
    error_rate: float = 0.0
    p95_latency_ms: float = 0.0
    token_throughput: float = 0.0
    safety_score_avg: float = 0.0
    memory_utilization: float = 0.0
    queue_depth: float = 0.0
    cpu_utilization: float = 0.0
    context_window_fill: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to 9-feature numpy vector matching oracle input format.

        Returns:
            Float32 array of shape (9,).
        """
        return np.array([
            self.request_rate,
            self.error_rate,
            self.p95_latency_ms,
            self.token_throughput,
            self.safety_score_avg,
            self.memory_utilization,
            self.queue_depth,
            self.cpu_utilization,
            self.context_window_fill,
        ], dtype=np.float32)

    @classmethod
    def from_kafka_message(cls, msg_value: bytes) -> "TelemetryEvent":
        """Deserialize from Kafka message bytes (JSON).

        Args:
            msg_value: Raw bytes from Kafka.

        Returns:
            TelemetryEvent instance.

        Raises:
            ValueError: If JSON is malformed or required fields missing.
        """
        data = json.loads(msg_value.decode("utf-8"))
        return cls(
            agent_id=data["agent_id"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            request_rate=float(data.get("request_rate", 0.0)),
            error_rate=float(data.get("error_rate", 0.0)),
            p95_latency_ms=float(data.get("p95_latency_ms", 0.0)),
            token_throughput=float(data.get("token_throughput", 0.0)),
            safety_score_avg=float(data.get("safety_score_avg", 0.0)),
            memory_utilization=float(data.get("memory_utilization", 0.0)),
            queue_depth=float(data.get("queue_depth", 0.0)),
            cpu_utilization=float(data.get("cpu_utilization", 0.0)),
            context_window_fill=float(data.get("context_window_fill", 0.0)),
        )


@dataclass
class OraclePrediction:
    """Failure risk prediction for a single agent.

    Args:
        agent_id: Agent being monitored.
        timestamp: Prediction timestamp.
        risk_score_30s: Predicted failure probability at 30s horizon.
        risk_score_60s: Predicted failure probability at 60s horizon.
        risk_score_90s: Predicted failure probability at 90s horizon.
        confidence_interval_low: Lower bound of conformal interval.
        confidence_interval_high: Upper bound of conformal interval.
        is_high_risk: True if any horizon score exceeds 0.70.
    """

    agent_id: str
    timestamp: str
    risk_score_30s: float
    risk_score_60s: float
    risk_score_90s: float
    confidence_interval_low: float
    confidence_interval_high: float
    is_high_risk: bool

    def to_kafka_payload(self) -> bytes:
        """Serialize to JSON bytes for Kafka publishing.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        return json.dumps(asdict(self)).encode("utf-8")


class TelemetryWindowBuffer:
    """Maintains a rolling deque of TelemetryEvents per agent.

    Args:
        seq_len: Maximum number of timesteps to retain per agent.
        agent_ttl_s: Seconds of inactivity before agent is evicted.
    """

    def __init__(self, seq_len: int = 60, agent_ttl_s: float = AGENT_TTL_SECONDS) -> None:
        self._seq_len = seq_len
        self._agent_ttl_s = agent_ttl_s
        self._windows: Dict[str, Deque[TelemetryEvent]] = defaultdict(
            lambda: deque(maxlen=seq_len)
        )
        self._last_seen: Dict[str, float] = {}

    def add(self, event: TelemetryEvent) -> None:
        """Add a TelemetryEvent to the agent's window.

        Args:
            event: Telemetry snapshot from agent.
        """
        self._windows[event.agent_id].append(event)
        self._last_seen[event.agent_id] = time.monotonic()

    def get_window_array(self, agent_id: str) -> Optional[np.ndarray]:
        """Get the current window as a numpy array for oracle inference.

        Args:
            agent_id: Agent identifier.

        Returns:
            Float32 array of shape (current_len, 9), or None if no data.
        """
        window = self._windows.get(agent_id)
        if not window:
            return None
        return np.stack([e.to_feature_vector() for e in window])

    def window_length(self, agent_id: str) -> int:
        """Return current window length for an agent."""
        return len(self._windows.get(agent_id, []))

    def evict_stale_agents(self) -> int:
        """Remove agents not seen within TTL.

        Returns:
            Number of agents evicted.
        """
        now = time.monotonic()
        stale = [
            aid for aid, ts in self._last_seen.items()
            if now - ts > self._agent_ttl_s
        ]
        for aid in stale:
            del self._windows[aid]
            del self._last_seen[aid]
        if stale:
            logger.info("stale_agents_evicted", count=len(stale), agents=stale)
        return len(stale)

    @property
    def active_agents(self) -> list:
        """List of agent IDs with active windows."""
        return list(self._windows.keys())


class TelemetryConsumer:
    """Consumes telemetry from Kafka and drives oracle predictions.

    Args:
        telemetry_topic: Kafka input topic for telemetry events.
        predictions_topic: Kafka output topic for oracle predictions.
        bootstrap_servers: Kafka broker addresses.
        consumer_group: Kafka consumer group ID.
        oracle_model_path: Path to trained PatchTST model checkpoint.
        calibrator_path: Path to pickled ConformalCalibrator.
        seq_len: Telemetry window length for oracle input.
        min_window_len: Minimum window before running inference.
        inference_interval_s: Minimum seconds between predictions per agent.
        device: Torch device.

    Example:
        consumer = TelemetryConsumer(
            oracle_model_path="models/oracle/patchtst.pt",
            calibrator_path="models/oracle/calibrator.pkl",
        )
        asyncio.run(consumer.run())
    """

    def __init__(
        self,
        telemetry_topic: str = "argus.telemetry",
        predictions_topic: str = "argus.risk.predictions",
        bootstrap_servers: str = "localhost:9092",
        consumer_group: str = "argus-oracle-consumer",
        oracle_model_path: str = "models/oracle/patchtst.pt",
        calibrator_path: str = "models/oracle/calibrator.pkl",
        seq_len: int = 60,
        min_window_len: int = 12,
        inference_interval_s: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self._telemetry_topic = telemetry_topic
        self._predictions_topic = predictions_topic
        self._bootstrap_servers = bootstrap_servers
        self._consumer_group = consumer_group
        self._oracle_model_path = Path(oracle_model_path)
        self._calibrator_path = Path(calibrator_path)
        self._seq_len = seq_len
        self._min_window_len = min_window_len
        self._inference_interval_s = inference_interval_s
        self._device = device

        self._buffer = TelemetryWindowBuffer(seq_len=seq_len)
        self._last_inference: Dict[str, float] = {}
        self._oracle: Optional[Any] = None
        self._calibrator: Optional[Any] = None
        self._producer: Optional[Any] = None
        self._running = False

    def _load_oracle(self) -> None:
        """Load oracle model and calibrator from disk."""
        import pickle

        import torch

        from src.predictive_oracle.oracle import ConformalCalibrator, PredictiveOracleModel

        logger.info("loading_oracle_model", path=str(self._oracle_model_path))

        # Load config for architecture params
        try:
            from src.config import load_oracle_config
            cfg = load_oracle_config()
            hidden_dim = cfg.model.hidden_dim
            num_heads = cfg.model.num_heads
            num_layers = cfg.model.num_layers
            patch_len = cfg.model.patch_len
            stride = cfg.model.stride
            num_features = cfg.model.num_features
            n_horizons = len(cfg.model.forecast_horizons)
        except Exception:
            hidden_dim, num_heads, num_layers = 128, 8, 3
            patch_len, stride, num_features, n_horizons = 12, 6, 9, 3

        self._oracle = PredictiveOracleModel(
            seq_len=self._seq_len,
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_len=patch_len,
            stride=stride,
            num_forecast_horizons=n_horizons,
        ).to(self._device)

        if self._oracle_model_path.exists():
            self._oracle.load_state_dict(
                torch.load(str(self._oracle_model_path), map_location=self._device)
            )
        else:
            logger.warning("oracle_model_not_found_using_random", path=str(self._oracle_model_path))

        self._oracle.eval()

        if self._calibrator_path.exists():
            with self._calibrator_path.open("rb") as f:
                self._calibrator = pickle.load(f)
            logger.info("calibrator_loaded", path=str(self._calibrator_path))
        else:
            logger.warning("calibrator_not_found", path=str(self._calibrator_path))

    def _make_prediction(self, agent_id: str) -> Optional[OraclePrediction]:
        """Run oracle inference for a single agent.

        Args:
            agent_id: Agent to predict for.

        Returns:
            OraclePrediction or None if insufficient window data.
        """
        import torch

        window_arr = self._buffer.get_window_array(agent_id)
        if window_arr is None or len(window_arr) < self._min_window_len:
            return None

        # Pad or truncate to seq_len
        if len(window_arr) < self._seq_len:
            pad_len = self._seq_len - len(window_arr)
            window_arr = np.vstack([np.zeros((pad_len, window_arr.shape[1])), window_arr])
        else:
            window_arr = window_arr[-self._seq_len:]

        with torch.no_grad():
            x = torch.from_numpy(window_arr).unsqueeze(0).to(self._device)  # (1, seq, feats)
            logits = self._oracle(x)  # (1, n_horizons)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        risk_30s = float(probs[0]) if len(probs) > 0 else 0.5
        risk_60s = float(probs[1]) if len(probs) > 1 else 0.5
        risk_90s = float(probs[2]) if len(probs) > 2 else 0.5

        # Conformal interval around 60s prediction
        ci_low, ci_high = risk_60s - 0.10, risk_60s + 0.10
        if self._calibrator is not None:
            try:
                ci_low, ci_high = self._calibrator.predict_interval(
                    np.array([risk_60s])
                )
            except Exception:
                pass
        ci_low = float(np.clip(ci_low, 0, 1))
        ci_high = float(np.clip(ci_high, 0, 1))

        is_high_risk = risk_60s >= 0.70

        return OraclePrediction(
            agent_id=agent_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            risk_score_30s=risk_30s,
            risk_score_60s=risk_60s,
            risk_score_90s=risk_90s,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            is_high_risk=is_high_risk,
        )

    def _publish_prediction(self, prediction: OraclePrediction) -> None:
        """Publish an OraclePrediction to the Kafka predictions topic.

        Args:
            prediction: The prediction to publish.
        """
        if self._producer is None:
            return
        try:
            self._producer.produce(
                topic=self._predictions_topic,
                key=prediction.agent_id.encode(),
                value=prediction.to_kafka_payload(),
            )
            self._producer.poll(0)
        except Exception as exc:
            logger.error("prediction_publish_failed", agent_id=prediction.agent_id, error=str(exc))

    async def run(self) -> None:
        """Start the consumer loop — poll Kafka, buffer events, run oracle.

        This coroutine runs until stop() is called.

        Raises:
            ImportError: If confluent_kafka is not installed.
        """
        try:
            from confluent_kafka import Consumer, KafkaError, Producer
        except ImportError:
            raise ImportError("Run: pip install confluent-kafka")

        # Load oracle model
        self._load_oracle()

        # Create Kafka producer (for publishing predictions)
        self._producer = Producer({"bootstrap.servers": self._bootstrap_servers})

        # Create Kafka consumer
        consumer = Consumer({
            "bootstrap.servers": self._bootstrap_servers,
            "group.id": self._consumer_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        })
        consumer.subscribe([self._telemetry_topic])
        self._running = True

        logger.info(
            "telemetry_consumer_started",
            topic=self._telemetry_topic,
            predictions_topic=self._predictions_topic,
        )

        last_evict_time = time.monotonic()

        try:
            while self._running:
                msg = consumer.poll(timeout=0.1)

                if msg is None:
                    await asyncio.sleep(0)
                    continue

                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        logger.error("kafka_consumer_error", error=str(msg.error()))
                    continue

                # Parse telemetry event
                try:
                    event = TelemetryEvent.from_kafka_message(msg.value())
                    self._buffer.add(event)
                except Exception as exc:
                    logger.warning("telemetry_parse_error", error=str(exc))
                    continue

                # Run oracle if enough time has elapsed since last prediction
                agent_id = event.agent_id
                now = time.monotonic()
                last_pred_time = self._last_inference.get(agent_id, 0.0)

                if now - last_pred_time >= self._inference_interval_s:
                    try:
                        prediction = self._make_prediction(agent_id)
                        if prediction is not None:
                            self._publish_prediction(prediction)
                            self._last_inference[agent_id] = now

                            if prediction.is_high_risk:
                                logger.warning(
                                    "high_risk_prediction",
                                    agent_id=agent_id,
                                    risk_60s=f"{prediction.risk_score_60s:.3f}",
                                )
                            else:
                                logger.debug(
                                    "prediction_published",
                                    agent_id=agent_id,
                                    risk_60s=f"{prediction.risk_score_60s:.3f}",
                                )
                    except Exception as exc:
                        logger.error("oracle_inference_error", agent_id=agent_id, error=str(exc))

                # Periodic stale agent cleanup
                if now - last_evict_time > 60.0:
                    self._buffer.evict_stale_agents()
                    last_evict_time = now

        finally:
            consumer.close()
            if self._producer:
                self._producer.flush(timeout=5)
            logger.info("telemetry_consumer_stopped")

    def stop(self) -> None:
        """Signal the consumer loop to stop gracefully."""
        self._running = False
        logger.info("telemetry_consumer_stop_requested")
