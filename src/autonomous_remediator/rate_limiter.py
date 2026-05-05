"""
MOD-06: Rate Limit Agent
=========================
Implements adaptive rate limiting for unsafe agents using Redis sorted sets
as a sliding-window counter. Applies exponential backoff on repeated violations.

Design:
- Sliding window (1 second) via Redis ZRANGEBYSCORE + ZADD
- Each violation record stored as (agent_id, violation_count) → backoff multiplier
- Max requests per window configurable per agent risk tier
- After 3+ violations within 60s, agent is soft-quarantined via RateLimit flag

Redis key structure:
    argus:ratelimit:window:{agent_id}  → sorted set of timestamps
    argus:ratelimit:violations:{agent_id} → integer violation counter (with TTL)
"""

from __future__ import annotations

import time
import uuid
from typing import Optional, Tuple

import structlog

from src.exceptions import RemediatorError

logger = structlog.get_logger(__name__)

# Window size in seconds for sliding rate limit
WINDOW_SECONDS: float = 1.0
# Default max requests per window for a normal agent
DEFAULT_MAX_REQUESTS: int = 10
# Default max requests per window for a high-risk agent
HIGH_RISK_MAX_REQUESTS: int = 3
# Max violations before triggering soft quarantine
VIOLATION_THRESHOLD: int = 3
# Violation counter TTL (60 seconds rolling window)
VIOLATION_TTL_SECONDS: int = 60
# Base backoff seconds per violation
BASE_BACKOFF_SECONDS: float = 2.0


class RateLimitAgent:
    """Applies sliding-window rate limiting with exponential backoff via Redis.

    Used by the AutonomousRemediator when an agent exceeds safety thresholds
    but doesn't yet warrant full quarantine (score 0.40–0.65).

    Args:
        redis_url: Redis connection URL.
        default_max_requests: Max requests per 1s window for normal agents.
        high_risk_max_requests: Max requests per window for flagged agents.
        violation_threshold: Number of violations before escalating.
        key_prefix: Redis key prefix.

    Example:
        limiter = RateLimitAgent("redis://localhost:6379/0")
        allowed, wait_ms = limiter.check_and_apply("agent_007", is_high_risk=True)
        if not allowed:
            logger.warning("rate_limited", wait_ms=wait_ms)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_max_requests: int = DEFAULT_MAX_REQUESTS,
        high_risk_max_requests: int = HIGH_RISK_MAX_REQUESTS,
        violation_threshold: int = VIOLATION_THRESHOLD,
        key_prefix: str = "argus:ratelimit:",
    ) -> None:
        self._redis_url = redis_url
        self._default_max = default_max_requests
        self._high_risk_max = high_risk_max_requests
        self._violation_threshold = violation_threshold
        self._key_prefix = key_prefix
        self._client: Optional[object] = None

    def _get_client(self) -> object:
        """Lazily create Redis client."""
        if self._client is None:
            try:
                import redis
            except ImportError:
                raise ImportError("Run: pip install redis>=5.0.0")
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _window_key(self, agent_id: str) -> str:
        return f"{self._key_prefix}window:{agent_id}"

    def _violation_key(self, agent_id: str) -> str:
        return f"{self._key_prefix}violations:{agent_id}"

    def check_and_apply(
        self,
        agent_id: str,
        is_high_risk: bool = False,
    ) -> Tuple[bool, float]:
        """Check if agent is within rate limit and record the request.

        Args:
            agent_id: Agent making the request.
            is_high_risk: Use stricter limit for flagged agents.

        Returns:
            Tuple of (allowed: bool, wait_ms: float).
            allowed=False means the request should be blocked.
            wait_ms is how long to wait before retrying (0 if allowed).
        """
        max_requests = self._high_risk_max if is_high_risk else self._default_max

        try:
            client = self._get_client()
            now = time.time()
            window_start = now - WINDOW_SECONDS
            window_key = self._window_key(agent_id)

            pipe = client.pipeline()  # type: ignore[union-attr]
            # Remove expired entries
            pipe.zremrangebyscore(window_key, "-inf", window_start)
            # Count current window requests
            pipe.zcard(window_key)
            results = pipe.execute()

            current_count = results[1]

            if current_count >= max_requests:
                # Record violation
                violation_count = self._increment_violations(agent_id)
                wait_ms = self._compute_backoff_ms(violation_count)
                logger.warning(
                    "rate_limit_exceeded",
                    agent_id=agent_id,
                    current_count=current_count,
                    max_requests=max_requests,
                    violations=violation_count,
                    wait_ms=wait_ms,
                )
                return False, wait_ms

            # Allow: record this request in the window
            pipe2 = client.pipeline()  # type: ignore[union-attr]
            pipe2.zadd(window_key, {str(uuid.uuid4()): now})
            pipe2.expire(window_key, int(WINDOW_SECONDS * 2))
            pipe2.execute()

            return True, 0.0

        except Exception as exc:
            logger.error("rate_limit_check_failed", agent_id=agent_id, error=str(exc))
            # Fail open: allow the request if Redis is unreachable
            return True, 0.0

    def _increment_violations(self, agent_id: str) -> int:
        """Increment violation counter for an agent.

        Args:
            agent_id: Agent to record violation for.

        Returns:
            New violation count.
        """
        try:
            client = self._get_client()
            vkey = self._violation_key(agent_id)
            pipe = client.pipeline()  # type: ignore[union-attr]
            pipe.incr(vkey)
            pipe.expire(vkey, VIOLATION_TTL_SECONDS)
            results = pipe.execute()
            count = int(results[0])
            logger.info(
                "violation_recorded",
                agent_id=agent_id,
                violation_count=count,
                threshold=self._violation_threshold,
            )
            return count
        except Exception as exc:
            logger.error("violation_increment_failed", agent_id=agent_id, error=str(exc))
            return 1

    def _compute_backoff_ms(self, violation_count: int) -> float:
        """Compute exponential backoff in milliseconds.

        Args:
            violation_count: Number of violations so far.

        Returns:
            Backoff duration in milliseconds.
        """
        backoff_s = BASE_BACKOFF_SECONDS * (2 ** min(violation_count - 1, 6))
        return backoff_s * 1000.0

    def get_violation_count(self, agent_id: str) -> int:
        """Get current violation count for an agent.

        Args:
            agent_id: Agent to check.

        Returns:
            Number of violations in the last VIOLATION_TTL_SECONDS.
        """
        try:
            client = self._get_client()
            val = client.get(self._violation_key(agent_id))  # type: ignore[union-attr]
            return int(val) if val else 0
        except Exception:
            return 0

    def is_violation_threshold_exceeded(self, agent_id: str) -> bool:
        """Check if agent has exceeded the violation threshold.

        Args:
            agent_id: Agent to check.

        Returns:
            True if violations >= violation_threshold.
        """
        return self.get_violation_count(agent_id) >= self._violation_threshold

    def reset_violations(self, agent_id: str) -> None:
        """Reset violation counter (called when agent recovers).

        Args:
            agent_id: Agent to reset.
        """
        try:
            client = self._get_client()
            client.delete(self._violation_key(agent_id))  # type: ignore[union-attr]
            logger.info("violations_reset", agent_id=agent_id)
        except Exception as exc:
            logger.error("violation_reset_failed", agent_id=agent_id, error=str(exc))

    def get_current_rate(self, agent_id: str) -> int:
        """Get current request count in the sliding window.

        Args:
            agent_id: Agent to check.

        Returns:
            Number of requests in the current window.
        """
        try:
            client = self._get_client()
            now = time.time()
            window_start = now - WINDOW_SECONDS
            count = client.zcount(  # type: ignore[union-attr]
                self._window_key(agent_id), window_start, now
            )
            return int(count)
        except Exception:
            return 0
