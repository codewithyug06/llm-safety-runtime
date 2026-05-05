"""
MOD-01: LatentSentinel Observability
======================================
Prometheus counters/histograms and OpenTelemetry trace spans for the
LatentSentinel module.  Used by ``sentinel.py`` to expose observability data.

Both ``prometheus_client`` and ``opentelemetry`` are treated as optional
dependencies: if either package is absent, no-op fallbacks are used so that
LatentSentinel can run in environments where observability libraries are not
installed.

Prometheus metrics exported
---------------------------
argus_probe_inference_total       — counter per (category, risk_level)
argus_hook_errors_total           — counter per (module, error_type)
argus_risk_level_total            — counter per risk_level
argus_sla_violations_total        — counter (no labels)
argus_probe_latency_ms            — histogram per category
argus_hook_extraction_latency_ms  — histogram (no category label)
argus_composite_risk_score        — histogram (no labels)

OpenTelemetry tracer
--------------------
``argus.latent_sentinel`` — module-level tracer returned by ``get_tracer()``.

Legacy helpers
--------------
The functions ``record_probe_latency``, ``record_risk_level``,
``record_probe_score``, ``record_hook_error`` (single-arg), ``record_sla_violation``,
``record_signal_published``, ``timed_forward_pass``, ``timed_probe``, and
``start_metrics_server`` are preserved for backward compatibility.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional

import structlog

logger = structlog.get_logger(__name__)

# ── Prometheus setup ──────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Histogram, start_http_server as _start_http_server

    # ── Counters ───────────────────────────────────────────────────────────────

    PROBE_INFERENCE_TOTAL: Counter = Counter(
        "argus_probe_inference_total",
        "Total number of probe inference calls, labelled by category and risk level.",
        labelnames=["category", "risk_level"],
    )

    HOOK_ERRORS_TOTAL: Counter = Counter(
        "argus_hook_errors_total",
        "Total number of hook or probe errors, labelled by module and error type.",
        labelnames=["module", "error_type"],
    )

    RISK_LEVEL_TOTAL: Counter = Counter(
        "argus_risk_level_total",
        "Total number of safety signals emitted per risk level.",
        labelnames=["risk_level"],
    )

    SLA_VIOLATIONS_TOTAL: Counter = Counter(
        "argus_sla_violations_total",
        "Total number of times hook-to-signal latency exceeded the SLA budget.",
    )

    # ── Histograms ─────────────────────────────────────────────────────────────

    PROBE_LATENCY_MS: Histogram = Histogram(
        "argus_probe_latency_ms",
        "Per-probe inference latency in milliseconds.",
        labelnames=["category"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50],
    )

    HOOK_EXTRACTION_LATENCY_MS: Histogram = Histogram(
        "argus_hook_extraction_latency_ms",
        "Latency for extracting activations from a single forward hook in milliseconds.",
        buckets=[0.5, 1, 2, 5, 10, 20, 50, 100],
    )

    COMPOSITE_RISK_SCORE: Histogram = Histogram(
        "argus_composite_risk_score",
        "Distribution of composite risk scores produced by RiskAggregator.",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    # ── Legacy metrics (kept for backward compat with older sentinel code) ─────

    _PROBE_LATENCY_LEGACY: Histogram = Histogram(
        "argus_probe_latency_ms_legacy",
        "Legacy hook-to-signal pipeline latency histogram (model_name label).",
        buckets=[1, 2, 3, 5, 7, 10, 15, 20, 50],
        labelnames=["model_name"],
    )

    _RISK_LEVEL_COUNTER_LEGACY: Counter = Counter(
        "argus_risk_level_model_total",
        "Legacy risk level counter with model_name label.",
        labelnames=["risk_level", "model_name"],
    )

    _PROBE_SCORE_HISTOGRAM: Histogram = Histogram(
        "argus_probe_score",
        "Distribution of individual probe risk scores (0–1).",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        labelnames=["category", "model_name"],
    )

    _HOOK_ERROR_COUNTER_LEGACY: Counter = Counter(
        "argus_hook_errors_legacy_total",
        "Legacy hook error counter with only error_type label.",
        labelnames=["error_type"],
    )

    _SLA_VIOLATION_COUNTER_LEGACY: Counter = Counter(
        "argus_sla_violations_model_total",
        "Legacy SLA violation counter with model_name label.",
        labelnames=["model_name"],
    )

    _SIGNALS_PUBLISHED_COUNTER: Counter = Counter(
        "argus_signals_published_total",
        "Number of SafetySignals published to Kafka.",
        labelnames=["risk_level"],
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_unavailable", hint="pip install prometheus-client")

    # No-op sentinel values so module-level names are always defined.
    PROBE_INFERENCE_TOTAL = None  # type: ignore[assignment]
    HOOK_ERRORS_TOTAL = None  # type: ignore[assignment]
    RISK_LEVEL_TOTAL = None  # type: ignore[assignment]
    SLA_VIOLATIONS_TOTAL = None  # type: ignore[assignment]
    PROBE_LATENCY_MS = None  # type: ignore[assignment]
    HOOK_EXTRACTION_LATENCY_MS = None  # type: ignore[assignment]
    COMPOSITE_RISK_SCORE = None  # type: ignore[assignment]


# ── OpenTelemetry setup ───────────────────────────────────────────────────────

try:
    from opentelemetry import trace as _otel_trace

    # Only import SDK components when they are available; the API alone is
    # sufficient for the tracer object used at runtime.
    try:
        from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor as _BatchSpanProcessor,
            ConsoleSpanExporter as _ConsoleSpanExporter,
        )

        _provider = _TracerProvider()
        _provider.add_span_processor(_BatchSpanProcessor(_ConsoleSpanExporter()))
        _otel_trace.set_tracer_provider(_provider)
    except ImportError:
        pass  # opentelemetry-api installed but sdk absent — use noop provider

    tracer: _otel_trace.Tracer = _otel_trace.get_tracer("argus.latent_sentinel")
    _OTEL_AVAILABLE = True

except ImportError:
    _OTEL_AVAILABLE = False
    tracer = None  # type: ignore[assignment]
    logger.warning("opentelemetry_unavailable", hint="pip install opentelemetry-api")


# ── Primary helper functions (new API) ────────────────────────────────────────

def record_probe_inference(
    category: str,
    risk_level: str,
    latency_ms: float,
) -> None:
    """Record a single probe inference call.

    Increments ``argus_probe_inference_total`` and observes
    ``argus_probe_latency_ms`` for the given category.

    Args:
        category: Safety probe category (e.g. ``"jailbreak"``).
        risk_level: Assessed risk level string (e.g. ``"high"``).
        latency_ms: Elapsed probe inference time in milliseconds.

    Example:
        >>> record_probe_inference("hallucination", "medium", 1.23)
    """
    if not _PROMETHEUS_AVAILABLE:
        return
    PROBE_INFERENCE_TOTAL.labels(category=category, risk_level=risk_level).inc()
    PROBE_LATENCY_MS.labels(category=category).observe(latency_ms)
    logger.debug(
        "probe_inference_recorded",
        category=category,
        risk_level=risk_level,
        latency_ms=latency_ms,
    )


def record_hook_extraction(latency_ms: float) -> None:
    """Record the latency of a single activation-extraction hook call.

    Observes ``argus_hook_extraction_latency_ms``.

    Args:
        latency_ms: Elapsed time for the hook extraction in milliseconds.

    Example:
        >>> record_hook_extraction(0.85)
    """
    if not _PROMETHEUS_AVAILABLE:
        return
    HOOK_EXTRACTION_LATENCY_MS.observe(latency_ms)
    logger.debug("hook_extraction_recorded", latency_ms=latency_ms)


def record_composite_score(score: float, risk_level: str) -> None:
    """Record a composite risk score and its corresponding risk level.

    Observes ``argus_composite_risk_score`` and increments
    ``argus_risk_level_total``.

    Args:
        score: Composite risk score in ``[0.0, 1.0]``.
        risk_level: Assessed risk level string (e.g. ``"critical"``).

    Example:
        >>> record_composite_score(0.87, "critical")
    """
    if not _PROMETHEUS_AVAILABLE:
        return
    COMPOSITE_RISK_SCORE.observe(score)
    RISK_LEVEL_TOTAL.labels(risk_level=risk_level).inc()
    logger.debug(
        "composite_score_recorded",
        score=score,
        risk_level=risk_level,
    )


def record_sla_violation(
    latency_ms: float,
    budget_ms: float,
    model_name: str = "unknown",
) -> None:
    """Record a latency SLA violation.

    Increments ``argus_sla_violations_total`` and emits a structured warning
    log entry with the measured and budgeted latency values.

    Args:
        latency_ms: Measured latency in milliseconds.
        budget_ms: Configured SLA budget in milliseconds.
        model_name: Identifier of the monitored model (for log context only).

    Example:
        >>> record_sla_violation(12.5, 10.0, model_name="llama-3.1-8b")
    """
    if _PROMETHEUS_AVAILABLE:
        SLA_VIOLATIONS_TOTAL.inc()
        # Also update legacy counter for dashboards that use model_name label.
        _SLA_VIOLATION_COUNTER_LEGACY.labels(model_name=model_name).inc()
    logger.warning(
        "sla_violation",
        latency_ms=latency_ms,
        budget_ms=budget_ms,
        model_name=model_name,
    )


def record_hook_error(
    module: str,
    error_type: str,
    model_name: str = "unknown",
) -> None:
    """Record a hook or probe error.

    Increments ``argus_hook_errors_total`` labelled by module and error type.

    Args:
        module: ARGUS sub-module where the error originated
                (e.g. ``"latent_sentinel"``).
        error_type: Short name of the exception class
                    (e.g. ``"HookRegistrationError"``).
        model_name: Model identifier (for legacy counter and log context only).

    Example:
        >>> record_hook_error("latent_sentinel", "ProbeInferenceError")
    """
    if _PROMETHEUS_AVAILABLE:
        HOOK_ERRORS_TOTAL.labels(module=module, error_type=error_type).inc()
        # Legacy single-label counter for backward compat
        _HOOK_ERROR_COUNTER_LEGACY.labels(error_type=error_type).inc()
    logger.error(
        "hook_error_recorded",
        module=module,
        error_type=error_type,
        model_name=model_name,
    )


def get_tracer() -> "Optional[_otel_trace.Tracer]":  # type: ignore[name-defined]
    """Return the module-level OpenTelemetry tracer.

    Returns ``None`` when ``opentelemetry-api`` is not installed, so callers
    must guard usage:

    Example:
        >>> t = get_tracer()
        >>> if t:
        ...     with t.start_as_current_span("my.span"):
        ...         pass

    Returns:
        The ``argus.latent_sentinel`` tracer, or ``None`` if OTEL is absent.
    """
    return tracer if _OTEL_AVAILABLE else None


# ── Legacy helper functions (backward compat) ─────────────────────────────────

def record_probe_latency(latency_ms: float, model_name: str = "unknown") -> None:
    """Record hook-to-signal pipeline latency (legacy API).

    Also triggers an SLA check against the 15 ms p99 hard limit.

    Args:
        latency_ms: Measured latency in milliseconds.
        model_name: Name of the monitored model.
    """
    if _PROMETHEUS_AVAILABLE:
        _PROBE_LATENCY_LEGACY.labels(model_name=model_name).observe(latency_ms)

    if latency_ms > 15.0:
        record_sla_violation(latency_ms=latency_ms, budget_ms=15.0, model_name=model_name)


def record_risk_level(risk_level: str, model_name: str = "unknown") -> None:
    """Increment the risk-level counter (legacy API, includes model_name label).

    Args:
        risk_level: String value from RiskLevel enum (e.g. ``"critical"``).
        model_name: Name of the monitored model.
    """
    if _PROMETHEUS_AVAILABLE:
        _RISK_LEVEL_COUNTER_LEGACY.labels(
            risk_level=risk_level, model_name=model_name
        ).inc()
        RISK_LEVEL_TOTAL.labels(risk_level=risk_level).inc()


def record_probe_score(
    score: float,
    category: str,
    model_name: str = "unknown",
) -> None:
    """Record an individual probe risk score (legacy API).

    Args:
        score: Probe output score in ``[0, 1]``.
        category: Probe category name (e.g. ``"hallucination"``).
        model_name: Name of the monitored model.
    """
    if _PROMETHEUS_AVAILABLE:
        _PROBE_SCORE_HISTOGRAM.labels(
            category=category, model_name=model_name
        ).observe(score)


def record_signal_published(risk_level: str) -> None:
    """Increment the Kafka publish counter (legacy API).

    Args:
        risk_level: Risk level of the published signal.
    """
    if _PROMETHEUS_AVAILABLE:
        _SIGNALS_PUBLISHED_COUNTER.labels(risk_level=risk_level).inc()


# ── Context managers (legacy API, kept for sentinel.py) ──────────────────────

@contextmanager
def timed_forward_pass(
    model_name: str = "unknown",
    sla_budget_ms: float = 10.0,
) -> Generator[None, None, None]:
    """Context manager that times a forward pass and records latency metrics.

    Also creates an OpenTelemetry span when OTEL is available.

    Args:
        model_name: Model identifier for metric labels.
        sla_budget_ms: SLA budget; logs a warning if exceeded.

    Example:
        >>> with timed_forward_pass(model_name="llama-3.1-8b"):
        ...     _ = model(input_ids)
    """
    if _OTEL_AVAILABLE and tracer is not None:
        with tracer.start_as_current_span("argus.latent_sentinel.forward_pass") as span:
            span.set_attribute("model_name", model_name)
            t0 = time.perf_counter()
            try:
                yield
            finally:
                latency_ms = (time.perf_counter() - t0) * 1_000
                span.set_attribute("latency_ms", latency_ms)
                record_probe_latency(latency_ms, model_name=model_name)
    else:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - t0) * 1_000
            record_probe_latency(latency_ms, model_name=model_name)


@contextmanager
def timed_probe(
    category: str,
    model_name: str = "unknown",
) -> Generator[None, None, None]:
    """Context manager that times a single probe inference pass.

    Args:
        category: Probe category being timed.
        model_name: Model identifier for metric labels.

    Example:
        >>> with timed_probe("jailbreak", model_name="llama-3.1-8b"):
        ...     score = probe.infer(activations)
    """
    if _OTEL_AVAILABLE and tracer is not None:
        with tracer.start_as_current_span("argus.probe.inference") as span:
            span.set_attribute("category", category)
            span.set_attribute("model_name", model_name)
            t0 = time.perf_counter()
            try:
                yield
            finally:
                latency_ms = (time.perf_counter() - t0) * 1_000
                PROBE_LATENCY_MS.labels(category=category).observe(latency_ms)
    else:
        yield


# ── Prometheus HTTP server ────────────────────────────────────────────────────

def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to expose ``/metrics`` on. Defaults to 8000.

    Raises:
        ImportError: If ``prometheus-client`` is not installed.

    Example:
        >>> start_metrics_server(port=9090)
    """
    if not _PROMETHEUS_AVAILABLE:
        raise ImportError(
            "prometheus-client not installed: pip install prometheus-client"
        )
    _start_http_server(port)
    logger.info("prometheus_metrics_server_started", port=port)
