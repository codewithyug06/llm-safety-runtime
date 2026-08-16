from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional

import structlog

logger = structlog.get_logger(__name__)
try:
    from prometheus_client import Counter, Histogram, start_http_server as _start_http_server

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

    PROBE_INFERENCE_TOTAL = None 
    HOOK_ERRORS_TOTAL = None 
    RISK_LEVEL_TOTAL = None 
    SLA_VIOLATIONS_TOTAL = None  
    PROBE_LATENCY_MS = None 
    HOOK_EXTRACTION_LATENCY_MS = None 
    COMPOSITE_RISK_SCORE = None 


try:
    from opentelemetry import trace as _otel_trace

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
        pass 
    tracer: _otel_trace.Tracer = _otel_trace.get_tracer("argus.latent_sentinel")
    _OTEL_AVAILABLE = True

except ImportError:
    _OTEL_AVAILABLE = False
    tracer = None
    logger.warning("opentelemetry_unavailable", hint="pip install opentelemetry-api")


def record_probe_inference(
    category: str,
    risk_level: str,
    latency_ms: float,
) -> None:

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

    if not _PROMETHEUS_AVAILABLE:
        return
    HOOK_EXTRACTION_LATENCY_MS.observe(latency_ms)
    logger.debug("hook_extraction_recorded", latency_ms=latency_ms)


def record_composite_score(score: float, risk_level: str) -> None:

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

    if _PROMETHEUS_AVAILABLE:
        SLA_VIOLATIONS_TOTAL.inc()
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
    if _PROMETHEUS_AVAILABLE:
        HOOK_ERRORS_TOTAL.labels(module=module, error_type=error_type).inc()
        _HOOK_ERROR_COUNTER_LEGACY.labels(error_type=error_type).inc()
    logger.error(
        "hook_error_recorded",
        module=module,
        error_type=error_type,
        model_name=model_name,
    )


def get_tracer() -> "Optional[_otel_trace.Tracer]": 

    return tracer if _OTEL_AVAILABLE else None


def record_probe_latency(latency_ms: float, model_name: str = "unknown") -> None:

    if _PROMETHEUS_AVAILABLE:
        _PROBE_LATENCY_LEGACY.labels(model_name=model_name).observe(latency_ms)

    if latency_ms > 15.0:
        record_sla_violation(latency_ms=latency_ms, budget_ms=15.0, model_name=model_name)


def record_risk_level(risk_level: str, model_name: str = "unknown") -> None:

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
    if _PROMETHEUS_AVAILABLE:
        _PROBE_SCORE_HISTOGRAM.labels(
            category=category, model_name=model_name
        ).observe(score)


def record_signal_published(risk_level: str) -> None:

    if _PROMETHEUS_AVAILABLE:
        _SIGNALS_PUBLISHED_COUNTER.labels(risk_level=risk_level).inc()


@contextmanager
def timed_forward_pass(
    model_name: str = "unknown",
    sla_budget_ms: float = 10.0,
) -> Generator[None, None, None]:

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


def start_metrics_server(port: int = 8000) -> None:

    if not _PROMETHEUS_AVAILABLE:
        raise ImportError(
            "prometheus-client not installed: pip install prometheus-client"
        )
    _start_http_server(port)
    logger.info("prometheus_metrics_server_started", port=port)
