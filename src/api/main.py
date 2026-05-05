"""
ARGUS Public API — FastAPI REST Layer
=======================================
Exposes the full ARGUS safety system over HTTP with API key authentication.
Serves as the single entry point for external consumers (dashboards, CI/CD hooks,
agent framework integrations).

Endpoints:
    POST /monitor                    — Register an LLM for monitoring
    GET  /signals/{agent_id}         — Latest safety signals for an agent
    GET  /predictions/{agent_id}     — Latest oracle predictions for an agent
    POST /remediate                  — Manually trigger remediation
    GET  /audit/{agent_id}           — Remediation audit log
    GET  /health                     — System health check

Auth: X-API-Key header (validated against ARGUS_API_KEYS env var, comma-separated)

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --workers 2
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Auth ──────────────────────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def _load_api_keys() -> set:
    """Load valid API keys from environment variable."""
    raw = os.environ.get("ARGUS_API_KEYS", "")
    if not raw:
        logger.warning("argus_api_keys_not_set_running_open")
        return set()  # Open in dev if not set
    return {k.strip() for k in raw.split(",") if k.strip()}

_VALID_KEYS = _load_api_keys()


async def verify_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> str:
    """Verify API key in X-API-Key header.

    Args:
        api_key: Key extracted from request header.

    Returns:
        The valid API key.

    Raises:
        HTTPException 401: If key is missing or invalid.
    """
    if not _VALID_KEYS:
        return "dev"  # Open in dev mode
    if not api_key or api_key not in _VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifespan — log startup and shutdown."""
    logger.info("argus_api_starting")
    yield
    logger.info("argus_api_shutdown")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ARGUS Safety OS API",
    description="Production AI Safety Operating System — monitoring, prediction, and remediation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# In-memory state stores (replaced by Redis/Spanner in production)
_registered_agents: Dict[str, Dict] = {}
_signal_store: Dict[str, List[Dict]] = {}
_prediction_store: Dict[str, List[Dict]] = {}
_audit_store: Dict[str, List[Dict]] = {}


# ── Request / Response Models ─────────────────────────────────────────────────

class MonitorRequest(BaseModel):
    """Register a new LLM agent for monitoring."""
    agent_id: str = Field(..., description="Unique agent identifier")
    model_name: str = Field(..., description="Model being served (e.g. llama-3.1-8b)")
    endpoint: Optional[str] = Field(default=None, description="Agent's serving endpoint")
    metadata: Dict = Field(default_factory=dict)


class MonitorResponse(BaseModel):
    agent_id: str
    status: str
    registered_at: str


class SafetySignalResponse(BaseModel):
    agent_id: str
    signals: List[Dict]
    count: int


class OraclePredictionResponse(BaseModel):
    agent_id: str
    predictions: List[Dict]
    count: int


class RemediateRequest(BaseModel):
    """Manual remediation trigger."""
    agent_id: str
    safety_score: float = Field(..., ge=0.0, le=1.0)
    risk_category: str = Field(default="borderline")
    modality: str = Field(default="text")
    content_preview: str = Field(default="", max_length=200)
    reason: str = Field(default="manual_trigger")


class RemediateResponse(BaseModel):
    agent_id: str
    action: str
    outcome: str
    latency_ms: float
    audit_record_id: str


class AuditResponse(BaseModel):
    agent_id: str
    records: List[Dict]
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    registered_agents: int
    uptime_s: float


_START_TIME = time.monotonic()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """System health check — no auth required."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        registered_agents=len(_registered_agents),
        uptime_s=time.monotonic() - _START_TIME,
    )


@app.post("/monitor", response_model=MonitorResponse, tags=["Monitoring"])
async def register_agent(
    request: MonitorRequest,
    _key: str = Depends(verify_api_key),
) -> MonitorResponse:
    """Register an LLM agent for real-time safety monitoring.

    Args:
        request: Agent registration details.

    Returns:
        Registration confirmation.
    """
    now = datetime.now(timezone.utc).isoformat()
    _registered_agents[request.agent_id] = {
        "model_name": request.model_name,
        "endpoint": request.endpoint,
        "metadata": request.metadata,
        "registered_at": now,
    }
    _signal_store.setdefault(request.agent_id, [])
    _prediction_store.setdefault(request.agent_id, [])
    _audit_store.setdefault(request.agent_id, [])

    logger.info(
        "agent_registered",
        agent_id=request.agent_id,
        model=request.model_name,
    )

    return MonitorResponse(
        agent_id=request.agent_id,
        status="monitoring",
        registered_at=now,
    )


@app.get("/signals/{agent_id}", response_model=SafetySignalResponse, tags=["Monitoring"])
async def get_signals(
    agent_id: str,
    limit: int = 50,
    _key: str = Depends(verify_api_key),
) -> SafetySignalResponse:
    """Get the latest safety signals for a monitored agent.

    Args:
        agent_id: Agent to query.
        limit: Maximum signals to return.

    Returns:
        List of safety signals in reverse-chronological order.

    Raises:
        HTTPException 404: If agent is not registered.
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not registered")

    signals = _signal_store.get(agent_id, [])[-limit:][::-1]
    return SafetySignalResponse(
        agent_id=agent_id,
        signals=signals,
        count=len(signals),
    )


@app.get("/predictions/{agent_id}", response_model=OraclePredictionResponse, tags=["Oracle"])
async def get_predictions(
    agent_id: str,
    limit: int = 20,
    _key: str = Depends(verify_api_key),
) -> OraclePredictionResponse:
    """Get the latest oracle risk predictions for an agent.

    Args:
        agent_id: Agent to query.
        limit: Maximum predictions to return.

    Returns:
        List of oracle predictions in reverse-chronological order.

    Raises:
        HTTPException 404: If agent is not registered.
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not registered")

    predictions = _prediction_store.get(agent_id, [])[-limit:][::-1]
    return OraclePredictionResponse(
        agent_id=agent_id,
        predictions=predictions,
        count=len(predictions),
    )


@app.post("/remediate", response_model=RemediateResponse, tags=["Remediation"])
async def manual_remediate(
    request: RemediateRequest,
    _key: str = Depends(verify_api_key),
) -> RemediateResponse:
    """Manually trigger the remediation graph for a safety signal.

    This endpoint is used for:
    - Human-in-the-loop interventions
    - Testing remediation flows
    - Backfilling missed signals

    Args:
        request: Remediation trigger details.

    Returns:
        Action taken, outcome, latency, and audit record ID.
    """
    try:
        from src.autonomous_remediator.langgraph_graph import run_remediation
        state = await run_remediation(
            agent_id=request.agent_id,
            safety_score=request.safety_score,
            risk_category=request.risk_category,
            modality=request.modality,
            content_preview=request.content_preview,
        )
    except Exception as exc:
        logger.error("manual_remediation_failed", agent_id=request.agent_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Remediation failed: {exc}")

    # Store in audit log
    audit_record = {
        "action": state["action"],
        "outcome": state["outcome"],
        "latency_ms": state["latency_ms"],
        "safety_score": request.safety_score,
        "reason": request.reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_record_id": state["audit_record_id"],
    }
    _audit_store.setdefault(request.agent_id, []).append(audit_record)

    logger.info(
        "manual_remediation_complete",
        agent_id=request.agent_id,
        action=state["action"],
        latency_ms=f"{state['latency_ms']:.1f}",
    )

    return RemediateResponse(
        agent_id=request.agent_id,
        action=state["action"],
        outcome=state["outcome"],
        latency_ms=state["latency_ms"],
        audit_record_id=state["audit_record_id"],
    )


@app.get("/audit/{agent_id}", response_model=AuditResponse, tags=["Audit"])
async def get_audit_log(
    agent_id: str,
    limit: int = 100,
    _key: str = Depends(verify_api_key),
) -> AuditResponse:
    """Get remediation audit log for an agent.

    Args:
        agent_id: Agent to query.
        limit: Maximum records to return.

    Returns:
        List of audit records in reverse-chronological order.
    """
    records = _audit_store.get(agent_id, [])[-limit:][::-1]
    return AuditResponse(
        agent_id=agent_id,
        records=records,
        count=len(records),
    )


# ── Internal signal ingestion (called by LatentSentinel via gRPC hot path) ──

@app.post("/internal/signal", include_in_schema=False)
async def ingest_signal(
    request: Request,
    _key: str = Depends(verify_api_key),
) -> Dict:
    """Ingest a safety signal from LatentSentinel (internal use).

    Called by the gRPC/HTTP bridge from the sentinel hot path.
    """
    body = await request.json()
    agent_id = body.get("agent_id", "unknown")
    _signal_store.setdefault(agent_id, []).append({
        **body,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    })
    # Keep last 1000 signals per agent
    if len(_signal_store[agent_id]) > 1000:
        _signal_store[agent_id] = _signal_store[agent_id][-1000:]
    return {"status": "ok", "agent_id": agent_id}


@app.post("/internal/prediction", include_in_schema=False)
async def ingest_prediction(
    request: Request,
    _key: str = Depends(verify_api_key),
) -> Dict:
    """Ingest an oracle prediction (internal use).

    Called by the TelemetryConsumer after each oracle inference.
    """
    body = await request.json()
    agent_id = body.get("agent_id", "unknown")
    _prediction_store.setdefault(agent_id, []).append({
        **body,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    })
    if len(_prediction_store[agent_id]) > 200:
        _prediction_store[agent_id] = _prediction_store[agent_id][-200:]
    return {"status": "ok", "agent_id": agent_id}
