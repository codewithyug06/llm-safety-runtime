"""
MOD-03: OmniSafetyCritic FastAPI Serving
==========================================
FastAPI application wrapping the vLLM-served OmniSafetyCritic.
Exposes a /score endpoint for synchronous and async safety scoring.

Target: p95 < 80ms for text/code scoring on batches of up to 8 inputs.

Run with:
    uvicorn src.safety_critic.server:app --host 0.0.0.0 --port 8001 --workers 4
"""

from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import structlog
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config import load_safety_critic_config
from src.exceptions import CriticInferenceTimeoutError, CriticServingError
from src.safety_critic.critic import (
    ContentModality,
    CriticInput,
    CriticOutput,
    OmniSafetyCriticModel,
    _parse_safety_score,
)

logger = structlog.get_logger(__name__)

# ── Global model state (loaded once on startup) ───────────────────────────────

_critic_model: Optional[OmniSafetyCriticModel] = None
_cfg = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan handler — loads model on startup, releases on shutdown."""
    global _critic_model, _cfg
    _cfg = load_safety_critic_config()

    logger.info("loading_safety_critic_server", model=_cfg.base_model.name)
    _critic_model = OmniSafetyCriticModel(
        model_name=_cfg.base_model.name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )
    _critic_model.load()
    logger.info("safety_critic_server_ready")

    yield

    logger.info("safety_critic_server_shutdown")
    _critic_model = None


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ARGUS OmniSafetyCritic",
    description="Multimodal safety scoring API for production LLM monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """Single-input scoring request."""

    content: str = Field(..., min_length=1, max_length=8192)
    modality: str = Field(default="text", description="text / code / image / structured")
    context: str = Field(default="", max_length=2048)
    image_b64: Optional[str] = Field(default=None, description="Base64-encoded image bytes")
    agent_id: str = Field(default="")

    @field_validator("modality")
    @classmethod
    def validate_modality(cls, v: str) -> str:
        allowed = {m.value for m in ContentModality}
        if v not in allowed:
            raise ValueError(f"modality must be one of {allowed}")
        return v


class ScoreResponse(BaseModel):
    """Single-input scoring response."""

    safety_score: float = Field(..., ge=0.0, le=1.0)
    risk_category: str
    modality: str
    latency_ms: float
    model_version: str


class BatchScoreRequest(BaseModel):
    """Batch scoring request (up to 8 inputs)."""

    inputs: List[ScoreRequest] = Field(..., min_length=1, max_length=8)


class BatchScoreResponse(BaseModel):
    """Batch scoring response."""

    results: List[ScoreResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return HealthResponse(
        status="healthy" if _critic_model and _critic_model.is_loaded() else "loading",
        model_loaded=_critic_model is not None and _critic_model.is_loaded(),
        model_name=_cfg.base_model.name if _cfg else "unknown",
    )


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    """Score a single agent output for safety.

    Args:
        request: The content to evaluate.

    Returns:
        Safety score in [0.0, 1.0] with risk category and latency.

    Raises:
        HTTPException 503: If model is not ready.
        HTTPException 408: If inference times out.
        HTTPException 500: On unexpected inference error.
    """
    if _critic_model is None or not _critic_model.is_loaded():
        raise HTTPException(status_code=503, detail="Safety critic model not ready")

    image_bytes: Optional[bytes] = None
    if request.image_b64:
        try:
            image_bytes = base64.b64decode(request.image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    critic_input = CriticInput(
        content=request.content,
        modality=ContentModality(request.modality),
        context=request.context,
        image_bytes=image_bytes,
        agent_id=request.agent_id,
    )

    try:
        output: CriticOutput = _critic_model.score(critic_input)
    except CriticInferenceTimeoutError as exc:
        raise HTTPException(status_code=408, detail=str(exc))
    except CriticServingError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(
        "score_request_complete",
        agent_id=request.agent_id,
        modality=request.modality,
        score=f"{output.safety_score:.3f}",
        risk=output.risk_category,
        latency_ms=f"{output.latency_ms:.1f}",
    )

    return ScoreResponse(
        safety_score=output.safety_score,
        risk_category=output.risk_category,
        modality=request.modality,
        latency_ms=output.latency_ms,
        model_version=output.model_version,
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(request: BatchScoreRequest) -> BatchScoreResponse:
    """Score a batch of agent outputs for safety.

    Args:
        request: Up to 8 inputs to evaluate.

    Returns:
        List of safety scores with total batch latency.
    """
    if _critic_model is None or not _critic_model.is_loaded():
        raise HTTPException(status_code=503, detail="Safety critic model not ready")

    t0 = time.perf_counter()
    results: List[ScoreResponse] = []

    for req in request.inputs:
        image_bytes = None
        if req.image_b64:
            image_bytes = base64.b64decode(req.image_b64)

        critic_input = CriticInput(
            content=req.content,
            modality=ContentModality(req.modality),
            context=req.context,
            image_bytes=image_bytes,
            agent_id=req.agent_id,
        )
        try:
            output = _critic_model.score(critic_input)
            results.append(ScoreResponse(
                safety_score=output.safety_score,
                risk_category=output.risk_category,
                modality=req.modality,
                latency_ms=output.latency_ms,
                model_version=output.model_version,
            ))
        except Exception as exc:
            logger.error("batch_item_failed", error=str(exc))
            results.append(ScoreResponse(
                safety_score=0.5,
                risk_category="borderline",
                modality=req.modality,
                latency_ms=0.0,
                model_version="error",
            ))

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info("batch_score_complete", batch_size=len(request.inputs), total_ms=f"{total_ms:.1f}")

    return BatchScoreResponse(results=results, total_latency_ms=total_ms)
