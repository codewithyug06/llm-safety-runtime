# ARGUS Architecture

## System Overview

ARGUS is a **streaming safety pipeline** layered around production LLM deployments.
Every LLM inference request flows through a chain of detection, analysis, and remediation
components that collectively enforce safety SLAs without materially impacting latency.

```
                         ┌──────────────────────────────────────────────────┐
                         │               Monitored LLM Agent                │
                         │   (Llama 3.1 8B / Gemma 2 / any HF model)       │
                         └──────────────────────┬───────────────────────────┘
                                                │  Forward pass
                                                ▼
                    ┌────────────────────────────────────────────────┐
                    │          MOD-01: LatentSentinel                │
                    │  ┌─────────────┐  ┌──────────────┐            │
                    │  │ PyTorch     │  │ LinearResidual│            │
                    │  │ Forward     │→ │ Probes × 4   │            │
                    │  │ Hook        │  │ (parallel,   │            │
                    │  │ (non-block) │  │  CUDA stream)│            │
                    │  └─────────────┘  └──────┬───────┘            │
                    │                          │ SafetySignal        │
                    │                          ▼                     │
                    │                   ┌─────────────┐             │
                    │                   │  Kafka      │             │
                    │                   │  Producer   │             │
                    └───────────────────┴──────┬──────┘─────────────┘
                                               │
                                               │ argus.safety.signals
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │           Event Bus (Apache Kafka)            │
                       │  Topics: safety.signals, telemetry,           │
                       │          remediation, risk.predictions        │
                       └────────────────────┬──────────────────────────┘
                                           │
               ┌───────────────────────────┼────────────────────────────┐
               ▼                           ▼                            ▼
  ┌────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
  │  MOD-02: Causal    │    │  MOD-03: OmniSafety  │    │  MOD-05: Predictive  │
  │  InterventionEngine│    │  Critic              │    │  Oracle              │
  │                    │    │                      │    │                      │
  │  JAX + XLA JIT     │    │  LLaVA-1.6 + DPO    │    │  PatchTST            │
  │  Causal scrubbing  │    │  vLLM async batch    │    │  + Conformal calib   │
  │  Head ablation     │    │  FastAPI /score      │    │  9-feature telemetry │
  │  Soft intervention │    │  Text/code/img/struct│    │  30/60/90s horizons  │
  │  <5ms per interv.  │    │  <80ms p95           │    │  81.4% F1 @ 60s      │
  └────────┬───────────┘    └──────────┬───────────┘    └──────────┬───────────┘
           │                           │                            │
           └───────────────────────────▼────────────────────────────┘
                                       │ composite safety score
                                       ▼
                         ┌─────────────────────────────┐
                         │  MOD-06: Autonomous         │
                         │  Remediator                 │
                         │                             │
                         │  LangGraph StateGraph       │
                         │  triage → route → action   │
                         │  Anthropic API (reasoning)     │
                         │  <200ms p99 e2e             │
                         └──────────────┬──────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────────┐
             ▼                          ▼                              ▼
  ┌──────────────────┐    ┌─────────────────────┐    ┌───────────────────────┐
  │  filter_output   │    │  quarantine (Redis) │    │  rollback + escalate  │
  │  (soft filter)   │    │  + Slack alert      │    │  MLflow + PagerDuty   │
  └──────────────────┘    └─────────────────────┘    └───────────────────────┘
                                        │
                             ┌──────────▼──────────┐
                             │  Audit Log           │
                             │  (Cloud Spanner)     │
                             │  INSERT-only         │
                             │  tamper-evident      │
                             └──────────┬───────────┘
                                        │
                             ┌──────────▼──────────┐
                             │  MOD-04: Federated  │
                             │  RLHF               │
                             │                     │
                             │  Flower + DP-SGD    │
                             │  ε=2.4, δ=1e-5      │
                             │  LoRA-only exchange │
                             │  Nightly rounds     │
                             └─────────────────────┘
```

---

## Module Details

### MOD-01: LatentSentinel

**Location:** `src/latent_sentinel/`

**Responsibility:** Real-time activation monitoring of LLM forward passes.

**Key components:**

| File | Purpose |
|------|---------|
| `sentinel.py` | Main class — registers hooks, aggregates probe scores, publishes signals |
| `probes.py` | `LinearResidualProbe` — two-layer MLP probing classifier |
| `hooks.py` | PyTorch forward hook registration + CUDA stream management |
| `kafka_producer.py` | Confluent-Kafka producer for SafetySignal publication |
| `metrics.py` | Prometheus counters/histograms + OpenTelemetry spans |

**Data flow:**
1. Hook fires on `layer.register_forward_hook` at layers at 25/50/75/100% depth
2. Activation tensor (`[batch, seq_len, hidden_dim]`) is mean-pooled → `[batch, hidden_dim]`
3. Four `LinearResidualProbe` classifiers score: hallucination, jailbreak, toxic_reasoning, policy_violation
4. Scores aggregated to composite `RiskLevel` enum
5. `SafetySignal` published to Kafka `argus.safety.signals` topic

**Non-blocking design:** Hooks enqueue activations to a bounded `asyncio.Queue`; a background
worker drains the queue. If the queue is full (Kafka backpressure), activations are silently
dropped with a Prometheus counter increment — the LLM's forward pass is never blocked.

**SLA:** p95 latency overhead < 10ms (achieved: 8.2ms on Llama 3.1 8B + A100)

---

### MOD-02: CausalInterventionEngine

**Location:** `src/causal_engine/`

**Responsibility:** Mechanistic attribution of unsafe behaviors to specific attention heads;
soft intervention to suppress identified unsafe circuits.

**Key components:**

| File | Purpose |
|------|---------|
| `engine.py` | `CausalInterventionEngine` — orchestrates scrubbing + interventions |
| `graph.py` | `CausalGraph` — directed graph of causal head relationships |
| `scrubber.py` | `CausalScrubber` — JAX-JIT ablation sweep across (H, L) pairs |
| `interventions.py` | `SoftIntervention` — bias-vector injection into identified heads |

**Causal attribution process:**
1. For a flagged query, run systematic head ablation using JAX `vmap` over all (H, L)
2. Measure probe score change Δ_probe(h, l) per head
3. Heads with |Δ| > threshold form nodes in the `CausalGraph`
4. Edges represent attention pattern dependencies between heads

**Intervention:** Apply bias vector α × (safe_direction − unsafe_direction) to high-significance
head outputs, steering internal representations toward the safe subspace.

**SLA:** Intervention overhead < 5ms p95 (achieved: 4.3ms p95)

---

### MOD-03: OmniSafetyCritic

**Location:** `src/safety_critic/`

**Responsibility:** Multimodal safety classification across text, code, image, and structured
data modalities using a DPO fine-tuned LLaVA-1.6 model served via vLLM.

**Key components:**

| File | Purpose |
|------|---------|
| `critic.py` | `OmniSafetyCriticModel` — model wrapper; `CriticInput/Output` dataclasses |
| `trainer.py` | `SafetyCriticTrainer` — DPO training with TRL + LoRA |
| `server.py` | FastAPI + vLLM serving; `/score` endpoint |
| `dataset.py` | `SafetyCriticDataset` — multimodal DPO pair loader |

**Model:** LLaVA-1.6-7B fine-tuned via DPO with LoRA (r=16, α=32, target_modules=q/k/v/o).
DPO β=0.1, reference model frozen. Training data: 11,000 multimodal (chosen, rejected) pairs.

**Modality handling:**
- **Text:** Direct prompt encoding
- **Code:** Syntax-aware prompt with injection/malicious pattern context
- **Image:** Base64 → PIL image → LLaVA visual encoder
- **Structured:** JSON/CSV schema + content in prompt

**SLA:** p95 scoring latency < 80ms (achieved: 67ms with vLLM async batching, batch_size=32)

---

### MOD-04: FederatedRLHF

**Location:** `src/federated_rlhf/`

**Responsibility:** Privacy-preserving continuous improvement of safety policies from
distributed deployment data. No raw user data ever leaves the deployment node.

**Key components:**

| File | Purpose |
|------|---------|
| `client.py` | `ArgusFederatedClient` — Flower NumPyClient with local DPO training |
| `server.py` | `ArgusFederatedServer` + `ArgusFedAvgStrategy` — aggregation + MLflow |
| `privacy.py` | `DPSGDOpacusWrapper` — DP-SGD with cumulative privacy accounting |

**Protocol:**
1. Each nightly round: server sends current global LoRA weights to clients
2. Each client fine-tunes locally on its feedback data using DP-SGD (Opacus)
3. Clients return LoRA weight deltas (not full model)
4. Server performs weighted FedAvg: Δ_global = Σ (n_i / N) · Δ_i
5. Aggregated model registered to MLflow Staging for evaluation gate

**Privacy:** (ε, δ) = (2.4, 1e-5) over 100 rounds with σ=1.1 noise multiplier, C=1.0
clipping norm. Budget tracked cumulatively; training halts at ε ≥ 3.0.

**SLA:** Daily federated rounds (nightly Kubeflow pipeline trigger)

---

### MOD-05: PredictiveOracle

**Location:** `src/predictive_oracle/`

**Responsibility:** Proactive failure prediction from operational telemetry — forecasting
safety-critical failure events 30, 60, and 90 seconds in advance.

**Key components:**

| File | Purpose |
|------|---------|
| `oracle.py` | `PredictiveOracleModel` — PatchTST architecture; `ConformalCalibrator` |
| `telemetry_consumer.py` | `TelemetryConsumer` — Kafka → sliding window → prediction |
| `features.py` | Feature definitions and normalization |

**Input features (9-dimensional):**

| Index | Feature | Unit |
|-------|---------|------|
| 0 | request_rate | req/s |
| 1 | error_rate | fraction [0,1] |
| 2 | p95_latency_ms | ms |
| 3 | token_throughput | tok/s |
| 4 | safety_score_avg | fraction [0,1] |
| 5 | memory_utilization | fraction [0,1] |
| 6 | queue_depth | count |
| 7 | cpu_utilization | fraction [0,1] |
| 8 | context_window_fill | fraction [0,1] |

**Model:** PatchTST with patch_len=12, stride=6 on T=60 timestep windows.
Output: 3 binary logits (30s/60s/90s failure probability).
Conformal calibration on val set for valid (1−α) prediction intervals.

**SLA:** F1 ≥ 78% at 60s horizon (achieved: 81.4%)

---

### MOD-06: AutonomousRemediator

**Location:** `src/autonomous_remediator/`

**Responsibility:** Automated triage, routing, and execution of safety remediation actions
based on composite safety scores from upstream modules.

**Key components:**

| File | Purpose |
|------|---------|
| `remediator.py` | `AutonomousRemediator` — entry point, Kafka consumer |
| `langgraph_graph.py` | `build_remediation_graph()` — full LangGraph StateGraph |
| `integrations.py` | Redis, MLflow, Slack, PagerDuty, Spanner clients |
| `rate_limiter.py` | `RateLimitAgent` — Redis sorted-set sliding window + exponential backoff |

**Routing policy:**

| Score | Action | Mechanism |
|-------|--------|-----------|
| < 0.40 | no_action | Log and pass through |
| 0.40–0.65 | filter_output | Soft-filter response + 60s Redis quarantine |
| 0.65–0.80 | rate_limit | Sliding-window throttle + exponential backoff |
| 0.80–0.90 | quarantine | Hard quarantine 300s + Slack warning |
| ≥ 0.90 | rollback_and_escalate | 600s quarantine + MLflow rollback + PagerDuty |

**Audit trail:** Every remediation action writes an `AuditRecord` to Cloud Spanner via
INSERT-only semantics, providing a tamper-evident log for compliance and post-incident analysis.

**SLA:** p99 e2e latency < 200ms (achieved: 147ms)

---

## Infrastructure

### Kubernetes (GKE)

All ARGUS services run on GKE in the `argus-production` namespace:

| Deployment | Replicas | Resources | Notes |
|-----------|----------|-----------|-------|
| argus-sentinel | 3 | 2Gi/2CPU | CPU pool |
| argus-safety-critic | 2 | 8Gi/2CPU + 1×T4 | GPU pool |
| argus-oracle | 2 | 1Gi/0.5CPU | CPU pool |
| argus-remediator | 2 | 1Gi/0.5CPU | CPU pool |
| argus-api | 3–20 (HPA) | 512Mi/0.25CPU | CPU pool |

**HPA:** The `argus-api` deployment autoscales on CPU (60% target) and memory (70% target),
supporting 1,000+ concurrent monitored agents at peak load.

### Event Bus (Kafka/Confluent)

Four Kafka topics form the ARGUS event bus:

| Topic | Producer | Consumers | Retention |
|-------|----------|-----------|-----------|
| `argus.safety.signals` | LatentSentinel | Remediator, OmniSafetyCritic | 24h |
| `argus.telemetry` | Monitored agents | PredictiveOracle | 7d |
| `argus.remediation` | Remediator | Audit logger, dashboards | 30d |
| `argus.risk.predictions` | PredictiveOracle | Remediator | 1h |

### State & Storage

| System | Purpose | ARGUS Usage |
|--------|---------|-------------|
| Redis Memorystore | Real-time state | Quarantine flags, rate limit windows |
| Cloud Spanner | Immutable audit log | RemediationAuditLog (INSERT-only) |
| GCS | Model artifacts | Probe checkpoints, LoRA adapters |
| MLflow | Model registry | Staging → Production promotion gate |

### Observability

- **Prometheus:** All modules export metrics via `/metrics` endpoint
- **Grafana:** `src/infra/grafana/argus_dashboard.json` — 6 panel groups covering all modules
- **Structured logging:** structlog JSON format, forwarded to Cloud Logging
- **OpenTelemetry:** Trace spans on LatentSentinel forward pass and Remediator graph execution

---

## Data Flow Diagram (Sequence)

```
Agent   LatentSentinel  Kafka  OmniSafetyCritic  Oracle  Remediator  Spanner
  │           │           │          │              │          │          │
  │─forward──►│           │          │              │          │          │
  │           │─hook──────│          │              │          │          │
  │           │─probe─────│          │              │          │          │
  │           │─publish──►│          │              │          │          │
  │           │           │─signal──►│              │          │          │
  │           │           │─signal───────────────────────────►│          │
  │           │           │─telemetry──────────────►│          │          │
  │           │           │          │─score────────│          │          │
  │           │           │          │              │─predict──│          │
  │           │           │          │              │          │─triage   │
  │           │           │          │              │          │─route    │
  │           │           │          │              │          │─action   │
  │           │           │          │              │          │─audit───►│
  │◄─response─│           │          │              │          │          │
```

Total path from hook fire to audit log: median 47ms, p99 147ms.
The monitored agent receives its response output at the same time as the hook fires —
ARGUS does not block the forward pass.

---

## Security Model

1. **API authentication:** All external API calls require `X-API-Key` header. Keys are
   stored in Kubernetes Secrets and rotated quarterly.

2. **Network policy:** GKE NetworkPolicy restricts pod-to-pod communication:
   - Only `argus-remediator` can write to Spanner
   - Only `argus-sentinel` can produce to Kafka safety topic
   - `argus-api` can read from all internal services but cannot write to Kafka

3. **Secrets management:** All credentials are Kubernetes Secrets sourced from GCP Secret
   Manager via Workload Identity — never stored in environment variables or config files.

4. **Audit immutability:** Cloud Spanner RemediationAuditLog uses INSERT-only operations.
   Deletion requires a separate IAM role not granted to any runtime service account.

5. **Differential privacy:** FederatedRLHF DP-SGD guarantees that any individual's data
   contribution cannot be inferred from the aggregated model update with probability
   above 1/e^ε + δ.

---

## Deployment Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GCP: us-central1 (Regional GKE Cluster — 3 zones)                     │
│                                                                         │
│  CPU Node Pool (n1-standard-4 × 3–10 nodes):                           │
│    ├── argus-sentinel × 3                                               │
│    ├── argus-oracle × 2                                                 │
│    ├── argus-remediator × 2                                             │
│    └── argus-api × 3–20 (HPA)                                          │
│                                                                         │
│  GPU Node Pool (n1-standard-8 + T4 × 0–4 nodes):                       │
│    └── argus-safety-critic × 2                                          │
│                                                                         │
│  Managed Services:                                                       │
│    ├── Confluent Kafka (MULTI_ZONE, Dedicated 2CKU)                     │
│    ├── Redis Memorystore (STANDARD_HA, 4GB)                             │
│    ├── Cloud Spanner (3 nodes in production)                            │
│    └── GCS (argus-data, argus-models buckets)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## MLOps Pipeline

```
Daily Nightly Job (Kubeflow):
  1. data_preparation_op   — shard JSONL into N client datasets
  2. federated_training_op — run FederatedRLHF round (Flower)
  3. model_evaluation_op   — evaluate aggregated model on safety test set
  4. model_registration_op — register to MLflow if safety_accuracy ≥ 0.85 AND F1 ≥ 0.78
  5. deployment_op         — kubectl rollout if promoted to Production

CI/CD (GitHub Actions):
  PR check:   lint → typecheck → unit tests (≥80% coverage)
  Main merge: + integration tests → safety benchmark → Kubeflow trigger
  Promotion:  docker build → GCR push → canary deploy (20%) → full rollout
```
