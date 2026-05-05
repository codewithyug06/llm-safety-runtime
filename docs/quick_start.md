# ARGUS Quick Start Guide

Get ARGUS running in under 5 minutes using stub models (no GPU required).

---

## Prerequisites

- Python 3.11+
- Git
- 4GB+ RAM (16GB+ for full GPU stack)
- Optional: CUDA 12.x + NVIDIA GPU (for production inference)

---

## Step 1 — Clone and Install

```bash
git clone https://github.com/argus-ai/argus.git
cd argus

# Install production dependencies
pip install -r requirements.txt

# Install development/test dependencies
pip install -r requirements-dev.txt
```

**Verify the install:**

```bash
python -c "import torch, structlog, fastapi; print('Dependencies OK')"
```

---

## Step 2 — Configure Environment

Copy the example env file and add your credentials:

```bash
cp .env.example .env
```

For local development, the only required key is:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...        # Required for AutonomousRemediator
MLFLOW_TRACKING_URI=http://localhost:5000  # Optional: local MLflow
```

All other variables (Kafka, Redis, Spanner, GCP) are optional — ARGUS runs in
**local simulation mode** without them (in-memory state, no Kafka, no Spanner writes).

---

## Step 3 — Generate Benchmark Data

Generate the oracle training/evaluation dataset (no GPU needed, ~30 seconds):

```bash
# Generate fault injection dataset (MOD-05: PredictiveOracle)
python scripts/generate_fault_injection_dataset.py

# Build safety critic evaluation dataset (MOD-03: OmniSafetyCritic)
python scripts/build_safety_dataset.py
```

You should see:
```
data/oracle/fault_injection_train.npz   (1050 episodes)
data/oracle/fault_injection_val.npz     (225 episodes)
data/oracle/fault_injection_test.npz    (225 episodes)
data/safety_critic/train.jsonl
data/safety_critic/val.jsonl
data/safety_critic/test.jsonl
```

---

## Step 4 — Run the Test Suite

```bash
# Unit tests (fast, no I/O dependencies)
pytest tests/unit/ -v

# End-to-end tests (uses in-memory mocks — no real Kafka/Redis/Spanner needed)
pytest tests/e2e/ -v
```

Expected output:
```
tests/unit/latent_sentinel/test_sentinel.py    ........ PASSED
tests/unit/causal_engine/test_intervention.py  ........ PASSED
tests/unit/safety_critic/test_critic.py        ........ PASSED
tests/unit/federated_rlhf/test_client.py       ........ PASSED
tests/e2e/test_full_pipeline.py                ........ PASSED

========= X passed in Y.Xs ==========
```

---

## Step 5 — Run the Safety Benchmark

Verify LatentSentinel meets its <10ms p95 latency SLA using a stub model (no GPU):

```bash
python scripts/benchmark_sentinel.py --assert-sla
```

Expected output:
```
=== LatentSentinel Latency Benchmark ===
  p50: 2.1ms
  p95: 6.3ms  ✓ (target <10ms)
  p99: 8.7ms

✓ SLA PASSED
```

---

## Step 6 — Start the REST API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open the interactive API docs: http://localhost:8080/docs

**Test the API:**

```bash
# Health check
curl http://localhost:8080/health

# Register an agent for monitoring
curl -X POST http://localhost:8080/monitor \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "my-llm-agent", "model_name": "llama3-8b", "tags": {"env": "dev"}}'

# Trigger manual remediation
curl -X POST http://localhost:8080/remediate \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "my-llm-agent", "safety_score": 0.85, "modality": "text"}'

# View audit log
curl http://localhost:8080/audit/my-llm-agent
```

---

## Step 7 — Run a Federated Round (Optional)

Simulate a 2-client federated RLHF round (MOD-04):

```bash
python scripts/run_federated_round.py \
  --num-clients 2 \
  --num-rounds 3 \
  --no-dp
```

This runs the full Flower federated learning protocol with simulated clients,
using local DPO training (without differential privacy for development speed).

---

## What's Next

### Train the Oracle (MOD-05)

```bash
python scripts/train_oracle.py \
  --data-dir data/oracle \
  --output-dir models/oracle \
  --num-epochs 30
```

Evaluate on the test set:

```bash
python scripts/eval_oracle.py
```

### Train the Safety Critic (MOD-03)

> Requires GPU (16GB+ VRAM recommended for LLaVA-1.6)

```bash
python scripts/train_safety_critic.py \
  --train-path data/safety_critic/train.jsonl \
  --val-path data/safety_critic/val.jsonl \
  --base-model llava-hf/llava-v1.6-vicuna-7b-hf \
  --output-dir models/safety_critic
```

### Run the Full Benchmark Suite

```bash
python scripts/run_all_benchmarks.py --skip-slow
```

Generates `docs/benchmarks/full_report.md` with all SLA results.

### Deploy to GKE

```bash
# 1. Provision infrastructure
cd src/infra/terraform
terraform init
terraform apply -var-file=environments/prod.tfvars

# 2. Build and push Docker image
docker build -t gcr.io/YOUR_PROJECT/argus:latest .
docker push gcr.io/YOUR_PROJECT/argus:latest

# 3. Apply Kubernetes manifests
kubectl apply -f src/infra/k8s/argus-deployments.yaml

# 4. Verify deployment
kubectl get pods -n argus-production
```

---

## Development Commands

| Command | Description |
|---------|-------------|
| `make check` | Lint (ruff) + type check (mypy) + format check (black/isort) |
| `make test` | All unit + e2e tests with coverage report |
| `make benchmark` | Full benchmark suite (CPU-only steps) |
| `make eval-probes` | Evaluate LatentSentinel probe accuracy |
| `make eval-critic` | Evaluate OmniSafetyCritic precision/latency |
| `make eval-oracle` | Evaluate PredictiveOracle F1 |
| `make eval-federated` | Compare FedDPO vs centralized |
| `make train-critic` | Train OmniSafetyCritic (requires GPU) |
| `make train-oracle` | Train PredictiveOracle |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run all commands from the project root (`argus/`), not from subdirectories.

### `ImportError: cannot import name 'flash_attn'` or similar

Flash attention is optional. Install with:
```bash
pip install flash-attn --no-build-isolation
```

Or disable it in your model config: `use_flash_attention_2=False`

### `KAFKA_BOOTSTRAP_SERVERS not set`

ARGUS runs without Kafka by default (safety signals are logged locally).
To enable Kafka, set `KAFKA_BOOTSTRAP_SERVERS=localhost:9092` in `.env`
and start Kafka with:
```bash
docker run -d -p 9092:9092 apache/kafka:latest
```

### `ANTHROPIC_API_KEY not set`

The AutonomousRemediator uses the Anthropic API for LLM-assisted reasoning in the
`rollback_and_escalate` action path. Without the key, remediation falls back to deterministic routing.
Set `ANTHROPIC_API_KEY=sk-ant-...` in `.env`.

### Tests failing with `FileNotFoundError: data/oracle/...`

Generate the required data first:
```bash
python scripts/generate_fault_injection_dataset.py
python scripts/build_safety_dataset.py
```

---

## Project Structure (key files)

```
argus/
├── src/
│   ├── latent_sentinel/     MOD-01: hooks, probes, Kafka producer
│   ├── causal_engine/       MOD-02: JAX causal scrubbing + interventions
│   ├── safety_critic/       MOD-03: LLaVA DPO critic + vLLM server
│   ├── federated_rlhf/      MOD-04: Flower + DP-SGD client/server
│   ├── predictive_oracle/   MOD-05: PatchTST + telemetry consumer
│   ├── autonomous_remediator/ MOD-06: LangGraph + integrations
│   ├── mlops/               Kubeflow pipeline + MLflow registry
│   ├── infra/               Terraform + K8s + Grafana
│   └── api/                 FastAPI REST layer
├── scripts/                 Training, eval, benchmark CLI scripts
├── tests/
│   ├── unit/                Per-module unit tests (≥80% coverage)
│   └── e2e/                 Full pipeline end-to-end tests
├── configs/                 YAML configuration files
└── docs/
    ├── architecture.md      Full system architecture
    ├── quick_start.md       This file
    └── papers/              Research paper drafts
```

---

## Getting Help

- **Architecture deep-dive:** `docs/architecture.md`
- **Benchmark report:** `docs/benchmarks/full_report.md` (after running benchmarks)
- **Paper:** `docs/papers/latentsentinel_paper.md`
- **Issues:** https://github.com/argus-ai/argus/issues
- **Contributing:** `docs/CONTRIBUTING.md`
