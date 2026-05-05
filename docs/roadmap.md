# ARGUS Build Roadmap

> Updated every time a month's milestones are completed.
> This is the file the build system reads to know "what to build next."

## Current Phase: Month 6 — Polish, Benchmark, Publish

**Status: 🔄 In Progress**
**Started:** 2026-05-03

---

## Completed Phases

### Month 1 — Foundation ✅
- [x] PyTorch forward hooks + LatentSentinel core classes
- [x] HookManager, ProbeRegistry, LatentSentinel
- [x] Unit tests for LatentSentinel (37 tests)
- [x] Kafka producer (`src/latent_sentinel/kafka_producer.py`)
- [x] Prometheus + OpenTelemetry (`src/latent_sentinel/metrics.py`)
- [x] Probes trained on Llama 3.1 8B activations — all 4 probes 1.0000 AUC ✅
- [x] Benchmark: hook overhead < 10ms p95 (actual: 0.37ms) ✅

### Month 2 — CausalInterventionEngine ✅
- [x] CausalInterventionEngine core (`src/causal_engine/intervention.py`)
- [x] 37 unit tests — 37/37 passing
- [x] `scripts/run_causal_scrubbing.py` — 32-layer ablation, 128 causal edges
- [x] `scripts/verify_interventions.py` — p95 0.55ms < 5ms SLA ✅
- [x] `models/causal/llama31_8b_causal_graph.json` — saved

---

---

## Month 1: Foundation — Monitoring Infrastructure

### Goals
Stand up the core telemetry and monitoring backbone.
Get the plumbing right before any serious ML work.

### Tasks
- [ ] GKE cluster setup (Terraform) — 3 nodes, T4 GPU pool
- [ ] Deploy Llama 3.1 8B via vLLM on GKE
- [x] Implement PyTorch forward hooks (`src/latent_sentinel/sentinel.py`)
- [x] `HookManager`, `ProbeRegistry`, `LatentSentinel` core classes
- [x] Unit tests for LatentSentinel (TDD complete)
- [ ] Kafka telemetry pipeline: agent request → topic → consumer group
- [ ] Prometheus + Grafana: token rates, latency, GPU utilization dashboards
- [ ] Train first linear probing classifier (hallucination detection) on TruthfulQA
- [ ] Benchmark: hook overhead < 10ms p95

### Deliverables
1. `src/latent_sentinel/` — complete with tests ✅ (core classes)
2. `src/infra/terraform/` — GKE cluster + Kafka + Redis
3. `src/infra/k8s/` — K8s manifests for all services
4. `configs/latent_sentinel.yaml` — probe config
5. First benchmark report: `docs/benchmarks/sentinel_month1.md`

---

## Month 2: LatentSentinel + CausalInterventionEngine

### Goals
Build full interpretability core — all 4 probes + JAX causal pipeline.

### Tasks
- [x] CausalInterventionEngine core (`src/causal_engine/intervention.py`)
- [ ] Train all 4 probing classifiers with proper datasets
- [ ] CUDA-stream parallelism for sub-10ms multi-probe inference
- [ ] Causal scrubbing experiments: identify top-5 unsafe heads on Llama 3.1
- [ ] Soft weight scaling intervention (no reload test)
- [ ] Publish causal head benchmark results on SafetyBench

---

## Month 3: OmniSafetyCritic — Multimodal RLHF  ✅ COMPLETE (2026-05-03)

### Tasks
- [x] `src/safety_critic/critic.py` — OmniSafetyCriticModel (log-prob DPO scoring), Client, Dataset
- [x] `src/safety_critic/trainer.py` — SafetyCriticTrainer (TRL DPOTrainer, QLoRA 4-bit, LoRA r=8)
- [x] `src/safety_critic/server.py` — FastAPI /score + /score/batch, health, vLLM serving
- [x] `scripts/build_safety_dataset.py` — 1,622 DPO pairs: text/code/structured splits
- [x] `scripts/train_safety_critic.py` — CLI with --dry-run, --load-in-4bit, MLflow + W&B
- [x] `scripts/eval_critic.py` — precision/recall/F1, per-modality, latency p95
- [x] `scripts/benchmark_critic.py` — p95 0.69ms << 80ms SLA PASS
- [x] `notebooks/ARGUS_Train_SafetyCritic_Colab.ipynb` — 30-cell Colab notebook (TinyLlama DPO)
- [x] `docs/benchmarks/safety_critic_latency.md` — full benchmark report with dataset stats
- [x] `src/latent_sentinel/sentinel.py` — `critic_endpoint` integration, async blend weight
- [x] `tests/unit/safety_critic/test_critic.py` — 42/42 passing
- [x] GPU eval COMPLETE: precision=0.529, recall=0.964, F1=0.684 (text F1=0.778) — results committed to README + benchmark docs

---

## Month 4: PredictiveOracle + AutonomousRemediator ✅ COMPLETE (2026-05-05)

### Goals
Build prediction layer and self-healing agents. Wire full pipeline.

### Tasks
- [x] PredictiveOracleModel (PatchTST + conformal) — `src/predictive_oracle/oracle.py`
- [x] AutonomousRemediator (LangGraph multi-agent) — `src/autonomous_remediator/remediator.py`
- [x] Generate synthetic failure dataset via fault injection — 1,500 episodes, 4 fault types
- [x] Train PatchTST oracle on 9-feature telemetry sequences — F1=1.000 @ 60s ✅
- [x] Conformal prediction calibration — q̂=0.985 (90% coverage) — `models/oracle/calibrator.pkl`
- [x] Real GPU eval (Colab T4): precision=1.0, recall=1.0, F1=1.0 @ 60s — p95 latency 4.9ms ✅
- [x] `models/oracle/patchtst.pt` + `calibrator.pkl` + `oracle_manifest.json` committed

---

## Month 5: FederatedRLHF + Full MLOps Stack ✅ COMPLETE (2026-05-03)

### Tasks
- [x] Flower federated coordinator + DP-SGD — `src/federated_rlhf/` (client, server, privacy)
- [x] Simulate 2 federated enterprise client nodes — `scripts/run_federated_round.py`
- [x] Kubeflow Pipeline: data prep → round → aggregation → registry — `src/mlops/pipelines/federated_rlhf_pipeline.py`
- [x] MLflow model registry with staged promotion — `src/mlops/registry.py` (safety gate: precision≥0.85, F1≥0.78)
- [x] CI/CD: GitHub Actions → Kubeflow → GKE — `.github/workflows/safety_eval.yml` + `deploy.yml`
- [x] FederatedRLHF eval: accuracy gap 2.50% < 5% SLA ✅, privacy ε=2.40 < 3.0 ✅
- [x] `tests/unit/federated_rlhf/test_client.py` — 43/43 passing (ArgusFederatedClient + DPSGDOpacusWrapper + ArgusFedAvgStrategy)
- [x] Grafana dashboard — `src/infra/grafana/argus_dashboard.json` (6 panel groups)

---

## Month 6: Polish, Benchmark, Publish & Showcase

### Tasks
- [ ] Write technical paper (8-page NeurIPS format)
- [ ] Record 15-min architecture walkthrough video
- [ ] Publish GitHub repo with full documentation
- [ ] Deploy single-node live demo on GCP
- [ ] Submit to ICLR 2026 Safety Workshop
- [ ] Present at Chennai AI meetup
- [x] Add quantified metrics to resume — `docs/resume_metrics.md` updated with real GPU + load test numbers ✅
- [x] 100-agent load test — `scripts/load_test_e2e.py`: p95=47ms / p99=82ms / 758 req/s / 1,000/1,000 pass ✅
