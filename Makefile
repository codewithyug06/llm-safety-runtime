# ARGUS Project Makefile
# =======================
# All development workflows available as `make {command}`
# Run `make help` to see all commands

.PHONY: help install install-dev check lint format typecheck test test-unit \
        test-integration test-e2e test-sentinel test-causal test-critic test-oracle \
        test-remediator benchmark benchmark-sentinel benchmark-e2e \
        eval-probes eval-critic eval-oracle eval-federated \
        train-critic train-oracle run-federated-round \
        docker-build docker-push deploy-staging deploy-prod \
        infra-plan infra-apply infra-destroy \
        pre-commit clean help

# ── Config ──────────────────────────────────────────────────────────────────
PYTHON := python3.11
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

SRC := src
TESTS := tests
CONFIGS := configs

# ── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║         ARGUS Development Commands           ║"
	@echo "╚══════════════════════════════════════════════╝"
	@echo ""
	@echo "  SETUP"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install all dev dependencies"
	@echo ""
	@echo "  QUALITY"
	@echo "  make check            Run all quality checks (lint+type)"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Auto-format with black+isort"
	@echo "  make typecheck        Run mypy type checker"
	@echo ""
	@echo "  TESTING"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-sentinel    Test MOD-01 LatentSentinel"
	@echo "  make test-causal      Test MOD-02 CausalEngine"
	@echo "  make test-critic      Test MOD-03 SafetyCritic"
	@echo "  make test-oracle      Test MOD-05 PredictiveOracle"
	@echo "  make test-remediator  Test MOD-06 Remediator"
	@echo ""
	@echo "  BENCHMARKS"
	@echo "  make benchmark        Run all benchmarks"
	@echo "  make benchmark-sentinel  Benchmark LatentSentinel latency"
	@echo "  make benchmark-e2e    Benchmark full pipeline e2e"
	@echo ""
	@echo "  EVALUATION"
	@echo "  make eval-probes      Evaluate probing classifiers"
	@echo "  make eval-critic      Evaluate OmniSafetyCritic"
	@echo "  make eval-oracle      Evaluate PredictiveOracle"
	@echo ""
	@echo "  TRAINING"
	@echo "  make train-critic     Fine-tune OmniSafetyCritic (DPO)"
	@echo "  make train-oracle     Train PredictiveOracle"
	@echo "  make run-federated-round   Run one federated RLHF round"
	@echo ""
	@echo "  DEPLOYMENT"
	@echo "  make deploy-staging   Deploy to argus-staging namespace"
	@echo "  make deploy-prod      Deploy to argus-prod namespace"
	@echo "  make infra-plan       Terraform plan"
	@echo "  make infra-apply      Terraform apply"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	pre-commit install

# ── Quality ──────────────────────────────────────────────────────────────────
check: lint typecheck

lint:
	$(RUFF) check $(SRC) $(TESTS)
	$(BLACK) --check $(SRC) $(TESTS)
	$(ISORT) --check-only $(SRC) $(TESTS)

format:
	$(BLACK) $(SRC) $(TESTS)
	$(ISORT) $(SRC) $(TESTS)
	$(RUFF) check --fix $(SRC) $(TESTS)

typecheck:
	$(MYPY) $(SRC) --ignore-missing-imports --strict

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	$(PYTEST) $(TESTS) -v --tb=short --cov=$(SRC) --cov-report=term-missing --cov-fail-under=80

test-unit:
	$(PYTEST) $(TESTS)/unit -v --tb=short

test-integration:
	$(PYTEST) $(TESTS)/integration -v --tb=short -m "integration"

test-e2e:
	$(PYTEST) $(TESTS)/e2e -v --tb=short -m "e2e"

test-sentinel:
	$(PYTEST) $(TESTS)/unit/latent_sentinel -v --tb=short -k "sentinel"

test-causal:
	$(PYTEST) $(TESTS)/unit/causal_engine -v --tb=short

test-critic:
	$(PYTEST) $(TESTS)/unit/safety_critic -v --tb=short

test-oracle:
	$(PYTEST) $(TESTS)/unit/predictive_oracle -v --tb=short

test-remediator:
	$(PYTEST) $(TESTS)/unit/autonomous_remediator -v --tb=short

# ── Benchmarks ───────────────────────────────────────────────────────────────
benchmark:
	$(PYTHON) scripts/benchmark_latency.py --all
	$(PYTHON) scripts/benchmark_throughput.py --all
	$(PYTHON) scripts/generate_benchmark_report.py

benchmark-sentinel:
	$(PYTHON) scripts/benchmark_latency.py --module latent_sentinel --n-requests 1000

benchmark-e2e:
	$(PYTHON) scripts/benchmark_e2e.py --agents 100 --duration 60

# ── Evaluation ───────────────────────────────────────────────────────────────
eval-probes:
	$(PYTHON) scripts/eval_probes.py \
		--datasets halueval,truthfulqa,safetybench \
		--output docs/benchmarks/probes_eval_$$(date +%Y%m%d).md

eval-critic:
	$(PYTHON) scripts/eval_critic.py \
		--test-set data/eval/multimodal_safety_test.jsonl \
		--output docs/benchmarks/critic_eval_$$(date +%Y%m%d).md

eval-oracle:
	$(PYTHON) scripts/eval_oracle.py \
		--test-set data/eval/failure_test_episodes.npz \
		--horizons 30,60,90 \
		--output docs/benchmarks/oracle_eval_$$(date +%Y%m%d).md

eval-federated:
	$(PYTHON) scripts/eval_federated.py \
		--rounds 10 \
		--clients 2 \
		--output docs/benchmarks/federated_eval_$$(date +%Y%m%d).md

# ── Training ─────────────────────────────────────────────────────────────────
train-critic:
	$(PYTHON) -m src.safety_critic.train \
		--config configs/safety_critic.yaml \
		--output models/critic_$$(date +%Y%m%d_%H%M)

train-oracle:
	$(PYTHON) -m src.predictive_oracle.train \
		--config configs/predictive_oracle.yaml \
		--output models/oracle_$$(date +%Y%m%d_%H%M)

run-federated-round:
	$(PYTHON) -m src.federated_rlhf.coordinator \
		--config configs/federated_rlhf.yaml \
		--round-id $$(date +%Y%m%d_%H%M)

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t argus-latent-sentinel:latest -f src/latent_sentinel/Dockerfile .
	docker build -t argus-remediator:latest -f src/autonomous_remediator/Dockerfile .
	docker build -t argus-oracle:latest -f src/predictive_oracle/Dockerfile .
	docker build -t argus-api:latest -f src/api/Dockerfile .

docker-push:
	docker push gcr.io/$$GCP_PROJECT_ID/argus-latent-sentinel:latest
	docker push gcr.io/$$GCP_PROJECT_ID/argus-remediator:latest
	docker push gcr.io/$$GCP_PROJECT_ID/argus-oracle:latest
	docker push gcr.io/$$GCP_PROJECT_ID/argus-api:latest

# ── Deployment ───────────────────────────────────────────────────────────────
deploy-staging:
	kubectl apply -f src/infra/k8s/ -n argus-staging --prune --all
	kubectl rollout status deployment -n argus-staging

deploy-prod:
	@echo "⚠️  Deploying to PRODUCTION. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	kubectl apply -f src/infra/k8s/ -n argus-prod
	kubectl rollout status deployment -n argus-prod

# ── Infra ─────────────────────────────────────────────────────────────────────
infra-plan:
	cd src/infra/terraform && terraform plan -var-file=prod.tfvars

infra-apply:
	cd src/infra/terraform && terraform apply -var-file=prod.tfvars -auto-approve

infra-destroy:
	@echo "⚠️  DESTROYING INFRASTRUCTURE. Type 'yes' to confirm:" && read ans && [ $$ans = yes ]
	cd src/infra/terraform && terraform destroy -var-file=prod.tfvars

# ── Pre-commit ────────────────────────────────────────────────────────────────
pre-commit:
	$(MAKE) format
	$(MAKE) check
	$(MAKE) test-unit

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null; true
	rm -rf htmlcov .coverage coverage.xml
