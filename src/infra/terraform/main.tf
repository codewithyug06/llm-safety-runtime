# ARGUS Infrastructure — Terraform Main Configuration
# =====================================================
# Provisions GKE cluster, Kafka (via Confluent), Redis,
# Cloud Spanner, and supporting GCP resources.
#
# Apply with:
#   terraform init
#   terraform plan -var-file=environments/prod.tfvars
#   terraform apply -var-file=environments/prod.tfvars

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.10"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.10"
    }
    confluent = {
      source  = "confluentinc/confluent"
      version = "~> 1.62"
    }
  }

  backend "gcs" {
    bucket = "argus-tf-state"
    prefix = "terraform/state"
  }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Primary GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Deployment environment: dev | staging | production"
  type        = string
  default     = "staging"
}

variable "gke_node_count" {
  description = "Number of nodes in the default node pool"
  type        = number
  default     = 3
}

variable "gke_machine_type" {
  description = "Machine type for CPU nodes"
  type        = string
  default     = "n1-standard-4"
}

variable "gke_gpu_machine_type" {
  description = "Machine type for GPU nodes"
  type        = string
  default     = "n1-standard-8"
}

variable "confluent_api_key" {
  description = "Confluent Cloud API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "confluent_api_secret" {
  description = "Confluent Cloud API secret"
  type        = string
  sensitive   = true
  default     = ""
}

# ── Providers ─────────────────────────────────────────────────────────────────

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

provider "confluent" {
  cloud_api_key    = var.confluent_api_key
  cloud_api_secret = var.confluent_api_secret
}

# ── GKE Cluster ───────────────────────────────────────────────────────────────

resource "google_container_cluster" "argus" {
  provider = google-beta
  name     = "argus-${var.environment}"
  location = var.region

  # Regional cluster for high availability
  node_locations = ["${var.region}-a", "${var.region}-b", "${var.region}-c"]

  # Remove default node pool; we manage node pools separately
  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  network_policy {
    enabled = true
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
}

# CPU node pool for API + orchestration workloads
resource "google_container_node_pool" "cpu_nodes" {
  name       = "cpu-pool"
  cluster    = google_container_cluster.argus.name
  location   = var.region
  node_count = var.gke_node_count

  node_config {
    machine_type = var.gke_machine_type
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      environment = var.environment
      pool-type   = "cpu"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }
}

# GPU node pool for ML inference (T4 GPUs)
resource "google_container_node_pool" "gpu_nodes" {
  name     = "gpu-pool"
  cluster  = google_container_cluster.argus.name
  location = var.region

  initial_node_count = 1

  node_config {
    machine_type = var.gke_gpu_machine_type
    disk_size_gb = 200
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      environment = var.environment
      pool-type   = "gpu"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  autoscaling {
    min_node_count = 0
    max_node_count = 4
  }
}

# ── Redis (Memorystore) ───────────────────────────────────────────────────────

resource "google_redis_instance" "argus" {
  name           = "argus-redis-${var.environment}"
  memory_size_gb = 4
  region         = var.region
  redis_version  = "REDIS_7_0"
  tier           = "STANDARD_HA"

  auth_enabled            = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"

  labels = {
    environment = var.environment
    component   = "state-store"
  }
}

# ── Cloud Spanner ─────────────────────────────────────────────────────────────

resource "google_spanner_instance" "argus" {
  name         = "argus-${var.environment}"
  config       = "regional-${var.region}"
  display_name = "ARGUS Audit Log"

  num_nodes = var.environment == "production" ? 3 : 1

  labels = {
    environment = var.environment
    component   = "audit-log"
  }
}

resource "google_spanner_database" "argus" {
  instance = google_spanner_instance.argus.name
  name     = "argus-db"

  ddl = [
    <<-DDL
      CREATE TABLE RemediationAuditLog (
        RecordId     STRING(36) NOT NULL,
        Timestamp    TIMESTAMP NOT NULL,
        AgentId      STRING(256) NOT NULL,
        SafetyScore  FLOAT64 NOT NULL,
        ActionTaken  STRING(64) NOT NULL,
        ActionDetail STRING(1024),
        TriggeredBy  STRING(256),
        Outcome      STRING(32),
        LatencyMs    FLOAT64,
      ) PRIMARY KEY (RecordId)
    DDL
    ,
    "CREATE INDEX AuditByAgent ON RemediationAuditLog(AgentId, Timestamp DESC)",
  ]

  deletion_protection = var.environment == "production" ? true : false
}

# ── Kafka (Confluent Cloud) ───────────────────────────────────────────────────

resource "confluent_environment" "argus" {
  display_name = "argus-${var.environment}"
}

resource "confluent_kafka_cluster" "argus" {
  display_name = "argus-kafka"
  availability = var.environment == "production" ? "MULTI_ZONE" : "SINGLE_ZONE"
  cloud        = "GCP"
  region       = var.region

  dedicated {
    cku = var.environment == "production" ? 2 : 1
  }

  environment {
    id = confluent_environment.argus.id
  }
}

# ── Kafka Topics ──────────────────────────────────────────────────────────────

locals {
  kafka_topics = [
    "argus.safety.signals",
    "argus.telemetry",
    "argus.remediation",
    "argus.risk.predictions",
  ]
}

# ── GCS Buckets ───────────────────────────────────────────────────────────────

resource "google_storage_bucket" "argus_data" {
  name          = "argus-data-${var.project_id}-${var.environment}"
  location      = var.region
  storage_class = "STANDARD"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

resource "google_storage_bucket" "argus_models" {
  name          = "argus-models-${var.project_id}-${var.environment}"
  location      = var.region
  storage_class = "STANDARD"

  versioning {
    enabled = true
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "gke_cluster_name" {
  value = google_container_cluster.argus.name
}

output "gke_cluster_endpoint" {
  value     = google_container_cluster.argus.endpoint
  sensitive = true
}

output "redis_host" {
  value = google_redis_instance.argus.host
}

output "redis_port" {
  value = google_redis_instance.argus.port
}

output "spanner_instance" {
  value = google_spanner_instance.argus.name
}

output "spanner_database" {
  value = google_spanner_database.argus.name
}

output "data_bucket" {
  value = google_storage_bucket.argus_data.name
}

output "models_bucket" {
  value = google_storage_bucket.argus_models.name
}
