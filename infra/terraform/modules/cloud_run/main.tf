# Compute layer: training and serving Cloud Run Jobs and service.
#
# Timeouts are layered, the API can answer before Cloud Run cuts the request:
#   job task timeout (job_timeout_seconds) < API wait (pipeline_timeout_seconds) < request timeout.

locals {
  common_env = {
    DEPLOYMENT_MODE      = "gcp"
    GCP_PROJECT_ID       = var.project_id
    GCP_REGION           = var.region
    GCP_BUCKET_ARTIFACTS = var.artifacts_bucket
    GCP_BUCKET_INPUT     = var.input_bucket
    BQ_DATASET           = var.dataset_id
    BQ_LOCATION          = var.region
    LOG_FORMAT           = "json"
  }

  jobs = {
    training = { image = var.training_image, memory = "2Gi" }
    serving  = { image = var.serving_image, memory = "1Gi" }
  }
  job_names = { for name, _ in local.jobs : name => "mle-${name}-${var.env}" }

  api_env = merge(local.common_env, {
    TRAINING_JOB_NAME   = local.job_names.training
    SERVING_JOB_NAME    = local.job_names.serving
    PIPELINE_TIMEOUT_S  = tostring(var.pipeline_timeout_seconds)
    RESULTS_CACHE_TTL_S = tostring(var.results_cache_ttl_seconds)
  })
}

# --- Pipelines (Cloud Run Jobs) -------------------------------------------------------------------

resource "google_cloud_run_v2_job" "pipeline" {
  for_each = local.jobs

  name                = local.job_names[each.key]
  location            = var.region
  deletion_protection = false
  labels              = var.labels

  template {
    task_count  = 1
    parallelism = 1
    labels      = var.labels

    template {
      service_account = var.service_account_email
      timeout         = "${var.job_timeout_seconds}s"
      max_retries     = 0 # a failed run surfaces as an API error instead of silently retrying

      containers {
        image = each.value.image

        resources {
          limits = {
            cpu    = "1"
            memory = each.value.memory
          }
        }

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.key
            value = env.value
          }
        }
      }
    }
  }
}

# --- API (Cloud Run service) ----------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  name                = "mle-api-${var.env}"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false
  labels              = var.labels

  template {
    service_account                  = var.service_account_email
    timeout                          = "${var.request_timeout_seconds}s"
    max_instance_request_concurrency = 80
    labels                           = var.labels

    scaling {
      min_instance_count = var.api_min_instances # >= 1 in prod: no cold starts for evaluators
      max_instance_count = var.api_max_instances # caps cost of the public endpoints
    }

    containers {
      image = var.api_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle          = true # request-based billing: no CPU charge between requests
        startup_cpu_boost = true # faster cold starts
      }

      dynamic "env" {
        for_each = local.api_env
        content {
          name  = env.key
          value = env.value
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        period_seconds    = 2
        timeout_seconds   = 2
        failure_threshold = 30
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds  = 30
        timeout_seconds = 5
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  name     = google_cloud_run_v2_service.api.name
  location = google_cloud_run_v2_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
