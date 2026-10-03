locals {
  services = toset([
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "bigquerystorage.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
    "serviceusage.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com", # token exchange for WIF
  ])

  # runtime identity of the API and both jobs
  # Least privilege rule applyed

  service_account_roles = toset([
    "roles/artifactregistry.writer", # push images (CD / Cloud Build)
    "roles/bigquery.dataOwner",      # datasets, tables, predictions writes
    "roles/bigquery.jobUser",        # load and query jobs
    "roles/bigquery.resourceViewer", # jobs created by other identities (terraform refresh)
    "roles/logging.logWriter",       # application and build logs
    "roles/run.admin",               # services/jobs, public invoker, runWithOverrides
    "roles/storage.admin",           # buckets, artifacts, terraform state
  ])
}

resource "google_project_service" "apis" {
  for_each = local.services

  service            = each.value
  disable_on_destroy = false
}

# --- Service account -------------------------------------------------------------------------

resource "google_service_account" "pipelines" {
  account_id   = var.service_account_id
  display_name = "MLE challenge (API, pipelines, CI/CD)"
  description  = "Identity for the API service, the training/serving jobs and GitHub Actions."

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "service_account" {
  for_each = local.service_account_roles

  project = var.project_id
  role    = each.value
  member  = google_service_account.pipelines.member
}

# Deploying Cloud Run resources.
resource "google_service_account_iam_member" "act_as_itself" {
  service_account_id = google_service_account.pipelines.name
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.pipelines.member
}

resource "google_service_account_iam_member" "operators" {
  for_each = toset(var.operator_members)

  service_account_id = google_service_account.pipelines.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = each.value
}

# --- Workload Identity Federation -------------------------------

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  description               = "Identities issued by GitHub Actions OIDC."

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions OIDC"

  attribute_mapping = {
    "google.subject"             = "assertion.sub"
    "attribute.repository"       = "assertion.repository"
    "attribute.repository_owner" = "assertion.repository_owner"
    "attribute.ref"              = "assertion.ref"
  }
  attribute_condition = "assertion.repository == '${var.github_repository}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_impersonation" {
  service_account_id = google_service_account.pipelines.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# --- Artifact Registry ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.artifact_repository_id
  format        = "DOCKER"
  description   = "Images of the API, training and serving components."

  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-recent-versions"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }
  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  depends_on = [google_project_service.apis]
}
