output "service_account_email" {
  description = "Single service account. GitHub variable GCP_SA_EMAIL."
  value       = google_service_account.pipelines.email
}

output "workload_identity_provider" {
  description = "WIF provider name. GitHub variable GCP_WIF_PROVIDER."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "artifact_registry_url" {
  description = "Docker registry prefix."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
