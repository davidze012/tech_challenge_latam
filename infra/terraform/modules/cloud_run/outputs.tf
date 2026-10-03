output "api_url" {
  description = "Public URL of the API."
  value       = google_cloud_run_v2_service.api.uri
}

output "api_service_name" {
  description = "Cloud Run service name of the API."
  value       = google_cloud_run_v2_service.api.name
}

output "job_names" {
  description = "Cloud Run job names by pipeline."
  value       = { for name, job in google_cloud_run_v2_job.pipeline : name => job.name }
}
