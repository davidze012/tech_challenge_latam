output "api_url" {
  description = "Public URL of the ML pipeline control API."
  value       = module.cloud_run.api_url
}

output "job_names" {
  description = "Cloud Run job names by pipeline."
  value       = module.cloud_run.job_names
}

output "input_bucket" {
  description = "Bucket with the raw CSVs."
  value       = module.data.input_bucket
}

output "artifacts_bucket" {
  description = "Bucket with model artifacts, metadata and latest.txt."
  value       = module.data.artifacts_bucket
}

output "dataset_id" {
  description = "BigQuery dataset."
  value       = module.data.dataset_id
}
