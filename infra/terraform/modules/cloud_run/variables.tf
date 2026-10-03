variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = "Cloud Run region."
  type        = string
}

variable "env" {
  description = "Environment name (staging or prod); suffixes every resource name."
  type        = string
}

variable "labels" {
  description = "Labels applied to every resource."
  type        = map(string)
  default     = {}
}

variable "service_account_email" {
  description = "Runtime identity of the API and both jobs."
  type        = string
}

variable "api_image" {
  description = "Image of the API service."
  type        = string
}

variable "training_image" {
  description = "Image of the training job."
  type        = string
}

variable "serving_image" {
  description = "Image of the serving job."
  type        = string
}

variable "input_bucket" {
  description = "Bucket with the raw CSVs."
  type        = string
}

variable "artifacts_bucket" {
  description = "Bucket with model artifacts and metadata."
  type        = string
}

variable "dataset_id" {
  description = "BigQuery dataset id."
  type        = string
}

variable "api_min_instances" {
  description = "Minimum API instances (1 keeps it warm)."
  type        = number
}

variable "api_max_instances" {
  description = "Maximum API instances."
  type        = number
}

variable "job_timeout_seconds" {
  description = "Task timeout of the training/serving jobs."
  type        = number
}

variable "pipeline_timeout_seconds" {
  description = "How long the API waits for a job execution before answering 504."
  type        = number
}

variable "request_timeout_seconds" {
  description = "Cloud Run request timeout of the API (max 3600)."
  type        = number
}

variable "results_cache_ttl_seconds" {
  description = "TTL of the API's in-memory predictions cache."
  type        = number
}
