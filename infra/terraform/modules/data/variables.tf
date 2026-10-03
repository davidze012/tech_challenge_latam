variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = "Location of the buckets and the BigQuery dataset."
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

variable "data_dir" {
  description = "Local folder holding data.csv and serving_input.csv."
  type        = string
}

variable "schemas_dir" {
  description = "Local folder holding the BigQuery JSON schemas (shared with infra/bootstrap.sh)."
  type        = string
}
