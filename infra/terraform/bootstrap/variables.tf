variable "project_id" {
  description = "GCP project that hosts the challenge."
  type        = string
}

variable "region" {
  description = "Region for regional resources (Artifact Registry)."
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  description = "GitHub repository (owner/name) allowed to impersonate the service account via WIF."
  type        = string
  default     = "davidze012/tech_challenge_latam"
}

variable "service_account_id" {
  description = "Account id of the single service account used by the API, the jobs and CI/CD."
  type        = string
  default     = "mle-challenge-sa"
}

variable "artifact_repository_id" {
  description = "Artifact Registry Docker repository for the three images."
  type        = string
  default     = "mle-challenge"
}

variable "operator_members" {
  description = <<-EOT
    Persons allowed to impersonate the service account (e.g. ["user:person@email.com"]), to run
    Terraform with the exact permissions of CI/CD. Pass it at runtime (TF_VAR_operator_members).
  EOT
  type        = list(string)
  default     = []
}
