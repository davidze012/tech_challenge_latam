terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.5"
    }
  }

  # Remote state in the tfstate bucket.
  # bucket/prefix are passed at init time: see `make tf-bootstrap-plan`.
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}
