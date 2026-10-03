# Environment root
#
#   make tf-plan  TF_ENV=staging   # terraform plan  -var-file=envs/staging.tfvars
#   make tf-apply TF_ENV=staging

locals {
  service_account_email = "${var.service_account_id}@${var.project_id}.iam.gserviceaccount.com"

  labels = merge(
    {
      app        = "mle-challenge"
      env        = var.env
      managed-by = "terraform"
    },
    var.labels,
  )
}

module "data" {
  source = "./modules/data"

  project_id  = var.project_id
  region      = var.region
  env         = var.env
  labels      = local.labels
  data_dir    = "${path.root}/../../data"
  schemas_dir = "${path.root}/../schemas"
}

module "cloud_run" {
  source = "./modules/cloud_run"

  project_id            = var.project_id
  region                = var.region
  env                   = var.env
  labels                = local.labels
  service_account_email = local.service_account_email

  api_image      = var.api_image
  training_image = var.training_image
  serving_image  = var.serving_image

  input_bucket     = module.data.input_bucket
  artifacts_bucket = module.data.artifacts_bucket
  dataset_id       = module.data.dataset_id

  api_min_instances         = var.api_min_instances
  api_max_instances         = var.api_max_instances
  job_timeout_seconds       = var.job_timeout_seconds
  pipeline_timeout_seconds  = var.pipeline_timeout_seconds
  request_timeout_seconds   = var.request_timeout_seconds
  results_cache_ttl_seconds = var.results_cache_ttl_seconds
}
