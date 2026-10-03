# Data layer: raw CSVs in Cloud Storage, loaded into BigQuery tables, plus
# the artifacts bucket used by the training/serving jobs.

locals {
  is_prod = var.env == "prod"

  # BigQuery table <- CSV object in the input bucket.
  csv_tables = {
    raw_flights   = "data.csv"
    serving_input = "serving_input.csv"
  }
  tables = toset(concat(keys(local.csv_tables), ["predictions"]))
}

# --- Cloud Storage -------------------------------------------------------------------------------

resource "google_storage_bucket" "input" {
  name                        = "${var.project_id}-mle-input-${var.env}"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = !local.is_prod
  labels                      = var.labels

  # Protects the raw CSVs against accidental overwrites.
  versioning {
    enabled = true
  }
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 30
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-mle-artifacts-${var.env}"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = !local.is_prod
  labels                      = var.labels

  # Overwritten objects (latest.txt) stay recoverable for a while.
  versioning {
    enabled = true
  }
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 30
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket_object" "csv" {
  for_each = local.csv_tables

  bucket       = google_storage_bucket.input.name
  name         = each.value
  source       = "${var.data_dir}/${each.value}"
  content_type = "text/csv"
}

# --- BigQuery ------------------------------------------------------------------------------------

resource "google_bigquery_dataset" "this" {
  dataset_id                 = "mle_challenge_${var.env}"
  location                   = var.region
  description                = "Flight delay challenge (${var.env}): raw flights, serving input and predictions."
  delete_contents_on_destroy = !local.is_prod
  labels                     = var.labels
}

resource "google_bigquery_table" "this" {
  for_each = local.tables

  dataset_id          = google_bigquery_dataset.this.dataset_id
  table_id            = each.key
  schema              = file("${var.schemas_dir}/${each.key}.json")
  deletion_protection = local.is_prod
  labels              = var.labels
}

# One load job per CSV: the job id embeds the file hash, so changing a CSV triggers a
# new WRITE_TRUNCATE load.
resource "google_bigquery_job" "load" {
  for_each = local.csv_tables

  job_id   = "load_${each.key}_${var.env}_${substr(filemd5("${var.data_dir}/${each.value}"), 0, 12)}"
  location = var.region
  labels   = var.labels

  load {
    source_uris = ["gs://${google_storage_bucket.input.name}/${google_storage_bucket_object.csv[each.key].name}"]

    destination_table {
      project_id = var.project_id
      dataset_id = google_bigquery_dataset.this.dataset_id
      table_id   = google_bigquery_table.this[each.key].table_id
    }

    source_format      = "CSV"
    skip_leading_rows  = 1     # header; columns map by position to the explicit schema
    autodetect         = false # explicit schema: Vlo-* are alphanumeric, Fecha-* DATETIME
    encoding           = "UTF-8"
    write_disposition  = "WRITE_TRUNCATE"
    create_disposition = "CREATE_NEVER"
  }
}
