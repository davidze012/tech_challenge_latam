output "input_bucket" {
  description = "Bucket holding the raw CSVs."
  value       = google_storage_bucket.input.name
}

output "artifacts_bucket" {
  description = "Bucket holding model artifacts, run metadata and latest.txt."
  value       = google_storage_bucket.artifacts.name
}

output "dataset_id" {
  description = "BigQuery dataset with raw_flights, serving_input and predictions."
  value       = google_bigquery_dataset.this.dataset_id
}

output "tables" {
  description = "Table ids by logical name."
  value       = { for name, table in google_bigquery_table.this : name => table.table_id }
}
