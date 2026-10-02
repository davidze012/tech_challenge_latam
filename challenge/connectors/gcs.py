"""Artifact-store protocol and filesystem."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import joblib
import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import storage

from challenge.config import get_settings
from challenge.connectors import bigquery as bigquery_connector
from challenge.connectors.protocols import (
    LATEST_POINTER,
    METADATA_FILENAME,
    MODEL_FILENAME,
    validate_run_id,
)

logger = logging.getLogger(__name__)


def _storage_client() -> storage.Client:
    # Resolved at call time (never at import) so no credentials are needed to import/construct.
    return storage.Client(project=get_settings().gcp_project_id or None)


class GCSArtifactStore:
    """Save and load model artifacts on Google Cloud Storage.

    Layout::

        gs://<bucket>/
            <run_id>/model.joblib
            <run_id>/metadata.json
            latest.txt
    """

    def _bucket_name(self) -> str:
        name = get_settings().gcp_bucket_artifacts
        if not name:
            raise ValueError("GCP_BUCKET_ARTIFACTS is not configured")
        return name

    def _bucket(self) -> storage.Bucket:
        return _storage_client().bucket(self._bucket_name())

    def save(self, model: Any, run_id: str) -> str:
        """Serialize and upload model to GCS; update latest.txt."""
        validate_run_id(run_id)
        bucket = self._bucket()
        blob_name = f"{run_id}/{MODEL_FILENAME}"

        buffer = io.BytesIO()
        joblib.dump(model._model, buffer)
        bucket.blob(blob_name).upload_from_string(
            buffer.getvalue(),
            content_type="application/octet-stream",
            if_generation_match=0,
        )
        bucket.blob(LATEST_POINTER).upload_from_string(run_id)

        uri = f"gs://{self._bucket_name()}/{blob_name}"
        logger.info("Saved model artifact %s", uri)
        return uri

    def load(self, model: Any, run_id: str | None) -> None:
        """Download and deserialize model from GCS into model._model."""
        bucket = self._bucket()
        if run_id is None:
            run_id = self._read_latest(bucket)
            if run_id is None:
                raise FileNotFoundError(
                    f"No trained model available: gs://{self._bucket_name()}/{LATEST_POINTER} "
                    "not found"
                )
        blob_name = f"{validate_run_id(run_id)}/{MODEL_FILENAME}"
        try:
            payload = bucket.blob(blob_name).download_as_bytes()
        except NotFound as exc:
            raise FileNotFoundError(
                f"Model artifact not found: gs://{self._bucket_name()}/{blob_name}"
            ) from exc
        model._model = joblib.load(io.BytesIO(payload))
        logger.info("Loaded model artifact gs://%s/%s", self._bucket_name(), blob_name)

    @staticmethod
    def _read_latest(bucket: storage.Bucket) -> str | None:
        blob = bucket.blob(LATEST_POINTER)
        if not blob.exists():
            return None
        try:
            return blob.download_as_text().strip() or None
        except NotFound:  # deleted between exists() and download
            return None

    def latest_run_id(self) -> str | None:
        """Return the run_id in latest.txt, or ``None`` when no model was saved yet."""
        return self._read_latest(self._bucket())

    def save_metadata(self, run_id: str, metadata: dict[str, Any]) -> str:
        """Upload ``<run_id>/metadata.json`` and return its URI."""
        blob_name = f"{validate_run_id(run_id)}/{METADATA_FILENAME}"
        self._bucket().blob(blob_name).upload_from_string(
            json.dumps(metadata, indent=2, default=str),
            content_type="application/json",
            if_generation_match=0,
        )
        return f"gs://{self._bucket_name()}/{blob_name}"

    def load_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Return the metadata of ``run_id``, or ``None`` if it does not exist."""
        blob_name = f"{validate_run_id(run_id)}/{METADATA_FILENAME}"
        try:
            return json.loads(self._bucket().blob(blob_name).download_as_text())
        except NotFound:
            return None


class GCSCSVClient:
    """Read flight data as a CSV object from the GCS input bucket."""

    def __init__(self, blob_name: str, _bq_writer: Any | None = None) -> None:
        """Initialise with the input-bucket object name and an optional injected writer."""
        self._blob_name = blob_name
        self._bq_writer = _bq_writer

    def read(self) -> pd.DataFrame:
        """Download and parse the CSV object from the configured input bucket."""
        bucket_name = get_settings().gcp_bucket_input
        if not bucket_name:
            raise ValueError("GCP_BUCKET_INPUT is not configured")
        blob = _storage_client().bucket(bucket_name).blob(self._blob_name)
        try:
            payload = blob.download_as_bytes()
        except NotFound as exc:
            raise FileNotFoundError(f"gs://{bucket_name}/{self._blob_name} not found") from exc
        data = pd.read_csv(io.BytesIO(payload), low_memory=False)
        logger.info("Read %d rows from gs://%s/%s", len(data), bucket_name, self._blob_name)
        return data

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to the configured BigQuery predictions table."""
        if self._bq_writer is None:
            self._bq_writer = bigquery_connector.BigQueryClient()
        self._bq_writer.write(predictions, identifiers)
