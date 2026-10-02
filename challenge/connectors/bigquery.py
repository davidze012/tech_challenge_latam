"""Data-reader implementation for BigQuery"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from challenge.config import get_settings
from challenge.connectors.protocols import PREDICTION_COLUMN, build_predictions_frame

logger = logging.getLogger(__name__)

#: Schema of the predictions table (kept in sync with infra/schemas/predictions.json).
PREDICTIONS_SCHEMA = [
    bigquery.SchemaField("OPERA", "STRING"),
    bigquery.SchemaField("TIPOVUELO", "STRING"),
    bigquery.SchemaField("MES", "INT64"),
    bigquery.SchemaField(PREDICTION_COLUMN, "INT64"),
]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _checked(identifier: str, kind: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(identifier or ""):
        raise ValueError(f"Invalid BigQuery {kind}: {identifier!r}")
    return identifier


class BigQueryClient:
    """
    Parameters
    ----------
    _client:
        Optional pre-built BQ client for testing. When ``None`` (default) a
        client is created using Application Default Credentials bound to
        ``Settings.gcp_project_id``.
    """

    _SQL_PATH: Path = Path(__file__).resolve().parents[2] / "assets" / "load_data.sql"

    def __init__(self, _client: bigquery.Client | None = None) -> None:
        """Initialise with optional injected BQ client."""
        self._client = _client

    @property
    def client(self) -> bigquery.Client:
        """Return the shared BigQuery client."""
        if self._client is None:
            settings = get_settings()
            self._client = bigquery.Client(
                project=settings.gcp_project_id or None, location=settings.bq_location
            )
        return self._client

    @staticmethod
    def table_id(table: str) -> str:
        """Return the fully-qualified ``project.dataset.table`` id for ``table``."""
        settings = get_settings()
        return ".".join(
            (
                _checked(settings.gcp_project_id, "project"),
                _checked(settings.bq_dataset, "dataset"),
                _checked(table, "table"),
            )
        )

    def read(self) -> pd.DataFrame:
        """Execute ``assets/load_data.sql`` against BigQuery and return results."""
        template = self._SQL_PATH.read_text(encoding="utf-8")
        settings = get_settings()
        sql = template.format(
            project=_checked(settings.gcp_project_id, "project"),
            dataset=_checked(settings.bq_dataset, "dataset"),
            raw_table=_checked(settings.bq_raw_table, "table"),
        )
        data = self.client.query(sql).to_dataframe(create_bqstorage_client=False)
        logger.info("Read %d rows from BigQuery table %s", len(data), settings.bq_raw_table)
        return data

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Insert predictions into the configured BigQuery predictions table."""
        frame = build_predictions_frame(predictions, identifiers)
        table_id = self.table_id(get_settings().bq_predictions_table)
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=PREDICTIONS_SCHEMA,
        )
        # A load job is atomic: readers see either the previous table or the new one.
        job = self.client.load_table_from_dataframe(frame, table_id, job_config=job_config)
        job.result()
        logger.info("Wrote %d predictions to %s", len(frame), table_id)
