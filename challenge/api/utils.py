"""Helper functions for the pipeline control API.

Every helper works in both deployment modes:

* ``gcp``   — Cloud Run Jobs (training/serving), BigQuery (predictions), GCS (artifacts).
* ``local`` — pipelines run in-process, predictions/artifacts live on the filesystem.
"""

from __future__ import annotations

import importlib
import itertools
import logging
import math
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, run_v2

from challenge.config import DELAY_THRESHOLD_MINUTES, get_settings
from challenge.connectors import gcs
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import PREDICTIONS_FILENAME, LocalArtifactStore
from challenge.connectors.protocols import IDENTIFIER_COLUMNS, PREDICTION_COLUMN, combination_key

logger = logging.getLogger(__name__)

PIPELINES = ("train", "predict")
#: Local-mode raw training file (same as ``challenge.training.TRAINING_FILENAME``).
_RAW_TRAINING_FILENAME = "data.csv"

#: order for pagination: rows that tie on all four columns are identical.
_ORDER_COLUMNS = ["OPERA", "TIPOVUELO", "MES", PREDICTION_COLUMN]
_MAX_BYTES_BILLED = 100 * 1024 * 1024

_PREDICTIONS_SQL = """
WITH filtered AS (
  SELECT OPERA, TIPOVUELO, MES, prediction
  FROM `{table_id}`
  -- BigQuery receives empty array parameters as NULL: IFNULL keeps "empty = no filter".
  -- @opera arrives upper-cased: the airline filter is case-insensitive.
  WHERE (IFNULL(ARRAY_LENGTH(@opera), 0) = 0 OR UPPER(OPERA) IN UNNEST(@opera))
    AND (IFNULL(ARRAY_LENGTH(@tipovuelo), 0) = 0 OR TIPOVUELO IN UNNEST(@tipovuelo))
    AND (IFNULL(ARRAY_LENGTH(@mes), 0) = 0 OR MES IN UNNEST(@mes))
),
-- Historical context: how often each (airline, flight type, month) was delayed in raw_flights.
history AS (
  SELECT
    OPERA, TIPOVUELO, MES,
    COUNT(*) AS historical_flights,
    ROUND(AVG(IF(DATETIME_DIFF(`Fecha-O`, `Fecha-I`, MINUTE) > @delay_threshold, 1, 0)), 4)
      AS historical_delay_rate
  FROM `{raw_table_id}`
  GROUP BY OPERA, TIPOVUELO, MES
)
SELECT
  (SELECT COUNT(*) FROM filtered) AS total,
  (SELECT COUNTIF(prediction = 1) FROM filtered) AS delayed,
  ARRAY(
    SELECT AS STRUCT
      f.OPERA, f.TIPOVUELO, f.MES, f.prediction AS predicted_delay,
      h.historical_flights, h.historical_delay_rate
    FROM filtered AS f
    LEFT JOIN history AS h
      ON f.OPERA = h.OPERA AND f.TIPOVUELO = h.TIPOVUELO AND f.MES = h.MES
    ORDER BY f.OPERA, f.TIPOVUELO, f.MES, f.prediction
    LIMIT @limit OFFSET @offset
  ) AS page_rows  -- not "rows": ROWS is a reserved keyword in GoogleSQL
"""


class PipelineFailedError(RuntimeError):
    """A pipeline execution finished without succeeding."""


@dataclass(frozen=True)
class PipelineExecution:
    """Outcome of a pipeline run: ``name`` identifies the execution for tracing."""

    name: str
    run_id: str | None = None


# --- Caching -----------------------------------------------------------------------------------


class TTLCache:
    """In-memory TTL cache with single-flight computation.

    Concurrent misses on the same key wait for one computation instead of stampeding the
    backend.
    """

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int = 1024,
        clock: Callable[[], float] = time.monotonic,
        stripes: int = 64,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: dict[Hashable, tuple[float, Any]] = {}
        self._guard = threading.Lock()
        # Striped locks bound memory regardless of how many distinct keys clients send.
        self._stripes = [threading.Lock() for _ in range(stripes)]
        self._generation = 0

    def _fresh(self, key: Hashable) -> tuple[bool, Any]:
        entry = self._entries.get(key)
        if entry is not None and entry[0] > self._clock():
            return True, entry[1]
        return False, None

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, computing it once when missing/expired."""
        with self._guard:
            hit, value = self._fresh(key)
        if hit:
            return value

        with self._stripes[hash(key) % len(self._stripes)]:
            with self._guard:  # another thread may have filled it while we waited
                hit, value = self._fresh(key)
                generation = self._generation
            if hit:
                return value
            value = compute()
            with self._guard:
                if generation == self._generation:  # skip if invalidated meanwhile
                    self._store(key, value)
            return value

    def _store(self, key: Hashable, value: Any) -> None:
        now = self._clock()
        if len(self._entries) >= self._max_entries:
            for stale in [k for k, (expiry, _) in self._entries.items() if expiry <= now]:
                del self._entries[stale]
        if len(self._entries) >= self._max_entries:
            del self._entries[min(self._entries, key=lambda k: self._entries[k][0])]
        self._entries[key] = (now + self._ttl, value)

    def invalidate(self) -> None:
        """Drop every entry."""
        with self._guard:
            self._entries.clear()
            self._generation += 1


@cache
def _results_cache() -> TTLCache:
    return TTLCache(ttl_seconds=get_settings().results_cache_ttl_s)


@cache
def _summary_cache() -> TTLCache:
    return TTLCache(ttl_seconds=get_settings().results_cache_ttl_s)


def invalidate_predictions_cache() -> None:
    """Forget cached pages and the serving summary so reads see the latest serving output."""
    _results_cache().invalidate()
    _summary_cache().invalidate()


# --- Clients (GCP mode) ------------------------------------------------------------------------


@cache
def _bigquery_client() -> bigquery.Client:
    settings = get_settings()
    return bigquery.Client(project=settings.gcp_project_id, location=settings.bq_location)


@cache
def _jobs_client() -> run_v2.JobsClient:
    return run_v2.JobsClient()


@cache
def _executions_client() -> run_v2.ExecutionsClient:
    return run_v2.ExecutionsClient()


def _artifact_store() -> gcs.GCSArtifactStore | LocalArtifactStore:
    if get_settings().deployment_mode == "gcp":
        return gcs.GCSArtifactStore()
    return LocalArtifactStore()


def warm_up() -> None:
    """Create the GCP clients eagerly (best effort) so the first request is not slower."""
    if get_settings().deployment_mode != "gcp":
        return
    try:
        _bigquery_client()
        _jobs_client()
        _executions_client()
    except Exception:
        logger.warning("Client warm-up failed", exc_info=True)


# --- Pre-flight checks -------------------------------------------------------------------------


def _check_raw_flights() -> int:
    """Return how many raw flights are available for training (0 when there is no data)."""
    settings = get_settings()
    if settings.deployment_mode == "gcp":
        try:
            table = _bigquery_client().get_table(BigQueryClient.table_id(settings.bq_raw_table))
        except NotFound:
            return 0
        return int(table.num_rows or 0)

    path = Path(settings.data_dir) / _RAW_TRAINING_FILENAME
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _check_model_exists() -> bool:
    """Whether a trained model has been promoted (``latest.txt`` exists)."""
    return _artifact_store().latest_run_id() is not None


def _latest_model_version() -> str | None:
    try:
        return _artifact_store().latest_run_id()
    except Exception:
        logger.warning("Could not resolve the latest model version", exc_info=True)
        return None


def _read_model_metadata(run_id: str | None = None) -> dict[str, Any]:
    """Return the metadata of ``run_id`` (latest when ``None``); ``{}`` if unavailable.

    Never raises: metadata enriches the API response but must not turn a successful
    pipeline run into an error.
    """
    try:
        store = _artifact_store()
        run_id = run_id or store.latest_run_id()
        return (store.load_metadata(run_id) or {}) if run_id else {}
    except Exception:
        logger.warning("Could not read model metadata for %s", run_id, exc_info=True)
        return {}


def _pipeline_running(pipeline: str) -> bool:
    """Whether an execution of ``pipeline`` is in progress in Cloud Run (any instance)."""
    settings = get_settings()
    if settings.deployment_mode != "gcp":
        return False
    parent = run_v2.JobsClient.job_path(
        settings.gcp_project_id, settings.gcp_region, _job_name(pipeline)
    )
    try:
        request = run_v2.ListExecutionsRequest(parent=parent, page_size=5)
        # Newest first: a running execution is always among the most recent ones.
        recent = itertools.islice(_executions_client().list_executions(request=request), 5)
        return any(not execution.completion_time for execution in recent)
    except Exception:
        logger.warning("Could not list executions of %s", parent, exc_info=True)
        return False


# --- Pipeline execution ------------------------------------------------------------------------


def _job_name(pipeline: str) -> str:
    settings = get_settings()
    return settings.training_job_name if pipeline == "train" else settings.serving_job_name


def _submit_and_wait(pipeline: str, run_id: str | None = None) -> PipelineExecution:
    """Run ``pipeline`` ("train" or "predict") and block until it finishes."""
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}; expected one of {PIPELINES}")
    settings = get_settings()

    if settings.deployment_mode != "gcp":
        if pipeline == "train":
            output = importlib.import_module("challenge.training").main(run_id=run_id)
        else:
            output = importlib.import_module("challenge.serving").main()
        logger.info("Local %s pipeline finished: %s", pipeline, output)
        return PipelineExecution(name=f"local/{pipeline}/{run_id or 'latest'}", run_id=run_id)

    job_path = run_v2.JobsClient.job_path(
        settings.gcp_project_id, settings.gcp_region, _job_name(pipeline)
    )
    overrides = None
    if run_id:
        overrides = run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[run_v2.EnvVar(name="RUN_ID", value=run_id)]
                )
            ]
        )
    logger.info("Executing Cloud Run job %s (run_id=%s)", job_path, run_id)
    operation = _jobs_client().run_job(
        request=run_v2.RunJobRequest(name=job_path, overrides=overrides)
    )
    execution = operation.result(timeout=settings.pipeline_timeout_s)

    if execution.failed_count or execution.succeeded_count < execution.task_count:
        raise PipelineFailedError(
            f"{_job_name(pipeline)} execution {execution.name} failed "
            f"({execution.failed_count} failed task(s)); logs: {execution.log_uri}"
        )
    logger.info("Cloud Run execution %s succeeded", execution.name)
    return PipelineExecution(name=execution.name, run_id=run_id)


# --- Predictions -------------------------------------------------------------------------------


def _normalise(values: Sequence[Any] | None) -> tuple[Any, ...]:
    return tuple(sorted(set(values))) if values else ()


def _query_predictions(
    page: int,
    page_size: int,
    opera: list[str] | None = None,
    tipovuelo: list[str] | None = None,
    mes: list[int] | None = None,
) -> dict[str, Any]:
    """Return ``{"total_predictions": int, "predictions": [...]}`` for one filtered page.

    Results are cached for ``RESULTS_CACHE_TTL_S`` seconds per (page, filters) key.
    Airlines match case-insensitively, so they travel upper-cased.
    """
    airlines = [name.upper() for name in opera] if opera else None
    key = (page, page_size, _normalise(airlines), _normalise(tipovuelo), _normalise(mes))
    return _results_cache().get_or_compute(key, lambda: _fetch_predictions(*key))


def _fetch_predictions(
    page: int,
    page_size: int,
    opera: tuple[str, ...],
    tipovuelo: tuple[str, ...],
    mes: tuple[int, ...],
) -> dict[str, Any]:
    if get_settings().deployment_mode == "gcp":
        return _fetch_predictions_bigquery(page, page_size, opera, tipovuelo, mes)
    return _fetch_predictions_local(page, page_size, opera, tipovuelo, mes)


def _fetch_predictions_bigquery(
    page: int,
    page_size: int,
    opera: tuple[str, ...],
    tipovuelo: tuple[str, ...],
    mes: tuple[int, ...],
) -> dict[str, Any]:
    settings = get_settings()
    table_id = BigQueryClient.table_id(settings.bq_predictions_table)
    raw_table_id = BigQueryClient.table_id(settings.bq_raw_table)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("opera", "STRING", list(opera)),
            bigquery.ArrayQueryParameter("tipovuelo", "STRING", list(tipovuelo)),
            bigquery.ArrayQueryParameter("mes", "INT64", list(mes)),
            bigquery.ScalarQueryParameter("limit", "INT64", page_size),
            bigquery.ScalarQueryParameter("offset", "INT64", (page - 1) * page_size),
            bigquery.ScalarQueryParameter("delay_threshold", "INT64", DELAY_THRESHOLD_MINUTES),
        ],
        maximum_bytes_billed=_MAX_BYTES_BILLED,
    )
    try:
        rows = list(
            _bigquery_client().query_and_wait(
                _PREDICTIONS_SQL.format(table_id=table_id, raw_table_id=raw_table_id),
                job_config=job_config,
            )
        )
    except NotFound:  # serving has not run yet
        return {"total_predictions": 0, "delayed": 0, "predictions": []}
    result = rows[0]
    return {
        "total_predictions": int(result["total"]),
        "delayed": int(result["delayed"] or 0),
        "predictions": [_prediction_record(row) for row in result["page_rows"]],
    }


def _fetch_predictions_local(
    page: int,
    page_size: int,
    opera: tuple[str, ...],
    tipovuelo: tuple[str, ...],
    mes: tuple[int, ...],
) -> dict[str, Any]:
    path = Path(get_settings().artifacts_dir) / PREDICTIONS_FILENAME
    if not path.is_file():
        return {"total_predictions": 0, "delayed": 0, "predictions": []}
    frame = pd.read_csv(path, dtype={"OPERA": "string", "TIPOVUELO": "string", "MES": "Int64"})
    if opera:
        frame = frame[frame["OPERA"].str.upper().isin(opera)]
    if tipovuelo:
        frame = frame[frame["TIPOVUELO"].isin(tipovuelo)]
    if mes:
        frame = frame[frame["MES"].isin(mes)]
    frame = frame.sort_values(_ORDER_COLUMNS, kind="stable")
    start = (page - 1) * page_size
    page_rows = frame.iloc[start : start + page_size].rename(
        columns={PREDICTION_COLUMN: "predicted_delay"}
    )
    history = _historical_stats_local()
    records = []
    for row in page_rows.to_dict("records"):
        flights, rate = history.get(
            combination_key(row["OPERA"], row["TIPOVUELO"], row["MES"]), (None, None)
        )
        records.append(
            _prediction_record(
                {**row, "historical_flights": flights, "historical_delay_rate": rate}
            )
        )
    return {
        "total_predictions": len(frame),
        "delayed": int(frame[PREDICTION_COLUMN].sum()),
        "predictions": records,
    }


def _historical_stats_local() -> dict[str, tuple[int, float]]:
    """Local equivalent of the ``history`` CTE, computed from ``DATA_DIR/data.csv``."""
    path = Path(get_settings().data_dir) / _RAW_TRAINING_FILENAME
    if not path.is_file():
        return {}
    return _historical_stats_from(str(path), path.stat().st_mtime)


@cache
def _historical_stats_from(path: str, _mtime: float) -> dict[str, tuple[int, float]]:
    # Cached per (file, modification time): recomputed only when data.csv changes.
    raw = pd.read_csv(path, usecols=["Fecha-I", "Fecha-O", *IDENTIFIER_COLUMNS], low_memory=False)
    minutes = (
        pd.to_datetime(raw["Fecha-O"]) - pd.to_datetime(raw["Fecha-I"])
    ).dt.total_seconds() / 60
    raw["delayed"] = (minutes > DELAY_THRESHOLD_MINUTES).astype(int)
    stats = raw.groupby(list(IDENTIFIER_COLUMNS))["delayed"].agg(["count", "mean"])
    return {
        combination_key(*key): (int(row["count"]), round(float(row["mean"]), 4))
        for key, row in stats.iterrows()
    }


def _prediction_record(row: Any) -> dict[str, Any]:
    """Plain record (no numpy/pandas scalars) ready for JSON serialisation."""
    month = row["MES"]
    flights, rate = row.get("historical_flights"), row.get("historical_delay_rate")
    return {
        "OPERA": str(row["OPERA"]),
        "TIPOVUELO": str(row["TIPOVUELO"]),
        "MES": None if month is None or pd.isna(month) else int(month),
        "predicted_delay": int(row["predicted_delay"]),
        "historical_flights": None if flights is None or pd.isna(flights) else int(flights),
        "historical_delay_rate": None if rate is None or pd.isna(rate) else float(rate),
    }


# --- Response enrichment -------------------------------------------------------------------------

_FLIGHT_TYPES = {"I": "International", "N": "National"}
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip


def _serving_summary() -> dict[str, Any] | None:
    """Summary of the last serving run (model lineage + probabilities), cached; never raises."""

    def load() -> dict[str, Any] | None:
        try:
            store = _artifact_store()
            return store.load_serving_summary()
        except Exception:
            logger.warning("Could not read the serving summary", exc_info=True)
            return None

    return _summary_cache().get_or_compute("serving-summary", load)


def _month_name(month: Any) -> str | None:
    return _MONTHS[month - 1] if isinstance(month, int) and 1 <= month <= len(_MONTHS) else None


def enrich_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add readable labels and the delay probability of the last serving run to each record."""
    summary = _serving_summary() or {}
    scores = summary.get("scores") or {}
    enriched = []
    for record in records:
        month = record.get("MES")
        key = combination_key(record["OPERA"], record["TIPOVUELO"], month)
        enriched.append(
            {
                **record,
                "predicted_label": "delayed" if record["predicted_delay"] == 1 else "on_time",
                "flight_type": _FLIGHT_TYPES.get(record["TIPOVUELO"]),
                "month_name": _month_name(month),
                "delay_probability": scores.get(key),
                "historical_flights": record.get("historical_flights"),
                "historical_delay_rate": record.get("historical_delay_rate"),
            }
        )
    return enriched


def summarize(result: dict[str, Any]) -> dict[str, Any] | None:
    """Delayed / on-time counts of the whole filtered set (``None`` if unknown)."""
    if "delayed" not in result:
        return None
    total, delayed = result["total_predictions"], result["delayed"]
    return {
        "delayed": delayed,
        "on_time": total - delayed,
        "delay_rate": round(delayed / total, 4) if total else None,
    }


def model_lineage() -> dict[str, Any] | None:
    """Model that produced the current predictions: version, dates and holdout metrics."""
    summary = _serving_summary()
    if not summary:
        return None
    return {
        "model_version": summary.get("model_version"),
        "trained_at": summary.get("trained_at"),
        "predicted_at": summary.get("predicted_at"),
        "metrics": summary.get("metrics") or {},
    }


def total_pages(total: int, page_size: int) -> int:
    """Number of pages needed to show ``total`` rows."""
    return math.ceil(total / page_size) if total else 0
