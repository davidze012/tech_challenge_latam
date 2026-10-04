"""Pipeline control FastAPI — triggers Cloud Run Job training/serving pipelines."""

import logging
import threading
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from challenge.api.utils import (
    _check_model_exists,
    _check_raw_flights,
    _latest_model_version,
    _pipeline_running,
    _query_predictions,
    _read_model_metadata,
    _submit_and_wait,
    enrich_predictions,
    invalidate_predictions_cache,
    model_lineage,
    summarize,
    total_pages,
    warm_up,
)
from challenge.config import MODEL_DISPLAY_NAME, configure_logging, get_settings

logger = logging.getLogger(__name__)

#: One execution per pipeline at a time on this instance.
_PIPELINE_LOCKS = {"train": threading.Lock(), "predict": threading.Lock()}
_VALID_FLIGHT_TYPES = {"I", "N"}
_VALID_MONTHS = set(range(1, 13))


# --- Schemas -----------------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: Literal["ok"]


class TrainResponse(BaseModel):
    """Metadata of the model produced by a training run."""

    # Allow field names starting with ``model_``.
    model_config = ConfigDict(protected_namespaces=())

    model_display_name: str = Field(examples=[MODEL_DISPLAY_NAME])
    model_version: str | None = Field(examples=["20261002T202458Z-a1a006fe"])
    pipeline_job_id: str
    trained_at: str | None = None
    metrics: dict[str, float | int] = Field(description="Holdout metrics of the trained model")


class Prediction(BaseModel):
    """A flight, its predicted delay (1 = delayed more than 15 minutes) and readable context."""

    OPERA: str
    TIPOVUELO: str
    MES: int | None
    predicted_delay: int
    predicted_label: Literal["delayed", "on_time"]
    flight_type: str | None = Field(None, examples=["International"])
    month_name: str | None = Field(None, examples=["July"])
    delay_probability: float | None = Field(
        None, description="Probability of delay estimated by the model that made the prediction"
    )
    historical_flights: int | None = Field(
        None, description="Flights with this airline, flight type and month in the 2017 data"
    )
    historical_delay_rate: float | None = Field(
        None, description="Share of those historical flights that were delayed (> 15 min)"
    )


class Summary(BaseModel):
    """Delayed / on-time counts over the whole filtered set (not just the page)."""

    delayed: int
    on_time: int
    delay_rate: float | None


class ModelLineage(BaseModel):
    """Model that produced the current predictions."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str | None
    trained_at: str | None
    predicted_at: str | None
    metrics: dict[str, float | int] = {}


class Filters(BaseModel):
    """Filters applied to the page (``null`` = not filtered)."""

    opera: list[str] | None
    tipovuelo: list[str] | None
    mes: list[int] | None


class Links(BaseModel):
    """Relative links to the neighbouring pages (``null`` when there is none)."""

    next: str | None
    prev: str | None


class PredictResponse(BaseModel):
    """Result of a serving run: the first predictions of the refreshed table."""

    model_config = ConfigDict(protected_namespaces=())

    pipeline_job_id: str
    model_version: str | None
    total_predictions: int
    summary: Summary | None = None
    model: ModelLineage | None = None
    predictions: list[Prediction]


class PredictionsPage(BaseModel):
    """A page of the predictions table."""

    page: int
    page_size: int
    total_predictions: int
    total_pages: int
    filters: Filters
    summary: Summary | None = None
    model: ModelLineage | None = None
    links: Links
    predictions: list[Prediction]


# --- App ---------------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and warm up GCP clients on startup."""
    configure_logging()
    settings = get_settings()
    logger.info(
        "Starting pipeline control API (mode=%s, project=%s, region=%s)",
        settings.deployment_mode,
        settings.gcp_project_id or "-",
        settings.gcp_region,
    )
    warm_up()
    yield


app = FastAPI(
    title="Flight Delay Pipelines API",
    summary="Control plane for the SCL flight-delay training and batch-serving pipelines.",
    version="1.0.0",
    lifespan=lifespan,
)


@contextmanager
def _exclusive(pipeline: str) -> Iterator[None]:
    """Reject a run while another execution of the same pipeline is in progress."""
    conflict = HTTPException(
        status_code=409, detail=f"A {pipeline} pipeline execution is already running"
    )
    lock = _PIPELINE_LOCKS[pipeline]
    if not lock.acquire(blocking=False):  # running on this instance
        raise conflict
    try:
        if _pipeline_running(pipeline):  # running from another instance
            raise conflict
        yield
    finally:
        lock.release()


def _run_pipeline(pipeline: str, run_id: str | None = None) -> Any:
    """Run a pipeline mapping failures to HTTP errors with a ``{"details": ...}`` body."""
    try:
        return _submit_and_wait(pipeline, run_id)
    except TimeoutError as exc:
        logger.exception("%s pipeline timed out", pipeline)
        raise HTTPException(status_code=504, detail={"details": str(exc)}) from exc
    except Exception as exc:
        logger.exception("%s pipeline failed", pipeline)
        raise HTTPException(status_code=500, detail={"details": str(exc)}) from exc


def _split_csv(raw: str | None) -> list[str] | None:
    """Split a comma-separated query value into trimmed, non-empty items (None if empty)."""
    items = [item.strip() for item in (raw or "").split(",") if item.strip()]
    return items or None


def _parse_months(raw: str | None) -> list[int] | None:
    """Parse ``mes`` ("3,6,12") into ints, rejecting anything outside 1-12 with a 422."""
    items = _split_csv(raw)
    if items is None:
        return None
    try:
        months = [int(item) for item in items]
    except ValueError:
        months = []
    if not months or not set(months) <= _VALID_MONTHS:
        raise HTTPException(
            status_code=422,
            detail="mes must be a comma-separated list of integers between 1 and 12",
        )
    return months


@app.get("/health", status_code=200, response_model=HealthResponse)
async def get_health() -> dict:
    """Return liveness status."""
    return {"status": "ok"}


@app.post("/pipeline/train", status_code=200, response_model=TrainResponse)
def post_pipeline_train() -> dict:
    """Trigger the training Cloud Run Job and return model metadata."""
    if _check_raw_flights() == 0:
        raise HTTPException(status_code=422, detail="raw_flights table is empty")

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    with _exclusive("train"):
        job = _run_pipeline("train", run_id)

    metadata = _read_model_metadata(run_id)
    return {
        "model_display_name": metadata.get("model_display_name", MODEL_DISPLAY_NAME),
        "model_version": metadata.get("model_version", run_id),
        "pipeline_job_id": job.name,
        "trained_at": metadata.get("trained_at"),
        "metrics": metadata.get("metrics", {}),
    }


@app.post("/pipeline/predict", status_code=200, response_model=PredictResponse)
def post_pipeline_predict() -> dict:
    """Trigger the serving Cloud Run Job and return first 10 predictions."""
    if not _check_model_exists():
        raise HTTPException(
            status_code=422,
            detail="no trained model available; run POST /pipeline/train first",
        )

    with _exclusive("predict"):
        job = _run_pipeline("predict")

    invalidate_predictions_cache()
    result = _query_predictions(1, 10, None, None, None)
    return {
        "pipeline_job_id": job.name,
        "model_version": _latest_model_version(),
        "total_predictions": result["total_predictions"],
        "summary": summarize(result),
        "model": model_lineage(),
        "predictions": enrich_predictions(result["predictions"]),
    }


@app.get("/pipeline/predict/results", status_code=200, response_model=PredictionsPage)
def get_prediction_results(
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(10, ge=1, le=100, description="Rows per page (max 100)"),
    opera: str | None = Query(None, description="Comma-separated airlines, e.g. Grupo LATAM"),
    tipovuelo: str | None = Query(None, description="Comma-separated flight types: I, N"),
    mes: str | None = Query(None, description="Comma-separated months 1-12, e.g. 7,12"),
) -> dict:
    """Return a paginated page of predictions, optionally filtered by OPERA/TIPOVUELO/MES."""
    opera_list = _split_csv(opera)

    tipovuelo_list = _split_csv(tipovuelo)
    if tipovuelo_list:
        tipovuelo_list = [value.upper() for value in tipovuelo_list]
        if not set(tipovuelo_list) <= _VALID_FLIGHT_TYPES:
            raise HTTPException(status_code=422, detail="tipovuelo must be I and/or N")

    mes_list = _parse_months(mes)
    result = _query_predictions(page, page_size, opera_list, tipovuelo_list, mes_list)
    pages = total_pages(result["total_predictions"], page_size)
    filters = {"opera": opera_list, "tipovuelo": tipovuelo_list, "mes": mes_list}
    return {
        "page": page,
        "page_size": page_size,
        "total_predictions": result["total_predictions"],
        "total_pages": pages,
        "filters": filters,
        "summary": summarize(result),
        "model": model_lineage(),
        "links": _page_links(page, page_size, pages, filters),
        "predictions": enrich_predictions(result["predictions"]),
    }


def _page_links(
    page: int, page_size: int, pages: int, filters: dict[str, list[Any] | None]
) -> dict[str, str | None]:
    """Relative URLs of the next/previous pages, preserving page_size and the filters."""
    query = {name: ",".join(map(str, values)) for name, values in filters.items() if values}

    def link(target: int) -> str:
        params = {"page": target, "page_size": page_size, **query}
        return f"/pipeline/predict/results?{urlencode(params)}"

    return {
        "next": link(page + 1) if page < pages else None,
        "prev": link(min(page - 1, pages)) if page > 1 and pages > 0 else None,
    }
