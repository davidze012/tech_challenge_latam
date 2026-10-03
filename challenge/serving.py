"""Cloud Run Job entry point for model serving.

Pipeline: load the latest (or pinned) model → load the serving input → preprocess →
predict → write the predictions (BigQuery in GCP, ``predictions.csv`` locally).
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

from challenge.config import Settings, configure_logging, get_settings
from challenge.connectors import gcs
from challenge.connectors.local import PREDICTIONS_FILENAME, LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import IDENTIFIER_COLUMNS
from challenge.model import DelayModel

logger = logging.getLogger(__name__)

#: Local-mode serving file inside ``DATA_DIR``.
SERVING_FILENAME = "serving_input.csv"

### Define here all the step for the pipeline implementation ###


def step_load_model(model: DelayModel, bucket: str, run_id: str | None = None) -> str | None:
    """Load the model to serve: ``run_id`` if pinned, otherwise the latest one.

    Fails loudly when nothing can be loaded, so the job never serves the untrained fallback.
    """
    resolved = run_id or model.latest_run_id()
    model.load(bucket=bucket, run_id=resolved)
    if not model.is_fitted:
        raise RuntimeError("No trained model could be loaded; run the training pipeline first")
    logger.info("Serving model %s", resolved or "latest")
    return resolved


def step_load_data(model: DelayModel) -> pd.DataFrame:
    """Load the serving input and validate it has the identifier columns."""
    data = model.load_data()
    missing = [column for column in IDENTIFIER_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Serving input is missing columns: {missing}")
    if data.empty:
        raise ValueError("Serving input is empty")
    logger.info("Loaded %d rows to score", len(data))
    return data


def step_preprocess(model: DelayModel, data: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix (rows are never dropped at serving time)."""
    return model.preprocess(data)


def step_predict(model: DelayModel, features: pd.DataFrame) -> list[int]:
    """Score every row."""
    predictions = model.predict(features)
    logger.info("Predicted %d rows (%d delayed)", len(predictions), sum(predictions))
    return predictions


def step_write(model: DelayModel, predictions: list[int], data: pd.DataFrame) -> None:
    """Persist the predictions next to their OPERA/TIPOVUELO/MES identifiers."""
    model.write_predictions(predictions, data.loc[:, list(IDENTIFIER_COLUMNS)])


# --- helpers ----------------------------------------------------------------------------------


def build_model(settings: Settings) -> DelayModel:
    """Wire the model with the reader/store of the current deployment mode."""
    if settings.deployment_mode == "gcp":
        return DelayModel(
            reader=gcs.GCSCSVClient(blob_name=settings.serving_input_blob),
            store=gcs.GCSArtifactStore(),
        )
    return DelayModel(reader=LocalCSVClient(filename=SERVING_FILENAME), store=LocalArtifactStore())


def describe_destination(settings: Settings) -> str:
    """Human-readable location of the predictions sink."""
    if settings.deployment_mode == "gcp":
        return (
            f"bigquery://{settings.gcp_project_id}.{settings.bq_dataset}"
            f".{settings.bq_predictions_table}"
        )
    return str(Path(os.environ.get("ARTIFACTS_DIR") or "artifacts") / PREDICTIONS_FILENAME)


def main() -> str:
    """Orchestrate the full serving pipeline."""
    settings = get_settings()
    model = build_model(settings)

    model_version = step_load_model(
        model, settings.gcp_bucket_artifacts, os.environ.get("MODEL_RUN_ID") or None
    )
    data = step_load_data(model)
    features = step_preprocess(model, data)
    predictions = step_predict(model, features)
    step_write(model, predictions, data)

    destination = describe_destination(settings)
    logger.info(
        "Serving finished: %d predictions written to %s (model=%s)",
        len(predictions),
        destination,
        model_version,
    )
    return destination


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        logger.exception("Serving job failed")
        sys.exit(1)
