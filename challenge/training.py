"""Cloud Run Job entry point for model training.

Pipeline: load raw flights → preprocess → train + evaluate → save metadata
→ model artifact → ``latest.txt`` pointer.
"""

import logging
import math
import os
import platform
import sys
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from challenge.config import MODEL_DISPLAY_NAME, Settings, configure_logging, get_settings
from challenge.connectors import gcs
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import validate_run_id
from challenge.model import (
    CATEGORICAL_COLUMNS,
    DATE_COLUMNS,
    FEATURES_COLS,
    TARGET_COLUMN,
    DelayModel,
)

logger = logging.getLogger(__name__)

TRAINING_FILENAME = "data.csv"

# Below this many rows the dataset is considered broken rather than "small".
MIN_TRAINING_ROWS = 1_000

#: Holdout used to report honest metrics (same split parameters as the DS notebook).
HOLDOUT_SIZE = 0.33
SPLIT_SEED = 42

#: Quality gate
MIN_RECALL_DELAY = 0.60
MIN_F1_DELAY = 0.30


class QualityGateError(RuntimeError):
    """The trained model does not meet the minimum metrics required to be promoted."""


### Define here all the step for the pipeline implementation ###


def step_load_data(model: DelayModel, project: str, bq_dataset: str) -> pd.DataFrame:
    """Load the raw flights and validate they are usable for training."""
    data = model.load_data(project=project, bq_dataset=bq_dataset)
    missing = [c for c in (*DATE_COLUMNS, *CATEGORICAL_COLUMNS) if c not in data.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    if len(data) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Training data has {len(data)} rows; at least {MIN_TRAINING_ROWS} are required"
        )
    logger.info("Loaded %d training rows", len(data))
    return data


def step_preprocess(model: DelayModel, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix and the ``delay`` target."""
    features, target = model.preprocess(data, target_column=TARGET_COLUMN)
    logger.info(
        "Preprocessed %d rows (delay rate %.2f%%)", len(target), 100 * target[TARGET_COLUMN].mean()
    )
    return features, target


def step_train(model: DelayModel, features: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    """Fit on a stratified train split, evaluate on the holdout and apply the quality gate."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=HOLDOUT_SIZE,
        random_state=SPLIT_SEED,
        stratify=target[TARGET_COLUMN],
    )
    model.fit(x_train, y_train)

    y_true = y_test[TARGET_COLUMN]
    y_pred = model.predict(x_test)
    metrics = _classification_metrics(y_true, y_pred, model.predict_proba(x_test))
    logger.info("Holdout metrics: %s", metrics)

    passed = metrics["recall_1"] >= MIN_RECALL_DELAY and metrics["f1_1"] >= MIN_F1_DELAY
    if not passed:
        raise QualityGateError(
            f"Quality gate failed: recall_1={metrics['recall_1']} (min {MIN_RECALL_DELAY}), "
            f"f1_1={metrics['f1_1']} (min {MIN_F1_DELAY}); the model will not be promoted"
        )
    return {
        "metrics": metrics,
        "quality_gate": {
            "min_recall_1": MIN_RECALL_DELAY,
            "min_f1_1": MIN_F1_DELAY,
            "passed": passed,
        },
        "dataset": {
            "n_rows": len(target),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "positive_rate": round(float(target[TARGET_COLUMN].mean()), 4),
            "fingerprint": _fingerprint(features, target),
        },
    }


def step_save(model: DelayModel, bucket: str, run_id: str, report: dict[str, Any]) -> str:
    """Persist metadata, then the model; the store moves ``latest.txt`` last."""
    if not model.is_fitted:
        raise RuntimeError("Refusing to save an untrained model")
    metadata = _build_metadata(model, run_id, report)
    metadata_uri = model.save_metadata(run_id, metadata)
    artifact_uri = model.save(bucket=bucket, run_id=run_id)
    logger.info("Saved model %s (artifact=%s, metadata=%s)", run_id, artifact_uri, metadata_uri)
    return artifact_uri


# --- helpers ----------------------------------------------------------------------------------


def new_run_id() -> str:
    """Return a sortable, unique run id."""
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def build_model(settings: Settings) -> DelayModel:
    """Wire the model with the reader/store."""
    if settings.deployment_mode == "gcp":
        return DelayModel(reader=BigQueryClient(), store=gcs.GCSArtifactStore())
    return DelayModel(reader=LocalCSVClient(filename=TRAINING_FILENAME), store=LocalArtifactStore())


def _classification_metrics(
    y_true: pd.Series, y_pred: list[int], y_score: Any
) -> dict[str, float | int]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {"accuracy": report["accuracy"]}
    for label in ("0", "1"):
        metrics[f"precision_{label}"] = report[label]["precision"]
        metrics[f"recall_{label}"] = report[label]["recall"]
        metrics[f"f1_{label}"] = report[label]["f1-score"]
    metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    metrics = {key: round(float(value), 4) for key, value in metrics.items()}
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def _fingerprint(features: pd.DataFrame, target: pd.DataFrame) -> str:
    """Stable hash of the training matrix, to trace which data produced a model."""
    hashed = pd.util.hash_pandas_object(features.join(target), index=False).sum()
    return f"{int(hashed) & 0xFFFFFFFFFFFFFFFF:016x}"


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("xgboost", "xgboost-cpu", "scikit-learn", "pandas", "numpy", "joblib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Keep scalar, JSON-valid hyper-parameters."""
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, str | bool | int)
        or (isinstance(value, float) and math.isfinite(value))
    }


def _build_metadata(model: DelayModel, run_id: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_version": run_id,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": os.environ.get("GIT_SHA", "unknown"),
        "algorithm": f"{type(model._model).__module__}.{type(model._model).__name__}",
        "params": _json_safe_params(model._model.get_params()),
        "features": FEATURES_COLS,
        **report,
        "library_versions": _library_versions(),
    }


def main(run_id: str | None = None) -> str:
    """Orchestrate the full training pipeline."""
    settings = get_settings()
    run_id = validate_run_id(run_id or os.environ.get("RUN_ID") or new_run_id())
    logger.info("Training run %s started (mode=%s)", run_id, settings.deployment_mode)

    model = build_model(settings)
    data = step_load_data(model, settings.gcp_project_id, settings.bq_dataset)
    features, target = step_preprocess(model, data)
    report = step_train(model, features, target)
    artifact_uri = step_save(model, settings.gcp_bucket_artifacts, run_id, report)

    logger.info("Training run %s finished: %s", run_id, artifact_uri)
    return artifact_uri


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)
