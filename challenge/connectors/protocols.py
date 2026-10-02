"""Protocols for data connectors"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

import pandas as pd

# --- Shared contracts used by every connector implementation ----------------------------------

#: Columns that identify a prediction row (written next to each prediction).
IDENTIFIER_COLUMNS: tuple[str, ...] = ("OPERA", "TIPOVUELO", "MES")
PREDICTION_COLUMN = "prediction"

#: Artifact layout shared by the local and GCS stores: ``<root>/<run_id>/<file>``.
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
LATEST_POINTER = "latest.txt"

# run_ids become path segments
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` unchanged if it is a safe path segment, else raise ``ValueError``."""
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            f"Invalid run_id {run_id!r}: use 1-128 chars of letters, digits, '.', '_' or '-'"
        )
    return run_id


def build_predictions_frame(predictions: list[int], identifiers: pd.DataFrame) -> pd.DataFrame:
    """Pair predictions with their identifiers using the canonical output layout.

    Returns: a frame with columns ``OPERA, TIPOVUELO, MES, prediction``,
    row-aligned with ``predictions``.
    """
    if len(predictions) != len(identifiers):
        raise ValueError(
            f"Got {len(predictions)} predictions for {len(identifiers)} identifier rows"
        )
    missing = [column for column in IDENTIFIER_COLUMNS if column not in identifiers.columns]
    if missing:
        raise ValueError(f"Identifiers are missing columns: {missing}")

    frame = identifiers.loc[:, list(IDENTIFIER_COLUMNS)].reset_index(drop=True)
    frame["MES"] = pd.to_numeric(frame["MES"], errors="coerce").astype("Int64")
    frame[PREDICTION_COLUMN] = pd.Series(predictions, dtype="int64")
    return frame


# --- Protocols -----------------------------------------------------------------------------------


@runtime_checkable
class DataReader(Protocol):
    """Interface for loading raw flight data and writing predictions."""

    def read(self) -> pd.DataFrame:
        """Return a DataFrame containing raw flight records."""
        ...

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Persist predictions, paired with their OPERA/TIPOVUELO/MES identifiers."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Interface for persisting and retrieving trained model artifacts."""

    def save(self, model: Any, run_id: str) -> str:
        """Persist model to storage and return the artifact path."""
        ...

    def load(self, model: Any, run_id: str | None) -> None:
        """Deserialise artifact into model._model; resolve latest when run_id is None."""
        ...


@runtime_checkable
class MetadataStore(Protocol):
    """Optional extension of an ArtifactStore: run metadata and the latest-run pointer."""

    def save_metadata(self, run_id: str, metadata: dict[str, Any]) -> str:
        """Persist the run metadata and return its path."""
        ...

    def load_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Return the metadata of ``run_id``, or ``None`` when it does not exist."""
        ...

    def latest_run_id(self) -> str | None:
        """Return the run_id stored in the latest pointer, or ``None`` if there is none."""
        ...
