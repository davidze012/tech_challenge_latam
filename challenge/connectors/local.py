"""Data-reader and Artifactor local implementation."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from challenge.connectors.protocols import (
    LATEST_POINTER,
    METADATA_FILENAME,
    MODEL_FILENAME,
    build_predictions_frame,
    validate_run_id,
)

logger = logging.getLogger(__name__)

PREDICTIONS_FILENAME = "predictions.csv"


def _artifacts_root() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR") or "artifacts")


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a temp file in the same directory, then atomically swap it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


class LocalCSVClient:
    """Read a CSV from the local data directory and write predictions to disk."""

    def __init__(self, filename: str | None = None) -> None:
        """Initialise the reader with an optional specific filename."""
        self._filename = filename

    def _resolve_path(self) -> Path:
        data_dir = Path(os.environ.get("DATA_DIR") or "data")
        if self._filename is not None:
            path = data_dir / self._filename
            if not path.is_file():
                raise FileNotFoundError(f"CSV not found: {path}")
            return path

        candidates = sorted(data_dir.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        if len(candidates) > 1:
            names = ", ".join(candidate.name for candidate in candidates)
            raise ValueError(
                f"Found {len(candidates)} CSV files in {data_dir} ({names}); "
                "pass filename=... to choose one"
            )
        return candidates[0]

    def read(self) -> pd.DataFrame:
        """Return a DataFrame from the target CSV file."""
        path = self._resolve_path()
        data = pd.read_csv(path, low_memory=False)
        logger.info("Read %d rows from %s", len(data), path)
        return data

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to ``<ARTIFACTS_DIR>/predictions.csv``.

        Output columns: 
        ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) followed by ``prediction`` (0|1).
        """
        frame = build_predictions_frame(predictions, identifiers)
        path = _artifacts_root() / PREDICTIONS_FILENAME
        _atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False))
        logger.info("Wrote %d predictions to %s", len(frame), path)


class LocalArtifactStore:
    """Save and load model artifacts on the local filesystem.

    Layout::

        <ARTIFACTS_DIR>/
            <run_id>/model.joblib
            <run_id>/metadata.json
            latest.txt           ← contains the most-recent run_id
    """

    def _root(self) -> Path:
        return Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

    def save(self, model: Any, run_id: str) -> str:
        """Save model to <ARTIFACTS_DIR>/<run_id>/model.joblib and update latest.txt."""
        validate_run_id(run_id)
        model_path = self._root() / run_id / MODEL_FILENAME
        _atomic_write(model_path, lambda tmp: joblib.dump(model._model, tmp))
        _atomic_write(self._root() / LATEST_POINTER, lambda tmp: tmp.write_text(run_id))
        logger.info("Saved model artifact %s", model_path)
        return str(model_path)

    def load(self, model: Any, run_id: str | None) -> None:
        """Load model from <ARTIFACTS_DIR>/<run_id>/model.joblib into model._model."""
        if run_id is None:
            run_id = self.latest_run_id()
            if run_id is None:
                raise FileNotFoundError(
                    f"No trained model available: {self._root() / LATEST_POINTER} not found"
                )
        model_path = self._root() / validate_run_id(run_id) / MODEL_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        model._model = joblib.load(model_path)
        logger.info("Loaded model artifact %s", model_path)

    def latest_run_id(self) -> str | None:
        """Return the run_id in latest.txt, or ``None`` when no model was saved yet."""
        pointer = self._root() / LATEST_POINTER
        if not pointer.is_file():
            return None
        return pointer.read_text().strip() or None

    def save_metadata(self, run_id: str, metadata: dict[str, Any]) -> str:
        """Write ``<ARTIFACTS_DIR>/<run_id>/metadata.json`` and return its path."""
        path = self._root() / validate_run_id(run_id) / METADATA_FILENAME
        payload = json.dumps(metadata, indent=2, default=str)
        _atomic_write(path, lambda tmp: tmp.write_text(payload))
        return str(path)

    def load_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Return the metadata of ``run_id``, or ``None`` if it does not exist."""
        path = self._root() / validate_run_id(run_id) / METADATA_FILENAME
        if not path.is_file():
            return None
        return json.loads(path.read_text())
