"""Flight-delay model for SCL departures: preprocessing, training and inference."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from challenge.config import DELAY_THRESHOLD_MINUTES
from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import (
    IDENTIFIER_COLUMNS,
    ArtifactStore,
    DataReader,
    MetadataStore,
)

logger = logging.getLogger(__name__)

FEATURES_COLS: list[str] = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]
TARGET_COLUMN = "delay"

#: Raw columns the features are derived from.
CATEGORICAL_COLUMNS: tuple[str, ...] = IDENTIFIER_COLUMNS

#: Raw columns the target is derived from (scheduled and operated datetimes).
DATE_COLUMNS: tuple[str, str] = ("Fecha-I", "Fecha-O")

#: Hyper-parameters from the DS notebook; the challenge does not ask for model improvements.
XGB_PARAMS: dict[str, Any] = {"learning_rate": 0.01, "random_state": 1}


def _require_columns(data: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def _to_datetime(values: pd.Series) -> pd.Series:
    """Parse to datetime; BigQuery DATETIME columns already arrive as datetime64."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    return pd.to_datetime(values, format="ISO8601", errors="coerce")


def compute_delay(
    data: pd.DataFrame, threshold_minutes: int = DELAY_THRESHOLD_MINUTES
) -> pd.Series:
    """Return the delay label: 1 if ``Fecha-O - Fecha-I > threshold_minutes``, else 0."""
    _require_columns(data, DATE_COLUMNS)
    scheduled, operated = (_to_datetime(data[column]) for column in DATE_COLUMNS)
    min_diff = (operated - scheduled).dt.total_seconds() / 60
    delay = (min_diff > threshold_minutes).astype("Int64")
    return delay.mask(min_diff.isna()).rename(TARGET_COLUMN)


class BaseDelayModel(ABC):
    """Base model - derived from the notebook"""

    def __init__(self) -> None:
        self._model = None  # fitted XGBClassifier, populated by fit()

    @abstractmethod
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference.

        Parameters
        ----------
        data : pd.DataFrame
            Raw flight records with columns needed for training and/or inference.
        target_column : str, optional
            Name of the target column to return (e.g. "delay").

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Features and target DataFrames when target_column is provided.
        pd.DataFrame
            Features DataFrame only when target_column is None.
        """

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train the selected model.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).
        target : pd.DataFrame
            Binary delay labels (column = "delay").
        """

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions for each row in features.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).

        Returns
        -------
        list[int]
            0 (no delay) or 1 (delayed) for each input row.
        """


class GCPModelMixin:
    """Optional GCP I/O operations."""

    def __init__(
        self,
        reader: DataReader | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Initialise with optional injectable clients.

        Parameters
        ----------
        reader : DataReader, optional
            Data-loading client; defaults to ``LocalCSVClient()``.
        store : ArtifactStore, optional
            Artifact-persistence client; defaults to ``LocalArtifactStore()``.
        """
        # Cooperative init: continues the MRO into BaseDelayModel, which sets ``_model``.
        super().__init__()
        self._reader: DataReader = reader if reader is not None else LocalCSVClient()
        self._store: ArtifactStore = store if store is not None else LocalArtifactStore()

    def load_data(self, project: str = "", bq_dataset: str = "") -> pd.DataFrame:
        """Load raw flight data via the injected DataReader.

        Parameters
        ----------
        project : str
            GCP project ID (unused in local mode; accepted for API compatibility).
        bq_dataset : str
            BigQuery dataset name (unused in local mode; accepted for API compatibility).

        Returns
        -------
        pd.DataFrame
            Raw flight records ready to pass to preprocess().
        """
        # The reader is already bound to its source; project/dataset are logged for lineage.
        logger.info(
            "Loading data with %s (project=%r, dataset=%r)",
            type(self._reader).__name__,
            project,
            bq_dataset,
        )
        return self._reader.read()

    def save(self, bucket: str = "", run_id: str = "") -> str:
        """Persist the fitted model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str
            Unique identifier for this training run.

        Returns
        -------
        str
            Path or URI of the saved artifact.
        """
        return self._store.save(self, run_id)

    def load(self, bucket: str = "", run_id: str | None = None) -> None:
        """Restore a model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str, optional
            Specific run to load; latest artifact when omitted.
        """
        self._store.load(self, run_id)

    def write_predictions(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Persist predictions through the injected DataReader's sink."""
        self._reader.write(predictions, identifiers)

    def save_metadata(self, run_id: str, metadata: dict[str, Any]) -> str:
        """Persist run metadata when the store supports it (see ``MetadataStore``)."""
        if not isinstance(self._store, MetadataStore):
            raise TypeError(f"{type(self._store).__name__} does not support run metadata")
        return self._store.save_metadata(run_id, metadata)

    def latest_run_id(self) -> str | None:
        """Return the store's most recent run_id (``None`` if unknown or unsupported)."""
        if not isinstance(self._store, MetadataStore):
            return None
        return self._store.latest_run_id()

    @property
    def is_fitted(self) -> bool:
        """Whether an estimator has been trained or loaded."""
        return getattr(self, "_model", None) is not None


class DelayModel(GCPModelMixin, BaseDelayModel):
    """Balanced XGBoost classifier that predicts if a SCL flight departs >15 min late."""

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference."""
        features = self._build_features(data)
        if target_column is None:
            return features

        target = self._build_target(data, target_column)
        valid = target.notna()
        if not valid.all():
            logger.warning(
                "Dropping %d rows with an undefined %r target",
                int((~valid).sum()),
                target_column,
            )
            features, target = features[valid], target[valid]
        return features, target.astype("int64").to_frame(name=target_column)

    @staticmethod
    def _build_features(data: pd.DataFrame) -> pd.DataFrame:
        _require_columns(data, CATEGORICAL_COLUMNS)
        month = pd.to_numeric(data["MES"], errors="coerce")
        categorical = pd.DataFrame(
            {
                "OPERA": data["OPERA"].astype("string").str.strip(),
                "TIPOVUELO": data["TIPOVUELO"].astype("string").str.strip().str.upper(),
                # Nullable Int64 (BigQuery's INT64 dtype)
                # yields MES_7, not MES_7.0.
                "MES": month.where(month == month.round()).astype("Int64"),
            },
            index=data.index,
        )
        dummies = pd.get_dummies(categorical, columns=list(CATEGORICAL_COLUMNS))

        return dummies.reindex(columns=FEATURES_COLS, fill_value=0).astype("int64")

    @staticmethod
    def _build_target(data: pd.DataFrame, target_column: str) -> pd.Series:
        if target_column in data.columns:
            return pd.to_numeric(data[target_column], errors="coerce").astype("Int64")
        if target_column == TARGET_COLUMN:
            return compute_delay(data)
        raise ValueError(f"Target column {target_column!r} is not in the data")

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train model with class-imbalance weighting."""
        x_train = self._select_features(features)
        y_train = self._as_label_series(target)
        if len(x_train) != len(y_train):
            raise ValueError(f"features has {len(x_train)} rows but target has {len(y_train)}")

        n_negative = int((y_train == 0).sum())
        n_positive = int((y_train == 1).sum())
        if n_negative == 0 or n_positive == 0:
            raise ValueError("The training target must contain both classes (0 and 1)")

        scale_pos_weight = n_negative / n_positive
        self._model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
        self._model.fit(x_train, y_train)
        logger.info(
            "Trained XGBClassifier on %d rows (positives=%d, scale_pos_weight=%.4f)",
            len(y_train),
            n_positive,
            scale_pos_weight,
        )

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions."""
        if self._model is None:
            logger.warning("predict() called on an untrained model: returning 0 for every row")
            return [0] * len(features)
        predictions = self._model.predict(self._select_features(features))
        return np.asarray(predictions, dtype=int).tolist()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return the predicted probability of delay for each row."""
        if self._model is None:
            raise RuntimeError("The model is not trained: call fit() or load() first")
        return self._model.predict_proba(self._select_features(features))[:, 1]

    @staticmethod
    def _select_features(features: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in FEATURES_COLS if column not in features.columns]
        if missing:
            raise ValueError(f"Features are missing columns: {missing}")
        return features.loc[:, FEATURES_COLS]

    @staticmethod
    def _as_label_series(target: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(target, pd.DataFrame):
            if target.shape[1] != 1:
                raise ValueError(f"target must have exactly one column, got {target.shape[1]}")
            target = target.iloc[:, 0]
        return target.astype("int64")
