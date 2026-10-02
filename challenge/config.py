"""Centralised, lazily-loaded configuration."""

import json
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache

VALID_DEPLOYMENT_MODES = ("local", "gcp")

# Settings fields that must be non-empty in GCP mode, mapped to the env var that feeds them.
_REQUIRED_GCP_SETTINGS = {
    "gcp_project_id": "GCP_PROJECT_ID",
    "gcp_bucket_artifacts": "GCP_BUCKET_ARTIFACTS",
    "gcp_bucket_input": "GCP_BUCKET_INPUT",
}


@dataclass(frozen=True)
class Settings:
    """environment-variable configuration."""

    deployment_mode: str
    data_dir: str
    artifacts_dir: str
    gcp_project_id: str
    gcp_region: str
    gcp_bucket_artifacts: str
    gcp_bucket_input: str
    bq_dataset: str
    bq_raw_table: str
    bq_predictions_table: str
    bq_location: str = "us-central1"
    training_job_name: str = "mle-training"
    serving_job_name: str = "mle-serving"
    serving_input_blob: str = "serving_input.csv"
    pipeline_timeout_s: int = 900
    results_cache_ttl_s: int = 30

    def __post_init__(self) -> None:
        """Validate required GCP vars when deployment_mode is 'gcp'."""
        if self.deployment_mode not in VALID_DEPLOYMENT_MODES:
            raise ValueError(
                f"DEPLOYMENT_MODE must be one of {VALID_DEPLOYMENT_MODES}, "
                f"got {self.deployment_mode!r}"
            )
        if self.deployment_mode == "gcp":
            missing = [
                env_var
                for field_name, env_var in _REQUIRED_GCP_SETTINGS.items()
                if not getattr(self, field_name)
            ]
            if missing:
                raise ValueError(
                    "Missing required env vars for DEPLOYMENT_MODE=gcp: " + ", ".join(missing)
                )


def _env(name: str, default: str = "") -> str:
    """Return a stripped env var, falling back to ``default`` when unset or blank."""
    return (os.environ.get(name) or "").strip() or default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings read from environment variables."""
    return Settings(
        deployment_mode=_env("DEPLOYMENT_MODE", "local").lower(),
        data_dir=_env("DATA_DIR", "data"),
        artifacts_dir=_env("ARTIFACTS_DIR", "artifacts"),
        gcp_project_id=_env("GCP_PROJECT_ID"),
        gcp_region=_env("GCP_REGION", "us-central1"),
        gcp_bucket_artifacts=_env("GCP_BUCKET_ARTIFACTS"),
        gcp_bucket_input=_env("GCP_BUCKET_INPUT"),
        bq_dataset=_env("BQ_DATASET", "mle_challenge"),
        bq_raw_table=_env("BQ_RAW_TABLE", "raw_flights"),
        bq_predictions_table=_env("BQ_PREDICTIONS_TABLE", "predictions"),
        bq_location=_env("BQ_LOCATION", "us-central1"),
        training_job_name=_env("TRAINING_JOB_NAME", "mle-training"),
        serving_job_name=_env("SERVING_JOB_NAME", "mle-serving"),
        serving_input_blob=_env("SERVING_INPUT_BLOB", "serving_input.csv"),
        pipeline_timeout_s=int(_env("PIPELINE_TIMEOUT_S", "900")),
        results_cache_ttl_s=int(_env("RESULTS_CACHE_TTL_S", "30")),
    )


def clear_settings_cache() -> None:
    """Invalidate the cached settings."""
    get_settings.cache_clear()


# --- Logging ---------------------------------------------------------------------------------

# Attributes every LogRecord has; anything else was passed through ``extra=`` and is logged.
_STANDARD_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", (), None))
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Single-line JSON records that Cloud Logging parses natively (``severity``/``message``)."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record, including any ``extra=`` fields, as JSON."""
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        payload.update(
            {
                key: value
                for key, value in vars(record).items()
                if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_")
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str | None = None) -> None:
    """Configure root logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    if _env("LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel((level or _env("LOG_LEVEL", "INFO")).upper())
