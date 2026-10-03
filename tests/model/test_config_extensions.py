"""Extensions of challenge.config beyond the provided suite: validation, defaults, logging."""

import json
import logging

import pytest

from challenge.config import (
    JsonFormatter,
    Settings,
    clear_settings_cache,
    configure_logging,
    get_settings,
)


@pytest.fixture(autouse=True)
def reset_settings_cache():
    clear_settings_cache()
    yield
    clear_settings_cache()


def _settings(**overrides) -> Settings:
    values = {
        "deployment_mode": "gcp",
        "data_dir": "data",
        "artifacts_dir": "artifacts",
        "gcp_project_id": "proj",
        "gcp_region": "us-central1",
        "gcp_bucket_artifacts": "artifacts-bucket",
        "gcp_bucket_input": "input-bucket",
        "bq_dataset": "ds",
        "bq_raw_table": "raw_flights",
        "bq_predictions_table": "predictions",
    }
    values.update(overrides)
    return Settings(**values)


def test_valid_gcp_settings_pass_validation():
    assert _settings().deployment_mode == "gcp"


def test_gcp_validation_lists_every_missing_variable():
    with pytest.raises(ValueError) as excinfo:
        _settings(gcp_project_id="", gcp_bucket_input="")
    assert "GCP_PROJECT_ID" in str(excinfo.value)
    assert "GCP_BUCKET_INPUT" in str(excinfo.value)
    assert "GCP_BUCKET_ARTIFACTS" not in str(excinfo.value)


def test_unknown_deployment_mode_is_rejected():
    with pytest.raises(ValueError, match="DEPLOYMENT_MODE must be one of"):
        _settings(deployment_mode="staging")


def test_local_mode_needs_no_gcp_values():
    assert _settings(deployment_mode="local", gcp_project_id="").gcp_project_id == ""


def test_get_settings_defaults_and_normalisation(monkeypatch):
    for var in ("DEPLOYMENT_MODE", "DATA_DIR", "BQ_DATASET", "PIPELINE_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEPLOYMENT_MODE", "  LOCAL ")
    monkeypatch.setenv("DATA_DIR", "")  # blank → default
    monkeypatch.setenv("PIPELINE_TIMEOUT_S", "1200")
    settings = get_settings()
    assert settings.deployment_mode == "local"
    assert settings.data_dir == "data"
    assert settings.bq_dataset == "mle_challenge"
    assert settings.pipeline_timeout_s == 1200


def test_get_settings_is_cached_until_cleared(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "first")
    assert get_settings() is get_settings()
    monkeypatch.setenv("DATA_DIR", "second")
    assert get_settings().data_dir == "first"
    clear_settings_cache()
    assert get_settings().data_dir == "second"


def test_json_formatter_emits_cloud_logging_fields_and_extras():
    record = logging.LogRecord(
        "challenge.test", logging.WARNING, __file__, 1, "hi %s", ("x",), None
    )
    record.run_id = "run-1"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["severity"] == "WARNING"
    assert payload["message"] == "hi x"
    assert payload["run_id"] == "run-1"


def test_configure_logging_switches_formatter_by_env(monkeypatch):
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    try:
        monkeypatch.setenv("LOG_FORMAT", "json")
        configure_logging("debug")
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.DEBUG
        monkeypatch.setenv("LOG_FORMAT", "text")
        configure_logging()
        assert not isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        root.handlers[:], root.level = previous_handlers, previous_level
