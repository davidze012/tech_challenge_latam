"""Pipeline local end-to-end run, gates and guards."""

import json
from pathlib import Path

import pandas as pd
import pytest

from challenge import serving, training
from challenge.config import Settings, clear_settings_cache
from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.model import DelayModel

DATA_DIR = Path("../data")


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    """Local mode, real data/ folder, isolated artifacts directory."""
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    monkeypatch.setenv("DATA_DIR", str(DATA_DIR))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.delenv("MODEL_RUN_ID", raising=False)
    clear_settings_cache()
    yield tmp_path
    clear_settings_cache()


def test_local_training_then_serving_end_to_end(local_env):
    artifact = training.main(run_id="run-e2e")

    assert artifact == str(local_env / "run-e2e" / "model.joblib")
    assert (local_env / "latest.txt").read_text() == "run-e2e"
    metadata = json.loads((local_env / "run-e2e" / "metadata.json").read_text())
    assert metadata["model_version"] == "run-e2e"
    assert metadata["metrics"]["recall_1"] >= training.MIN_RECALL_DELAY
    assert metadata["quality_gate"]["passed"] is True
    assert metadata["dataset"]["n_rows"] == 68206

    destination = serving.main()

    predictions = pd.read_csv(destination)
    assert len(predictions) == 100
    assert list(predictions.columns) == ["OPERA", "TIPOVUELO", "MES", "prediction"]
    assert set(predictions["prediction"]) <= {0, 1}


def test_training_uses_run_id_from_environment(local_env, monkeypatch):
    monkeypatch.setenv("RUN_ID", "run-from-env")
    assert training.main().endswith("run-from-env/model.joblib")


def test_training_rejects_unsafe_run_id(local_env):
    with pytest.raises(ValueError, match="Invalid run_id"):
        training.main(run_id="../escape")


def test_quality_gate_blocks_promotion(local_env, monkeypatch):
    monkeypatch.setattr(training, "MIN_RECALL_DELAY", 0.99)
    with pytest.raises(training.QualityGateError, match="will not be promoted"):
        training.main(run_id="run-gated")
    assert not (local_env / "latest.txt").exists()


def test_serving_can_pin_a_model_version(local_env, monkeypatch):
    training.main(run_id="run-a")
    training.main(run_id="run-b")
    monkeypatch.setenv("MODEL_RUN_ID", "run-a")
    model = serving.build_model(Settings(**_local_settings()))
    assert serving.step_load_model(model, bucket="", run_id="run-a") == "run-a"
    assert serving.main().endswith("predictions.csv")


def test_serving_fails_without_a_trained_model(local_env):
    with pytest.raises(FileNotFoundError, match=r"latest\.txt"):
        serving.main()


def test_step_load_data_rejects_small_or_incomplete_data():
    model = DelayModel(reader=_StaticReader(pd.DataFrame({"OPERA": ["Grupo LATAM"]})))
    with pytest.raises(ValueError, match="missing columns"):
        training.step_load_data(model, "proj", "ds")

    tiny = pd.DataFrame(
        {
            "Fecha-I": ["2017-01-01 10:00:00"],
            "Fecha-O": ["2017-01-01 10:30:00"],
            "OPERA": ["Grupo LATAM"],
            "TIPOVUELO": ["I"],
            "MES": [1],
        }
    )
    with pytest.raises(ValueError, match="at least"):
        training.step_load_data(DelayModel(reader=_StaticReader(tiny)), "proj", "ds")


def test_step_save_refuses_an_untrained_model():
    with pytest.raises(RuntimeError, match="untrained"):
        training.step_save(DelayModel(), "bucket", "run-1", {})


def test_serving_step_load_data_validates_input():
    with pytest.raises(ValueError, match="missing columns"):
        serving.step_load_data(DelayModel(reader=_StaticReader(pd.DataFrame({"x": [1]}))))
    empty = pd.DataFrame(columns=["OPERA", "TIPOVUELO", "MES"])
    with pytest.raises(ValueError, match="empty"):
        serving.step_load_data(DelayModel(reader=_StaticReader(empty)))


def test_build_model_wires_local_connectors():
    model = training.build_model(Settings(**_local_settings()))
    assert isinstance(model._reader, LocalCSVClient)
    assert isinstance(model._store, LocalArtifactStore)


def test_describe_destination_in_gcp_mode():
    settings = Settings(**{**_local_settings(), "deployment_mode": "gcp"})
    assert serving.describe_destination(settings) == "bigquery://proj.ds.predictions"


# --- helpers --------------------------------------------------------------------------------


class _StaticReader:
    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    def read(self) -> pd.DataFrame:
        return self._data

    def write(self, predictions, identifiers) -> None:
        raise NotImplementedError


def _local_settings() -> dict:
    return {
        "deployment_mode": "local",
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


def test_serving_publishes_lineage_and_probabilities(local_env):
    training.main(run_id="run-summary")
    serving.main()

    summary = json.loads((local_env / "serving" / "latest.json").read_text())
    assert summary["model_version"] == "run-summary"
    assert summary["metrics"]["recall_1"] >= training.MIN_RECALL_DELAY
    assert summary["rows"] == 100
    assert 0 <= summary["delayed"] <= 100
    assert len(summary["scores"]) > 0
    assert all(0.0 <= p <= 1.0 for p in summary["scores"].values())


def test_summary_failure_never_fails_serving(local_env, monkeypatch):
    training.main(run_id="run-ok")

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(LocalArtifactStore, "save_serving_summary", broken)
    assert serving.main().endswith("predictions.csv")  # predictions are still written
