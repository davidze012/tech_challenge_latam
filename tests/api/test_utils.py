"""Unit tests for challenge.api.utils"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound

from challenge import app
from challenge.api import api as api_module
from challenge.api import utils
from challenge.config import clear_settings_cache

PREDICTIONS = pd.DataFrame(
    {
        "OPERA": ["Sky Airline", "Grupo LATAM", "Grupo LATAM", "Copa Air", "Grupo LATAM"],
        "TIPOVUELO": ["N", "I", "N", "I", "I"],
        "MES": [7, 12, 7, 1, 7],
        "prediction": [1, 0, 1, 0, 1],
    }
)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Isolate settings and the module-level caches/clients between tests."""
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    clear_settings_cache()
    utils._results_cache.cache_clear()
    yield
    clear_settings_cache()
    utils._results_cache.cache_clear()


@pytest.fixture
def local_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    clear_settings_cache()
    return tmp_path


@pytest.fixture
def gcp_mode(monkeypatch):
    for name, value in {
        "DEPLOYMENT_MODE": "gcp",
        "GCP_PROJECT_ID": "proj",
        "GCP_REGION": "us-central1",
        "GCP_BUCKET_ARTIFACTS": "artifacts-bucket",
        "GCP_BUCKET_INPUT": "input-bucket",
        "BQ_DATASET": "ds",
        "TRAINING_JOB_NAME": "mle-training-prod",
        "SERVING_JOB_NAME": "mle-serving-prod",
    }.items():
        monkeypatch.setenv(name, value)
    clear_settings_cache()


# --- TTLCache ------------------------------------------------------------------------------------


def test_cache_returns_cached_value_until_ttl_expires():
    now = [0.0]
    cache = utils.TTLCache(ttl_seconds=30, clock=lambda: now[0])
    compute = MagicMock(side_effect=[1, 2])
    assert cache.get_or_compute("k", compute) == 1
    assert cache.get_or_compute("k", compute) == 1
    now[0] = 31
    assert cache.get_or_compute("k", compute) == 2
    assert compute.call_count == 2


def test_cache_is_single_flight_under_concurrency():
    cache = utils.TTLCache(ttl_seconds=30)
    calls = []

    def slow_compute():
        calls.append(1)
        time.sleep(0.05)
        return "value"

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.get_or_compute("k", slow_compute)))
        for _ in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == ["value"] * 20
    assert len(calls) == 1


def test_cache_invalidate_and_bounded_size():
    cache = utils.TTLCache(ttl_seconds=30, max_entries=2)
    for key in ("a", "b", "c"):
        cache.get_or_compute(key, lambda key=key: key)
    assert len(cache._entries) == 2
    cache.invalidate()
    assert cache._entries == {}


# --- local mode ----------------------------------------------------------------------------------


def test_local_query_filters_sorts_and_paginates(local_artifacts):
    PREDICTIONS.to_csv(local_artifacts / "predictions.csv", index=False)

    result = utils._query_predictions(1, 2, ["Grupo LATAM"], None, [7, 12])
    assert result["total_predictions"] == 3
    assert result["predictions"] == [
        {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 7, "predicted_delay": 1},
        {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 12, "predicted_delay": 0},
    ]
    page_2 = utils._query_predictions(2, 2, ["Grupo LATAM"], None, [7, 12])
    assert [p["TIPOVUELO"] for p in page_2["predictions"]] == ["N"]
    beyond = utils._query_predictions(9, 2, ["Grupo LATAM"], None, None)
    assert beyond == {"total_predictions": 3, "predictions": []}


def test_local_query_without_predictions_file_is_empty(local_artifacts):
    assert utils._query_predictions(1, 10) == {"total_predictions": 0, "predictions": []}


def test_cache_key_ignores_filter_order_and_duplicates(local_artifacts):
    with patch.object(utils, "_fetch_predictions", return_value={"x": 1}) as fetch:
        utils._query_predictions(1, 10, ["b", "a"], None, [12, 7, 7])
        utils._query_predictions(1, 10, ["a", "b"], None, [7, 12])
    fetch.assert_called_once_with(1, 10, ("a", "b"), (), (7, 12))


def test_local_checks_and_metadata(local_artifacts, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text("a\n1\n2\n3\n")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    clear_settings_cache()
    assert utils._check_raw_flights() == 3
    assert utils._check_model_exists() is False
    assert utils._read_model_metadata() == {}

    (local_artifacts / "latest.txt").write_text("run-1")
    (local_artifacts / "run-1").mkdir()
    (local_artifacts / "run-1" / "metadata.json").write_text('{"model_version": "run-1"}')
    assert utils._check_model_exists() is True
    assert utils._read_model_metadata() == {"model_version": "run-1"}
    assert utils._latest_model_version() == "run-1"


def test_local_submit_runs_pipelines_in_process():
    training = SimpleNamespace(main=MagicMock(return_value="artifacts/run-1/model.joblib"))
    serving = SimpleNamespace(main=MagicMock(return_value="artifacts/predictions.csv"))
    modules = {"challenge.training": training, "challenge.serving": serving}
    with patch.object(utils.importlib, "import_module", side_effect=modules.__getitem__):
        assert utils._submit_and_wait("train", "run-1").name == "local/train/run-1"
        assert utils._submit_and_wait("predict").name == "local/predict/latest"
    training.main.assert_called_once_with(run_id="run-1")
    serving.main.assert_called_once_with()


def test_submit_rejects_unknown_pipeline():
    with pytest.raises(ValueError, match="Unknown pipeline"):
        utils._submit_and_wait("deploy")


# --- gcp mode ------------------------------------------------------------------------------------


def _execution(**overrides):
    values = {
        "name": "projects/proj/locations/us-central1/jobs/mle-training-prod/executions/e1",
        "task_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
        "log_uri": "https://console.cloud.google.com/logs",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gcp_submit_runs_job_with_run_id_override(gcp_mode):
    jobs = MagicMock()
    jobs.run_job.return_value.result.return_value = _execution()
    with patch.object(utils, "_jobs_client", return_value=jobs):
        execution = utils._submit_and_wait("train", "run-42")

    request = jobs.run_job.call_args.kwargs["request"]
    assert request.name == "projects/proj/locations/us-central1/jobs/mle-training-prod"
    env = request.overrides.container_overrides[0].env[0]
    assert (env.name, env.value) == ("RUN_ID", "run-42")
    jobs.run_job.return_value.result.assert_called_once_with(timeout=900)
    assert execution.name.endswith("/executions/e1")


def test_gcp_submit_detects_failed_execution(gcp_mode):
    jobs = MagicMock()
    jobs.run_job.return_value.result.return_value = _execution(succeeded_count=0, failed_count=1)
    with (
        patch.object(utils, "_jobs_client", return_value=jobs),
        pytest.raises(utils.PipelineFailedError, match="failed"),
    ):
        utils._submit_and_wait("predict")
    assert "mle-serving-prod" in jobs.run_job.call_args.kwargs["request"].name


def test_gcp_query_is_parameterised(gcp_mode):
    client = MagicMock()
    client.query_and_wait.return_value = [
        {
            "total": 42,
            "page_rows": [{"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 7, "predicted_delay": 1}],
        }
    ]
    with patch.object(utils, "_bigquery_client", return_value=client):
        result = utils._query_predictions(3, 20, ["Grupo LATAM"], ["I"], [7])

    assert result["total_predictions"] == 42
    assert result["predictions"][0]["predicted_delay"] == 1
    sql = client.query_and_wait.call_args.args[0]
    assert "`proj.ds.predictions`" in sql
    assert "Grupo LATAM" not in sql
    params = {
        p.name: p for p in client.query_and_wait.call_args.kwargs["job_config"].query_parameters
    }
    assert params["opera"].values == ["Grupo LATAM"]
    assert params["mes"].values == [7]
    assert (params["limit"].value, params["offset"].value) == (20, 40)


def test_gcp_query_on_missing_table_is_empty(gcp_mode):
    client = MagicMock()
    client.query_and_wait.side_effect = NotFound("no table")
    with patch.object(utils, "_bigquery_client", return_value=client):
        assert utils._query_predictions(1, 10) == {"total_predictions": 0, "predictions": []}


def test_gcp_raw_flights_uses_table_metadata(gcp_mode):
    client = MagicMock()
    client.get_table.return_value.num_rows = 68206
    with patch.object(utils, "_bigquery_client", return_value=client):
        assert utils._check_raw_flights() == 68206
        client.get_table.side_effect = NotFound("missing")
        assert utils._check_raw_flights() == 0
    client.query.assert_not_called()


def test_gcp_pipeline_running_inspects_executions(gcp_mode):
    executions = MagicMock()
    executions.list_executions.return_value = [SimpleNamespace(completion_time=None)]
    with patch.object(utils, "_executions_client", return_value=executions):
        assert utils._pipeline_running("train") is True
        executions.list_executions.side_effect = RuntimeError("api down")
        assert utils._pipeline_running("train") is False


def test_read_metadata_never_raises(gcp_mode):
    with patch.object(utils, "_artifact_store", side_effect=RuntimeError("boom")):
        assert utils._read_model_metadata("run-1") == {}
        assert utils._latest_model_version() is None


# --- API behaviours ------------------------------------------------------------------------------

client = TestClient(app)


def test_train_returns_metadata_of_the_run():
    metadata = {
        "model_display_name": "delay-classifier",
        "model_version": "run-9",
        "trained_at": "2026-10-02T20:00:00+00:00",
        "metrics": {"recall_1": 0.69, "f1_1": 0.36, "tp": 2865},
    }
    with (
        patch.object(api_module, "_check_raw_flights", return_value=68206),
        patch.object(api_module, "_submit_and_wait", return_value=SimpleNamespace(name="exec-1")),
        patch.object(api_module, "_read_model_metadata", return_value=metadata) as read_metadata,
    ):
        response = client.post("/pipeline/train")
    assert response.status_code == 200
    assert response.json() == {**metadata, "pipeline_job_id": "exec-1"}
    run_id = read_metadata.call_args.args[0]
    assert run_id


def test_train_timeout_maps_to_504():
    with (
        patch.object(api_module, "_check_raw_flights", return_value=1),
        patch.object(api_module, "_submit_and_wait", side_effect=TimeoutError("too slow")),
    ):
        response = client.post("/pipeline/train")
    assert response.status_code == 504
    assert response.json()["detail"] == {"details": "too slow"}


def test_concurrent_runs_are_rejected_with_409():
    lock = api_module._PIPELINE_LOCKS["train"]
    lock.acquire()
    try:
        with patch.object(api_module, "_check_raw_flights", return_value=1):
            response = client.post("/pipeline/train")
    finally:
        lock.release()
    assert response.status_code == 409


def test_run_in_another_instance_is_rejected_with_409():
    with (
        patch.object(api_module, "_check_model_exists", return_value=True),
        patch.object(api_module, "_pipeline_running", return_value=True),
        patch.object(api_module, "_submit_and_wait") as submit,
    ):
        response = client.post("/pipeline/predict")
    assert response.status_code == 409
    submit.assert_not_called()
    assert not api_module._PIPELINE_LOCKS["predict"].locked()


def test_predict_invalidates_cache_and_returns_first_page():
    with (
        patch.object(api_module, "_check_model_exists", return_value=True),
        patch.object(api_module, "_submit_and_wait", return_value=SimpleNamespace(name="exec-2")),
        patch.object(api_module, "invalidate_predictions_cache") as invalidate,
        patch.object(api_module, "_latest_model_version", return_value="run-9"),
        patch.object(
            api_module,
            "_query_predictions",
            return_value={"total_predictions": 0, "predictions": []},
        ) as query,
    ):
        response = client.post("/pipeline/predict")
    assert response.status_code == 200
    assert response.json()["model_version"] == "run-9"
    invalidate.assert_called_once()
    query.assert_called_once_with(1, 10, None, None, None)


@pytest.mark.parametrize(
    "query",
    ["mes=0", "mes=13", "mes=7,x", "tipovuelo=X", "page=0", "page_size=101"],
)
def test_results_rejects_invalid_parameters(query):
    with patch.object(api_module, "_query_predictions") as query_predictions:
        response = client.get(f"/pipeline/predict/results?{query}")
    assert response.status_code == 422
    query_predictions.assert_not_called()


def test_results_normalises_filters_and_reports_total_pages():
    with patch.object(
        api_module,
        "_query_predictions",
        return_value={"total_predictions": 21, "predictions": []},
    ) as query:
        response = client.get(
            "/pipeline/predict/results?page_size=10&tipovuelo=i,%20n&opera=Copa%20Air,&mes=7"
        )
    assert response.status_code == 200
    assert response.json()["total_pages"] == 3
    query.assert_called_once_with(1, 10, ["Copa Air"], ["I", "N"], [7])
