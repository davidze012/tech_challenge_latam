"""Extensions of the connectors beyond the provided suite: metadata, guards, error mapping."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from google.api_core.exceptions import NotFound

from challenge.config import clear_settings_cache
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.gcs import GCSArtifactStore, GCSCSVClient
from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import (
    MetadataStore,
    build_predictions_frame,
    combination_key,
    validate_run_id,
)
from challenge.model import DelayModel

# --- protocols helpers ------------------------------------------------------------------------


@pytest.mark.parametrize("run_id", ["20261002T101500Z-1a2b3c4d", "run-001", "v1.2_rc"])
def test_validate_run_id_accepts_safe_ids(run_id):
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["", "../etc", "a/b", "-leading-dash", "x" * 129, None])
def test_validate_run_id_rejects_unsafe_ids(run_id):
    with pytest.raises(ValueError, match="Invalid run_id"):
        validate_run_id(run_id)


def test_build_predictions_frame_layout_and_alignment():
    identifiers = pd.DataFrame(
        {
            "MES": ["7", "1"],
            "OPERA": ["Grupo LATAM", "Sky Airline"],
            "TIPOVUELO": ["I", "N"],
        },
        index=[10, 20],
    )
    frame = build_predictions_frame([1, 0], identifiers)
    assert list(frame.columns) == ["OPERA", "TIPOVUELO", "MES", "prediction"]
    assert frame["MES"].tolist() == [7, 1]
    assert frame.index.tolist() == [0, 1]


def test_build_predictions_frame_rejects_length_mismatch():
    identifiers = pd.DataFrame({"OPERA": ["a"], "TIPOVUELO": ["I"], "MES": [1]})
    with pytest.raises(ValueError, match="2 predictions for 1"):
        build_predictions_frame([0, 1], identifiers)


# --- local connectors -------------------------------------------------------------------------


def test_local_reader_requires_filename_when_several_csvs(tmp_path, monkeypatch):
    (tmp_path / "data.csv").write_text("a\n1\n")
    (tmp_path / "serving_input.csv").write_text("a\n2\n")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="pass filename"):
        LocalCSVClient().read()
    assert LocalCSVClient(filename="serving_input.csv").read()["a"].tolist() == [2]


def test_local_reader_missing_explicit_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="CSV not found"):
        LocalCSVClient(filename="nope.csv").read()


def test_local_store_metadata_roundtrip_and_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    store = LocalArtifactStore()
    assert isinstance(store, MetadataStore)
    assert store.latest_run_id() is None
    assert store.load_metadata("run-1") is None

    store.save_metadata("run-1", {"model_version": "run-1", "metrics": {"recall_1": 0.69}})
    assert store.load_metadata("run-1")["metrics"]["recall_1"] == 0.69


def test_local_store_load_missing_run_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        LocalArtifactStore().load(DelayModel(), "run-missing")


# --- GCS connectors -----------------------------------------------------------------------------


def _patched_bucket():
    bucket = MagicMock()
    client = MagicMock()
    client.bucket.return_value = bucket
    return patch("google.cloud.storage.Client", return_value=client), bucket


@patch("challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="b")
def test_gcs_model_upload_is_immutable_and_pointer_moves_last(_):
    storage_patch, bucket = _patched_bucket()
    model = DelayModel()
    model._model = {"fake": "estimator"}
    with storage_patch:
        GCSArtifactStore().save(model, "run-1")
    names = [c.args[0] for c in bucket.blob.call_args_list]
    assert names == ["run-1/model.joblib", "latest.txt"]
    upload_kwargs = bucket.blob.return_value.upload_from_string.call_args_list[0].kwargs
    assert upload_kwargs["if_generation_match"] == 0


@patch("challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="b")
def test_gcs_load_maps_not_found_to_file_not_found(_):
    storage_patch, bucket = _patched_bucket()
    bucket.blob.return_value.download_as_bytes.side_effect = NotFound("gone")
    with (
        storage_patch,
        pytest.raises(FileNotFoundError, match="Model artifact not found"),
    ):
        GCSArtifactStore().load(DelayModel(), "run-1")


@patch("challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="b")
def test_gcs_metadata_roundtrip(_):
    storage_patch, bucket = _patched_bucket()
    blob = bucket.blob.return_value
    blob.download_as_text.return_value = json.dumps({"model_version": "run-1"})
    with storage_patch:
        store = GCSArtifactStore()
        assert store.save_metadata("run-1", {"a": 1}) == "gs://b/run-1/metadata.json"
        assert store.load_metadata("run-1") == {"model_version": "run-1"}
        blob.download_as_text.side_effect = NotFound("gone")
        assert store.load_metadata("run-1") is None


def test_gcs_bucket_names_are_required(monkeypatch):
    monkeypatch.delenv("GCP_BUCKET_ARTIFACTS", raising=False)
    monkeypatch.delenv("GCP_BUCKET_INPUT", raising=False)
    clear_settings_cache()
    with pytest.raises(ValueError, match="GCP_BUCKET_ARTIFACTS"):
        GCSArtifactStore().latest_run_id()
    with pytest.raises(ValueError, match="GCP_BUCKET_INPUT"):
        GCSCSVClient(blob_name="serving_input.csv").read()
    clear_settings_cache()


def test_gcs_csv_client_reads_csv_from_input_bucket(monkeypatch):
    monkeypatch.setenv("GCP_BUCKET_INPUT", "input-bucket")
    clear_settings_cache()
    storage_patch, bucket = _patched_bucket()
    bucket.blob.return_value.download_as_bytes.return_value = (
        b"OPERA,TIPOVUELO,MES\nGrupo LATAM,I,7\n"
    )
    with storage_patch:
        data = GCSCSVClient(blob_name="serving_input.csv").read()
    bucket.blob.assert_called_once_with("serving_input.csv")
    assert data.to_dict("records") == [{"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 7}]
    clear_settings_cache()


# --- BigQuery connector ---------------------------------------------------------------------------


def test_bigquery_write_truncates_with_explicit_schema(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "p")
    monkeypatch.setenv("BQ_DATASET", "d")
    monkeypatch.setenv("BQ_PREDICTIONS_TABLE", "t")
    clear_settings_cache()
    client = MagicMock()
    identifiers = pd.DataFrame({"OPERA": ["Grupo LATAM"], "TIPOVUELO": ["I"], "MES": [1]})
    BigQueryClient(_client=client).write([1], identifiers)
    job_config = client.load_table_from_dataframe.call_args.kwargs["job_config"]
    assert job_config.write_disposition == "WRITE_TRUNCATE"
    assert [f.name for f in job_config.schema] == [
        "OPERA",
        "TIPOVUELO",
        "MES",
        "prediction",
    ]
    client.load_table_from_dataframe.return_value.result.assert_called_once()
    clear_settings_cache()


def test_bigquery_rejects_unsafe_identifiers(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "p")
    monkeypatch.setenv("BQ_DATASET", "d`; DROP TABLE x; --")
    clear_settings_cache()
    with pytest.raises(ValueError, match="Invalid BigQuery dataset"):
        BigQueryClient.table_id("t")
    clear_settings_cache()


def test_bigquery_client_is_created_lazily():
    with patch("challenge.connectors.bigquery.bigquery.Client") as client_cls:
        reader = BigQueryClient()
        client_cls.assert_not_called()
        assert reader.client is client_cls.return_value
        assert reader.client is client_cls.return_value
    client_cls.assert_called_once()


# --- Serving summary (model lineage + probabilities) ------------------------------------------


def test_local_store_serving_summary_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    store = LocalArtifactStore()
    assert store.load_serving_summary() is None
    path = store.save_serving_summary({"model_version": "run-1", "scores": {"a|I|7": 0.6}})
    assert path == str(tmp_path / "serving" / "latest.json")
    assert store.load_serving_summary()["scores"] == {"a|I|7": 0.6}


@patch("challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="b")
def test_gcs_store_serving_summary_roundtrip(_):
    storage_patch, bucket = _patched_bucket()
    blob = bucket.blob.return_value
    with storage_patch:
        store = GCSArtifactStore()
        assert (
            store.save_serving_summary({"model_version": "run-1"}) == "gs://b/serving/latest.json"
        )
        bucket.blob.assert_called_with("serving/latest.json")
        blob.download_as_text.return_value = json.dumps({"model_version": "run-1"})
        assert store.load_serving_summary() == {"model_version": "run-1"}
        blob.download_as_text.side_effect = NotFound("not yet")
        assert store.load_serving_summary() is None


def test_combination_key_is_stable_across_dtypes():
    assert combination_key("Grupo LATAM", "I", 7) == "Grupo LATAM|I|7"
    assert combination_key("Grupo LATAM", "I", 7.0) == "Grupo LATAM|I|7"
    assert combination_key("Grupo LATAM", "I", pd.NA) == "Grupo LATAM|I|"
