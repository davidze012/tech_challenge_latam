"""Edge cases of DelayModel beyond the provided suite: dtypes, skew, labels, guards."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from challenge.model import FEATURES_COLS, DelayModel, compute_delay


def _flights(**overrides) -> pd.DataFrame:
    base = {
        "Fecha-I": [
            "2017-01-01 10:00:00",
            "2017-07-01 10:00:00",
            "2017-12-01 10:00:00",
        ],
        "Fecha-O": [
            "2017-01-01 10:15:00",
            "2017-07-01 10:16:00",
            "2017-12-01 09:50:00",
        ],
        "OPERA": ["Grupo LATAM", "Sky Airline", "Copa Air"],
        "TIPOVUELO": ["I", "N", "I"],
        "MES": [1, 7, 12],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# --- target ---------------------------------------------------------------------------------


def test_delay_threshold_is_strictly_greater_than_15_minutes():
    """15 min late is on time; 16 min late is delayed; early is on time."""
    assert compute_delay(_flights()).tolist() == [0, 1, 0]


def test_delay_accepts_datetime_dtypes_from_bigquery():
    """BigQuery DATETIME columns arrive as datetime64 and must be used as-is."""
    data = _flights()
    data["Fecha-I"] = pd.to_datetime(data["Fecha-I"])
    data["Fecha-O"] = pd.to_datetime(data["Fecha-O"])
    assert compute_delay(data).tolist() == [0, 1, 0]


def test_existing_target_column_is_used_instead_of_recomputed():
    data = _flights(delay=[1, 0, 1])
    _, target = DelayModel().preprocess(data, target_column="delay")
    assert target["delay"].tolist() == [1, 0, 1]


def test_rows_with_unparseable_dates_are_dropped_from_features_and_target():
    data = _flights(**{"Fecha-O": ["2017-01-01 10:15:00", "not-a-date", None]})
    features, target = DelayModel().preprocess(data, target_column="delay")
    assert len(features) == len(target) == 1
    assert features.index.equals(target.index)


def test_unknown_target_column_raises():
    with pytest.raises(ValueError, match="not in the data"):
        DelayModel().preprocess(_flights(), target_column="cancelled")


# --- features ---------------------------------------------------------------------------------


def test_serving_input_without_dates_produces_canonical_features():
    """serving_input.csv only has OPERA/TIPOVUELO/MES; output must keep every row."""
    data = pd.DataFrame(
        {
            "OPERA": ["Grupo LATAM", "Aerolínea Nueva"],
            "TIPOVUELO": ["I", "N"],
            "MES": [7, 3],
        }
    )
    features = DelayModel().preprocess(data)
    assert list(features.columns) == FEATURES_COLS
    assert len(features) == 2
    # Unknown airline and non-top-10 month → no active OPERA/MES feature.
    assert features.iloc[1].sum() == 0
    assert features.loc[0, ["OPERA_Grupo LATAM", "MES_7", "TIPOVUELO_I"]].tolist() == [
        1,
        1,
        1,
    ]


def test_features_tolerate_bigquery_nullable_ints_and_messy_strings():
    data = pd.DataFrame(
        {
            "OPERA": [" Grupo LATAM ", None],
            "TIPOVUELO": ["i", "N"],
            "MES": pd.array([7, None], dtype="Int64"),
        }
    )
    features = DelayModel().preprocess(data)
    assert features.loc[0, ["OPERA_Grupo LATAM", "TIPOVUELO_I", "MES_7"]].tolist() == [
        1,
        1,
        1,
    ]
    assert features.iloc[1].sum() == 0
    assert (features.dtypes == "int64").all()


def test_preprocess_does_not_mutate_its_input():
    data = _flights()
    snapshot = data.copy()
    DelayModel().preprocess(data, target_column="delay")
    pd.testing.assert_frame_equal(data, snapshot)


def test_missing_required_columns_raise_a_clear_error():
    with pytest.raises(ValueError, match="missing required columns"):
        DelayModel().preprocess(pd.DataFrame({"OPERA": ["Grupo LATAM"]}))


# --- fit / predict ------------------------------------------------------------------------------


def _training_set() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 40
    data = pd.DataFrame(
        {
            "OPERA": ["Grupo LATAM", "Sky Airline"] * (rows // 2),
            "TIPOVUELO": ["I", "N"] * (rows // 2),
            "MES": [7, 1] * (rows // 2),
            "delay": [1, 0] * (rows // 2),
        }
    )
    return DelayModel().preprocess(data, target_column="delay")


def test_fit_sets_scale_pos_weight_from_class_ratio():
    features, target = _training_set()
    target.iloc[:10] = 0  # 25 negatives / 15 positives
    model = DelayModel()
    model.fit(features, target)
    assert model.is_fitted
    assert model._model.get_params()["scale_pos_weight"] == pytest.approx(25 / 15)


def test_fit_rejects_a_single_class_target():
    features, target = _training_set()
    with pytest.raises(ValueError, match="both classes"):
        DelayModel().fit(features, target * 0)


def test_predict_returns_python_ints_for_a_fitted_model():
    features, target = _training_set()
    model = DelayModel()
    model.fit(features, target)
    predictions = model.predict(features[list(reversed(FEATURES_COLS))])  # any column order
    assert len(predictions) == len(features)
    assert all(type(p) is int and p in (0, 1) for p in predictions)


def test_predict_rejects_features_without_the_contract_columns():
    features, target = _training_set()
    model = DelayModel()
    model.fit(features, target)
    with pytest.raises(ValueError, match="missing columns"):
        model.predict(features.drop(columns=["MES_7"]))


def test_untrained_predict_returns_zeros():
    assert DelayModel().predict(pd.DataFrame({c: [0, 1] for c in FEATURES_COLS})) == [
        0,
        0,
    ]


# --- mixin extensions -----------------------------------------------------------------------------


def test_write_predictions_delegates_to_reader():
    reader = MagicMock()
    identifiers = pd.DataFrame({"OPERA": ["Grupo LATAM"], "TIPOVUELO": ["I"], "MES": [1]})
    DelayModel(reader=reader).write_predictions([1], identifiers)
    reader.write.assert_called_once_with([1], identifiers)


def test_save_metadata_requires_a_metadata_capable_store():
    store = MagicMock(spec=["save", "load"])  # plain ArtifactStore
    with pytest.raises(TypeError, match="does not support run metadata"):
        DelayModel(store=store).save_metadata("run-1", {})
