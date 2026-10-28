import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export_model import export_catboost_model, load_metadata  # noqa: E402
from models import catboost_model  # noqa: E402


def _toy_data(n=40, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"temp_min": rng.uniform(-5, 15, n), "humidity_pct": rng.uniform(40, 100, n)})
    y = X["temp_min"] - 0.02 * X["humidity_pct"] + rng.normal(0, 0.1, n)
    return X, y


def test_export_round_trip_predictions_match(tmp_path):
    pytest.importorskip("catboost")
    from catboost import CatBoostRegressor

    X, y = _toy_data()
    model = catboost_model(n_estimators=20).fit(X, y)
    original_preds = model.predict(X)

    paths = export_catboost_model(
        model,
        feature_columns=["temp_min", "humidity_pct"],
        output_dir=str(tmp_path),
        version="test_v1",
        classification_threshold=3.0,
        label_variant="frost_label_broad",
        training_description="toy data for a unit test",
        eval_metrics={"mae": 1.23, "peirce_skill": 0.8},
    )

    assert os.path.exists(paths["model_path"])
    assert os.path.exists(paths["metadata_path"])

    reloaded = CatBoostRegressor()
    reloaded.load_model(paths["model_path"])
    reloaded_preds = reloaded.predict(X[["temp_min", "humidity_pct"]])

    # Same model, same inputs -> identical predictions after a round trip.
    np.testing.assert_allclose(original_preds, reloaded_preds, rtol=1e-6)


def test_export_metadata_contains_what_repo_b_needs(tmp_path):
    pytest.importorskip("catboost")
    X, y = _toy_data()
    model = catboost_model(n_estimators=10).fit(X, y)

    paths = export_catboost_model(
        model,
        feature_columns=["temp_min", "humidity_pct"],
        output_dir=str(tmp_path),
        version="test_v2",
        classification_threshold=3.0,
        label_variant="frost_label_broad",
        training_description="toy data",
        eval_metrics={"mae": 1.0},
    )

    meta = load_metadata(paths["metadata_path"])
    assert meta["feature_columns_in_order"] == ["temp_min", "humidity_pct"]
    assert meta["classification_threshold_c"] == 3.0
    assert "3.0" in meta["classification_rule"]
    assert meta["version"] == "test_v2"
    assert len(meta["known_limitations"]) >= 1


def test_export_accepts_raw_estimator_not_just_wrapper(tmp_path):
    pytest.importorskip("catboost")
    from catboost import CatBoostRegressor

    X, y = _toy_data()
    raw_estimator = CatBoostRegressor(n_estimators=10, verbose=False).fit(X, y)

    # Passing the raw estimator directly (no TreeModelWrapper) should still work.
    paths = export_catboost_model(
        raw_estimator,
        feature_columns=["temp_min", "humidity_pct"],
        output_dir=str(tmp_path),
        version="test_v3",
        classification_threshold=3.0,
        label_variant="frost_label_broad",
        training_description="toy data",
        eval_metrics={},
    )
    assert os.path.exists(paths["model_path"])


def test_logreg_json_round_trip_matches_sklearn(tmp_path):
    import json

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from export_model import export_logreg_json, predict_logreg_json

    X, y = _toy_data(n=200)
    yb = (y < y.median()).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(X, yb)
    path = export_logreg_json(pipe, list(X.columns), str(tmp_path / "lr.json"), 0.4, {"version": "t"})
    payload = json.load(open(path))
    np.testing.assert_allclose(predict_logreg_json(payload, X), pipe.predict_proba(X)[:, 1], rtol=1e-9)
    assert payload["probability_threshold"] == 0.4 and payload["feature_columns_in_order"] == list(X.columns)
