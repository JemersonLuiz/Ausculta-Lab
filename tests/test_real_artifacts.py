"""Integration checks require genuine trained/reported artifacts, not test doubles."""

import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from cardio_audio.api import create_app
from cardio_audio.audio import decode_audio, select_segment
from cardio_audio.predictor import Predictor


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_model_hash_and_metadata_correspond_to_file(real_project):
    metadata = read_json(real_project / "models/model.json")
    assert metadata["weights_sha256"] == hashlib.sha256((real_project / "models/cnn.pt").read_bytes()).hexdigest()
    assert metadata["clinical_validation"] is False
    assert metadata["arrhythmia_available"] is False
    assert metadata["epochs_executed"] >= 1
    assert metadata["trained_at"]
    Predictor(real_project / "models")


def test_real_metrics_are_recomputable_from_reserved_predictions(real_project):
    metrics = read_json(real_project / "models/metrics.json")
    predictions = pd.read_csv(real_project / "reports/test_predictions.csv", dtype={"patient_id": str})
    assert set(predictions.split) == {"test"}
    assert len(predictions) > 0
    for name, column in [("cnn", "cnn_prediction"), ("random_forest", "random_forest_prediction")]:
        assert metrics[name]["test"]["accuracy"] == pytest.approx((predictions.label == predictions[column]).mean())
        assert metrics[name]["test"]["balanced_accuracy"] == pytest.approx(
            balanced_accuracy_score(predictions.label, predictions[column]))
        expected = f1_score(predictions.label, predictions[column], labels=list(range(4)),
                            average="macro", zero_division=0)
        assert metrics[name]["test"]["macro_f1"] == pytest.approx(expected)
        assert metrics[name]["test"]["confusion_matrix"] == confusion_matrix(
            predictions.label, predictions[column], labels=list(range(4))).tolist()
    assert metrics["selection"]["test_used_for_selection"] is False


def test_real_manifest_has_patient_and_content_disjoint_splits(real_project):
    manifest = pd.read_csv(real_project / "data/manifest.csv", dtype={"patient_id": str})
    predictions = pd.read_csv(real_project / "reports/test_predictions.csv", dtype={"patient_id": str})
    audit = read_json(real_project / "data/manifest.audit.json")
    assert set(manifest.split) == {"train", "validation", "test"}
    assert manifest.groupby("patient_id").split.nunique().max() == 1
    assert manifest.groupby("audio_sha256").split.nunique().max() == 1
    assert audit["raw_cycles"] - audit["excluded_cycle_count"] == len(manifest)
    assert sum(item["excluded_cycle_count"] for item in audit["official_overlap_train_exclusions"]) == audit["official_overlap_excluded_cycle_count"]
    excluded_recordings = {item["recording_id"] for item in audit["official_overlap_train_exclusions"]}
    assert not excluded_recordings & set(manifest.recording_id)
    overlap_patients = set(map(str, audit["official_overlap_patients"]))
    assert set(manifest.loc[manifest.patient_id.isin(overlap_patients), "split"]) == {"test"}
    held_out = manifest[manifest.split == "test"]
    assert len(predictions) == len(held_out)
    assert set(predictions.patient_id) == set(held_out.patient_id)
    keys = ["patient_id", "recording_id", "start_s", "end_s", "label"]
    pd.testing.assert_frame_equal(
        predictions[keys].sort_values(keys).reset_index(drop=True),
        held_out[keys].sort_values(keys).reset_index(drop=True), check_dtype=False)


def test_real_examples_come_only_from_the_test_partition(real_project):
    examples = read_json(real_project / "examples/index.json")
    manifest = pd.read_csv(real_project / "data/manifest.csv", dtype={"patient_id": str})
    assert len(examples) >= 1
    for item in examples:
        match = manifest[(manifest.patient_id == str(item["patient_id"])) &
                         (manifest.recording_id == item["recording_id"]) &
                         np.isclose(manifest.start_s, item["source_start_s"]) &
                         np.isclose(manifest.end_s, item["source_end_s"])]
        assert len(match) == 1
        assert match.iloc[0].split == "test"
        assert "não selecionado pelo acerto" in item["selection"]


def test_packaged_examples_reproduce_reported_predictions(real_project):
    predictor = Predictor(real_project / "models")
    predictions = pd.read_csv(real_project / "reports/test_predictions.csv")
    predictions = predictions.set_index(["recording_id", "start_s", "end_s"])
    assert predictions.index.is_unique
    for item in read_json(real_project / "examples/index.json"):
        key = (item["recording_id"], item["source_start_s"], item["source_end_s"])
        reported = predictions.loc[key]
        assert predictor.metadata["classes"][int(reported["label"])] == item["label"]
        audio = decode_audio((real_project / "examples" / item["file"]).read_bytes())
        samples = select_segment(audio, item["start"], item["end"])
        predicted = predictor.predict(samples, audio.sample_rate)
        assert predicted["index"] == reported["cnn_prediction"], item["file"]


def test_real_api_prediction_matches_loaded_model_and_is_repeatable(real_project):
    predictor = Predictor(real_project / "models")
    application = create_app(project_root=real_project)
    with TestClient(application) as client:
        status = client.get("/api/status").json()
        assert status["model_ready"] is True
        examples = client.get("/api/examples").json()
        assert examples
        for item in examples:
            wav_response = client.get(f"/api/examples/{item['id']}/audio")
            assert wav_response.status_code == 200
            payload = wav_response.content
            data = decode_audio(payload)
            selected = select_segment(data, float(item["start"]), float(item["end"]))
            expected = predictor.predict(selected, data.sample_rate)
            response = client.post("/api/analyze", files={"file": ("example.wav", payload, "audio/wav")},
                                   data={"start": item["start"], "end": item["end"]})
            assert response.status_code == 200
            result = response.json()
            assert result["prediction"] == expected
            scores = [row["score"] for row in result["prediction"]["scores"]]
            assert np.isfinite(scores).all()
            assert sum(scores) == pytest.approx(1, abs=1e-6)
            repeated = client.post("/api/analyze", files={"file": ("example.wav", payload, "audio/wav")},
                                   data={"start": item["start"], "end": item["end"]})
            assert repeated.json()["prediction"] == result["prediction"]
