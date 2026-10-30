"""Model artifact safety and train/inference preprocessing parity."""

import hashlib
import json

import numpy as np
import pytest
import torch

from cardio_audio.audio import CLASS_NAMES, FEATURE_CONFIG, extract_features
from cardio_audio.model import SmallCNN
from cardio_audio.predictor import ModelUnavailableError, Predictor


@pytest.fixture
def artifact_factory(tmp_path):
    """Random test weights ONLY validate serialization, never accuracy."""
    def create():
        directory = tmp_path / "model"
        directory.mkdir(exist_ok=True)
        torch.manual_seed(17)
        model = SmallCNN()
        model.eval()
        torch.save(model.state_dict(), directory / "cnn.pt")
        metadata = {"feature_config": FEATURE_CONFIG, "classes": CLASS_NAMES,
                    "weights_sha256": hashlib.sha256((directory / "cnn.pt").read_bytes()).hexdigest()}
        (directory / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
        return directory, model, metadata
    return create


def test_loading_saved_model_preserves_probabilities(artifact_factory):
    directory, original, _ = artifact_factory()
    features = np.full((64, 256), 0.4, dtype=np.float32)
    with torch.inference_mode():
        expected = torch.softmax(original(torch.from_numpy(features)[None, None]), dim=1)[0].numpy()
    result = Predictor(directory).predict_features(features)
    np.testing.assert_allclose([row["score"] for row in result["scores"]], expected, rtol=1e-6, atol=1e-7)
    assert result["index"] == int(np.argmax(expected))
    assert result["label"] == CLASS_NAMES[result["index"]]


def test_predict_and_training_feature_path_are_identical(artifact_factory):
    directory, _, _ = artifact_factory()
    sample_rate = 8000
    time = np.arange(sample_rate) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    predictor = Predictor(directory)
    direct = predictor.predict_features(extract_features(audio, sample_rate))
    composed = predictor.predict(audio, sample_rate)
    assert direct == composed


def test_tampered_model_checksum_is_rejected(artifact_factory):
    directory, _, _ = artifact_factory()
    with (directory / "cnn.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ModelUnavailableError):
        Predictor(directory)


def test_mismatched_class_mapping_is_rejected(artifact_factory):
    directory, _, metadata = artifact_factory()
    metadata["classes"] = list(reversed(CLASS_NAMES))
    (directory / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        Predictor(directory)


def test_mismatched_preprocessing_is_rejected(artifact_factory):
    directory, _, metadata = artifact_factory()
    metadata["feature_config"] = {**FEATURE_CONFIG, "sample_rate": 8000}
    (directory / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        Predictor(directory)


@pytest.mark.parametrize("metadata", [[], "unexpected text", None, 3])
def test_wrong_metadata_json_type_is_unavailable_not_unhandled(tmp_path, metadata):
    (tmp_path / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        Predictor(tmp_path)


def test_bad_weights_with_valid_checksum_are_unavailable_not_unhandled(artifact_factory):
    directory, _, metadata = artifact_factory()
    invalid = b"not a pytorch model"
    (directory / "cnn.pt").write_bytes(invalid)
    metadata["weights_sha256"] = hashlib.sha256(invalid).hexdigest()
    (directory / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        Predictor(directory)
