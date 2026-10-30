"""Black-box local API tests: file content, boundaries, model state and paths."""

import json
import base64

from fastapi.testclient import TestClient
import numpy as np
import pytest

from cardio_audio.api import create_app
from cardio_audio.audio import CLASS_NAMES


@pytest.fixture
def isolated_app(tmp_path):
    return create_app(project_root=tmp_path, model_dir=tmp_path / "models")


@pytest.fixture
def client(isolated_app):
    with TestClient(isolated_app) as test_client:
        yield test_client


def upload(client, payload, start="0", end="1", filename="test.wav", content_type="audio/wav"):
    return client.post("/api/analyze", files={"file": (filename, payload, content_type)},
                       data={"start": start, "end": end})


def test_status_has_no_fake_model_or_metrics_without_artifacts(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_ready"] is False
    assert data["model"]["metrics"] == {}
    assert len(data["model"]["classes"]) == 4
    assert data["limits"]["max_upload_mb"] == 20


def test_valid_audio_cannot_produce_prediction_without_model(client, wav_factory):
    response = upload(client, wav_factory())
    assert response.status_code == 503
    assert isinstance(response.json()["detail"], str)
    assert "prediction" not in response.json()


@pytest.mark.parametrize("payload", [b"", b"hello", b"RIFF0000WAVEgarbage", b"%PDF-1.7 fake"])
def test_filename_and_content_type_cannot_disguise_invalid_audio(client, payload):
    response = upload(client, payload, filename="apparently_valid.wav", content_type="audio/wav")
    assert 400 <= response.status_code < 500
    assert "prediction" not in response.json()
    assert "Traceback" not in response.text


@pytest.mark.parametrize("start,end", [("nan", "1"), ("0", "inf"), ("-1", "1"),
                                       ("1", "0"), ("0", "2"), ("0", "0.01")])
def test_invalid_ranges_are_user_errors(client, wav_factory, start, end):
    response = upload(client, wav_factory(), start=start, end=end)
    assert 400 <= response.status_code < 500
    assert response.status_code != 503
    assert "prediction" not in response.json()


def test_oversized_upload_is_rejected(client):
    response = upload(client, b"x" * (20 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_missing_required_file_is_rejected(client):
    response = client.post("/api/analyze", data={"start": "0", "end": "1"})
    assert response.status_code == 422


def test_examples_do_not_expose_filesystem_paths(client, tmp_path):
    directory = tmp_path / "examples"
    directory.mkdir()
    (directory / "index.json").write_text(json.dumps([{
        "id": "one", "name": "Example", "label": "Sibilos", "start": 0, "end": 1,
        "file": "secret.wav", "internal_path": "C:/private/something"
    }]), encoding="utf-8")
    response = client.get("/api/examples")
    assert response.status_code == 200
    assert set(response.json()[0]) == {"id", "name", "label", "start", "end"}


def test_example_file_must_stay_under_examples_directory(client, tmp_path, wav_factory):
    directory = tmp_path / "examples"
    directory.mkdir()
    (tmp_path / "outside.wav").write_bytes(wav_factory())
    (directory / "index.json").write_text(json.dumps([{
        "id": "escape", "file": "../outside.wav", "name": "Invalid path"
    }]), encoding="utf-8")
    response = client.get("/api/examples/escape/audio")
    assert response.status_code == 404


@pytest.mark.parametrize("example_id", ["unknown", "%2e%2e%2fREADME.md", "%2e%2e%5cREADME.md"])
def test_unknown_or_traversal_example_ids_are_rejected(client, example_id):
    response = client.get(f"/api/examples/{example_id}/audio")
    assert response.status_code == 404


def test_cross_origin_access_is_rejected(client):
    response = client.get("/api/status", headers={"Origin": "https://unrelated.example"})
    assert response.status_code == 403


def test_unknown_host_is_rejected(client):
    response = client.get("/api/status", headers={"Host": "untrusted.example"})
    assert response.status_code == 400


def test_security_headers_are_present(client):
    response = client.get("/api/status")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_success_uses_decoded_audio_features_and_preserves_contract(isolated_app, wav_factory, tmp_path):
    class StubPredictor:
        """Test double for HTTP contract, not a trained/scientific model."""
        metadata = {"test_double": True}
        received = None

        def predict_features(self, features):
            self.received = features
            return {"label": CLASS_NAMES[1], "index": 1,
                    "scores": [{"label": label, "score": float(index == 1)}
                               for index, label in enumerate(CLASS_NAMES)]}

    stub = StubPredictor()
    isolated_app.state.predictor = stub
    with TestClient(isolated_app) as success_client:
        response = upload(success_client, wav_factory(channels=2), start="0.2", end="0.8",
                          filename="../../escape.wav", content_type="application/octet-stream")
    assert response.status_code == 200
    body = response.json()
    assert body["audio"]["channels"] == 2
    assert body["segment"]["duration_s"] == pytest.approx(0.6)
    assert body["prediction"]["index"] == 1
    assert len(body["prediction"]["scores"]) == 4
    assert len(body["limitations"]) > 0
    assert stub.received.shape == (64, 256)
    assert np.isfinite(stub.received).all()
    image = body["spectrogram_png"]
    assert image.startswith("data:image/png;base64,")
    assert base64.b64decode(image.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.rglob("*.wav")), "The untrusted upload was persisted to disk."
