"""Verificação posterior do pacote e, opcionalmente, das previsões em toda a base."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi.testclient import TestClient

from cardio_audio.api import create_app
from cardio_audio.audio import decode_audio, extract_features, select_segment
from cardio_audio.predictor import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Reextrai os WAVs reais de teste e recalcula previsões e métricas (requer a base local).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "reports" / "qa" / ("reteste_completo.json" if args.full else "reteste_integracao.json")
    predictor = Predictor(ROOT / "models")
    app = create_app(project_root=ROOT)
    evidence = {"model_sha256": hashlib.sha256((ROOT / "models" / "cnn.pt").read_bytes()).hexdigest(),
                "checks": [], "examples": [], "synthetic_data_used_for_metrics": False}
    with TestClient(app) as client:
        status = client.get("/api/status")
        assert status.status_code == 200 and status.json()["model_ready"] is True
        evidence["checks"].append("modelo real carregado e metadados disponíveis")
        rows = client.get("/api/examples").json()
        assert len(rows) >= 1, "Não há exemplos reais de teste distribuídos."
        for row in rows:
            audio_response = client.get(f"/api/examples/{row['id']}/audio")
            assert audio_response.status_code == 200
            payload = audio_response.content
            audio = decode_audio(payload)
            selected = select_segment(audio, row["start"], row["end"])
            expected = predictor.predict_features(extract_features(selected, audio.sample_rate))
            request = {"files": {"file": ("example.wav", payload, "audio/wav")}, "data": {"start": row["start"], "end": row["end"]}}
            first = client.post("/api/analyze", **request)
            second = client.post("/api/analyze", **request)
            assert first.status_code == second.status_code == 200, (first.text, second.text)
            result = first.json()
            assert result["prediction"] == second.json()["prediction"] == expected
            assert result["spectrogram_png"].startswith("data:image/png;base64,")
            evidence["examples"].append({"id": row["id"], "annotated_label": row["label"],
                                         "predicted_label": result["prediction"]["label"],
                                         "repeated_output_identical": True,
                                         "api_matches_direct_inference": True})
        evidence["checks"].append("inferência real repetida; API igual à inferência direta em todos os exemplos")
        bad = client.post("/api/analyze", files={"file": ("fake.wav", b"not-a-wave", "audio/wav")}, data={"start": 0, "end": 1})
        assert bad.status_code == 422 and "prediction" not in bad.json()
        evidence["checks"].append("arquivo inválido rejeitado sem previsão")
    if args.full:
        import pandas as pd
        from cardio_audio.metrics import classification_metrics
        from train import make_predictions
        manifest = ROOT / "data" / "manifest.csv"
        actual_manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert actual_manifest_sha == predictor.metadata["manifest_sha256"]
        fingerprint = hashlib.sha256(manifest.read_bytes() + json.dumps(predictor.metadata["feature_config"], sort_keys=True).encode()).hexdigest()
        cache = ROOT / "data" / "processed" / f"features_{fingerprint[:16]}.npz"
        frame = pd.read_csv(manifest, dtype={"patient_id": str})
        assert frame.groupby("patient_id").split.nunique().max() == 1
        with np.load(cache, allow_pickle=False) as stored:
            assert str(stored["fingerprint"].item()) == fingerprint
            features = stored["features"].astype(np.float32)
        indices = np.flatnonzero(frame.split.to_numpy() == "test")
        # Novo caminho de leitura: WAV original -> segmentos -> features, antes de inferir.
        import soundfile as sf
        from cardio_audio.audio import AudioData
        raw_features = np.empty((len(indices), 64, 256), dtype=np.float32)
        held_out = frame.iloc[indices].reset_index(drop=True)
        for count, (_, recording) in enumerate(held_out.groupby("audio_path", sort=False), 1):
            path = Path(recording.iloc[0].audio_path)
            if not path.is_absolute():
                path = ROOT / path
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            original = AudioData(samples.mean(axis=1), int(rate), samples.shape[1], len(samples) / rate)
            for index, row in recording.iterrows():
                segment = select_segment(original, float(row.start_s), float(row.end_s))
                raw_features[index] = extract_features(segment, original.sample_rate)
            if count % 100 == 0:
                print(f"reteste: {count} WAVs originais reprocessados", flush=True)
        # Limite absoluto de quatro epsilons float32 para arredondamento entre kernels BLAS.
        # Não tolera mudança de categoria nem diferença nas métricas salvas abaixo.
        tolerance = float(4 * np.finfo(np.float32).eps)
        assert np.isfinite(raw_features).all() and np.isfinite(features[indices]).all()
        evidence["max_abs_feature_difference"] = float(np.max(np.abs(raw_features - features[indices])))
        evidence["feature_absolute_tolerance"] = tolerance
        np.testing.assert_allclose(raw_features, features[indices], rtol=0, atol=tolerance, equal_nan=False)
        predictions = make_predictions(predictor.model, raw_features)
        expected_frame = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
        np.testing.assert_array_equal(predictions, expected_frame.cnn_prediction.to_numpy())
        metrics = classification_metrics(frame.iloc[indices].label, predictions)
        saved = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))["cnn"]["test"]
        for name in ("macro_f1", "balanced_accuracy", "accuracy", "icbhi_score"):
            assert abs(metrics[name] - saved[name]) < 1e-12
        evidence["full_test_cycles"] = len(indices)
        evidence["raw_wav_features_match_training_cache"] = True
        evidence["full_test_predictions_identical"] = True
        evidence["checks"].append("WAVs originais do teste reprocessados; features, previsões e métricas reproduzidos")
    evidence["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "examples": len(evidence["examples"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
