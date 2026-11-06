"""Carregamento local de pesos, validado por hash e sem pickle arbitrário."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .audio import CLASS_NAMES, FEATURE_CONFIG, extract_features
from .model import build_model


class ModelUnavailableError(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        try:
            self.metadata = json.loads((self.model_dir / "model.json").read_text(encoding="utf-8"))
            if not isinstance(self.metadata, dict):
                raise ValueError("Metadados devem ser um objeto JSON.")
            if self.metadata.get("feature_config") != FEATURE_CONFIG:
                raise ValueError("Configuração de áudio incompatível.")
            if self.metadata.get("classes") != CLASS_NAMES:
                raise ValueError("Categorias incompatíveis.")
            weights = self.model_dir / "cnn.pt"
            actual = hashlib.sha256(weights.read_bytes()).hexdigest()
            if actual != self.metadata.get("weights_sha256"):
                raise ValueError("O checksum do modelo não confere.")
            self.model = build_model(self.metadata.get("architecture_id", "small_cnn_v1"))
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            self.model.eval()
            if any(not torch.isfinite(value).all() for value in self.model.state_dict().values() if value.is_floating_point()):
                raise ValueError("Parâmetros inválidos no modelo.")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, EOFError, pickle.UnpicklingError) as exc:
            raise ModelUnavailableError("Modelo treinado ausente, inválido ou incompatível. Consulte o treinamento documentado.") from exc

    def predict_features(self, features: np.ndarray) -> dict:
        with torch.inference_mode():
            tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
            logits = self.model(tensor)
            scores = torch.softmax(logits, dim=1).cpu().numpy()[0]
        if not np.isfinite(scores).all():
            raise ModelUnavailableError("O modelo produziu uma saída inválida.")
        index = int(np.argmax(scores))
        return {"label": CLASS_NAMES[index], "index": index,
                "scores": [{"label": name, "score": float(score)} for name, score in zip(CLASS_NAMES, scores, strict=True)]}

    def predict(self, samples: np.ndarray, sample_rate: int) -> dict:
        return self.predict_features(extract_features(samples, sample_rate))

