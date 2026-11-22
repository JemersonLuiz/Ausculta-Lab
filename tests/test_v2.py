"""Testes da versão 2.0: pré-processamento pulmonar, mapa de contraste, arquiteturas e front-end offline."""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from cardio_audio.audio import (
    CLASS_NAMES,
    FEATURE_CONFIG,
    contrast_png,
    decode_audio,
    extract_features,
    spectrogram_png,
)
from cardio_audio.model import (
    EnsembleRespiraNet,
    ExtendedCMNStats,
    RespiraDilatedNet,
    RespiraSEBandNet,
    SpectralGatedMLP,
    build_model,
)
from cardio_audio.predictor import Predictor


def test_v2_feature_config_and_bandpass_attenuates_sub_80hz_rumble():
    assert FEATURE_CONFIG["version"] == "logmel-bp-v3"
    assert FEATURE_CONFIG["fmin_hz"] == 80.0
    assert FEATURE_CONFIG["fmax_hz"] == 1200.0
    sr = 4000
    t = np.arange(sr * 2) / sr
    # Sinal pulmonar a 400 Hz + ruído cardíaco forte a 30 Hz
    lung = (0.15 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    with_heart = (lung + 0.80 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
    f_clean = extract_features(lung, sr)
    f_heart = extract_features(with_heart, sr)
    assert f_clean.shape == f_heart.shape == (64, 256)
    assert np.isfinite(f_heart).all()
    # A filtragem passa-faixa deve deixar o espectrograma altamente consistente mesmo com forte ruído em 30 Hz
    assert float(np.mean(np.abs(f_clean - f_heart))) < 0.05


def test_spectrogram_and_contrast_png_are_valid_images(wav_factory):
    audio = decode_audio(wav_factory(duration=1.2, sample_rate=8000, frequency=440.0))
    feats = extract_features(audio.samples, audio.sample_rate)
    for url in (spectrogram_png(feats), contrast_png(feats)):
        assert url.startswith("data:image/png;base64,")
        raw = base64.b64decode(url.split(",", 1)[1])
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("arch", ["respira_seband_v2", "respira_dilated_v2", "respira_stats_mlp_v2", "respira_ensemble_v2"])
def test_v2_architectures_are_deterministic_finite_and_safe_to_serialize(tmp_path, arch):
    torch.manual_seed(7)
    model = build_model(arch)
    model.eval()
    weights = tmp_path / "cnn.pt"
    torch.save(model.state_dict(), weights)
    meta = {
        "architecture_id": arch,
        "feature_config": FEATURE_CONFIG,
        "classes": CLASS_NAMES,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (tmp_path / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    predictor = Predictor(tmp_path)
    x = np.linspace(0.0, 1.0, 64 * 256, dtype=np.float32).reshape(64, 256)
    out1 = predictor.predict_features(x)
    out2 = predictor.predict_features(x)
    assert out1 == out2
    scores = [row["score"] for row in out1["scores"]]
    assert np.isfinite(scores).all()
    assert sum(scores) == pytest.approx(1.0, abs=1e-6)
    # Também deve suportar entrada silenciosa/constante sem NaN/Inf
    silent = predictor.predict_features(np.zeros((64, 256), dtype=np.float32))
    assert np.isfinite([row["score"] for row in silent["scores"]]).all()


def test_extended_cmn_stats_is_192d_and_invariant_to_frequency_eq():
    ext = ExtendedCMNStats()
    ext.eval()
    torch.manual_seed(11)
    base = torch.rand((2, 1, 64, 256), dtype=torch.float32) * 0.6 + 0.2
    # Simula resposta de frequência estacionária aditiva de outro estetoscópio (constante no tempo)
    eq_curve = torch.linspace(-0.25, 0.25, 64).view(1, 1, 64, 1)
    v1 = ext(base)
    v2 = ext(base + eq_curve)
    assert v1.shape == (2, 192)
    assert torch.isfinite(v1).all()
    # Os primeiros 128 + 32 (fcon/tdiff) + cooc são invariantes a deslocamento estacionário por frequência
    assert float((v1[:, :160] - v2[:, :160]).abs().max()) < 1e-4


def test_frontend_v2_is_offline_and_dom_ids_consistent():
    web = Path(__file__).resolve().parents[1] / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    css = (web / "style.css").read_text(encoding="utf-8")
    js = (web / "app.js").read_text(encoding="utf-8")
    for name, text in (("index.html", html), ("style.css", css), ("app.js", js)):
        urls = [u for u in re.findall(r"https?://[^\s'\"<>]+", text) if "w3.org/2000/svg" not in u]
        assert not urls, f"Recurso externo encontrado em {name}: {urls}"
    html_ids = set(re.findall(r'\bid="([^"]+)"', html))
    match = re.search(r"const el = Object\.fromEntries\(\[(.*?)\]\.map", js, re.S)
    assert match is not None
    js_ids = re.findall(r'"([^"]+)"', match.group(1))
    assert set(js_ids).issubset(html_ids)
