"""Test-only WAV fixtures; these are never research/training observations."""

from __future__ import annotations

from array import array
from io import BytesIO
import math
from pathlib import Path
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pytest_addoption(parser):
    parser.addoption("--require-real-artifacts", action="store_true", default=False,
                     help="Fail, rather than skip, if real trained model/evaluation artifacts are absent.")


@pytest.fixture
def real_project(request):
    paths = ["models/cnn.pt", "models/model.json", "models/metrics.json", "models/dataset.json",
             "data/manifest.csv", "data/manifest.audit.json", "reports/test_predictions.csv", "examples/index.json"]
    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing:
        detail = "Real experiment artifacts missing: " + ", ".join(missing)
        if request.config.getoption("--require-real-artifacts"):
            pytest.fail(detail)
        pytest.skip(detail)
    return ROOT


def make_pcm_wav(
    duration: float = 1.0,
    sample_rate: int = 8000,
    channels: int = 1,
    frequency: float = 220.0,
    amplitude: float = 0.2,
) -> bytes:
    """Produce a small PCM WAV for deterministic software behavior tests."""
    frames = int(duration * sample_rate)
    values = array("h")
    for frame in range(frames):
        sample = int(32767 * amplitude * math.sin(2 * math.pi * frequency * frame / sample_rate))
        for channel in range(channels):
            values.append(sample if channel == 0 else sample // 2)
    if sys.byteorder != "little":
        values.byteswap()
    buffer = BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(values.tobytes())
    return buffer.getvalue()


@pytest.fixture
def wav_factory():
    return make_pcm_wav


@pytest.fixture
def icbhi_fixture(tmp_path):
    """Small artificial file layout for testing parser/split logic only."""
    raw = tmp_path / "icbhi"
    raw.mkdir()
    for patient in range(101, 117):
        for recording in range(1, 3):
            stem = f"{patient}_{recording}b1_Al_sc_Meditron"
            content = make_pcm_wav(duration=4.0, sample_rate=4000,
                                   frequency=150 + patient + recording / 10)
            (raw / f"{stem}.wav").write_bytes(content)
            (raw / f"{stem}.txt").write_text(
                "0.0\t1.0\t0\t0\n1.0\t2.0\t1\t0\n"
                "2.0\t3.0\t0\t1\n3.0\t4.0\t1\t1\n",
                encoding="utf-8",
            )
    return raw
