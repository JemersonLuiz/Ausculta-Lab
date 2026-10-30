"""Preparação determinística compartilhada pelo treinamento e pela inferência."""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from scipy import signal
from scipy.fft import dct
from scipy.ndimage import zoom

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DURATION_S = 120.0
MIN_SEGMENT_S = 0.1
MAX_SEGMENT_S = 20.0
TARGET_RATE = 4000
FEATURE_CONFIG = {
    "version": "logmel-v1",
    "sample_rate": TARGET_RATE,
    "n_fft": 256,
    "hop_length": 64,
    "n_mels": 64,
    "time_bins": 256,
    "dynamic_range_db": 80.0,
    "temporal_policy": "interpolate_whole_cycle_no_crop",
}
CLASS_NAMES = ["Sem sibilos/estertores", "Estertores", "Sibilos", "Ambos"]


class AudioValidationError(ValueError):
    """Entrada não atende ao contrato de áudio."""


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration_s: float


def decode_audio(payload: bytes) -> AudioData:
    if not payload:
        raise AudioValidationError("O arquivo está vazio.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise AudioValidationError("O arquivo excede o limite de 20 MB.")
    if len(payload) < 12 or payload[:4] not in (b"RIFF", b"RIFX", b"RF64") or payload[8:12] != b"WAVE":
        raise AudioValidationError("Envie um arquivo WAV válido; renomear a extensão não converte o áudio.")
    try:
        with sf.SoundFile(io.BytesIO(payload)) as source:
            if source.format not in {"WAV", "WAVEX", "RF64"}:
                raise AudioValidationError("Formato não suportado. Utilize WAV.")
            rate, channels, frames = int(source.samplerate), int(source.channels), int(len(source))
            if not 2000 <= rate <= 192000:
                raise AudioValidationError("A taxa de amostragem deve estar entre 2.000 e 192.000 Hz.")
            if channels not in (1, 2):
                raise AudioValidationError("Utilize áudio mono ou estéreo.")
            duration = frames / rate
            if not frames or duration > MAX_DURATION_S:
                raise AudioValidationError("O áudio deve ter duração positiva de até 120 segundos.")
            samples = source.read(dtype="float32", always_2d=True)
    except AudioValidationError:
        raise
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise AudioValidationError("O WAV está corrompido ou não pôde ser decodificado.") from exc
    if samples.shape != (frames, channels) or not np.isfinite(samples).all():
        raise AudioValidationError("O áudio contém amostras inválidas ou dados incompletos.")
    if np.max(np.abs(samples)) > 16:
        raise AudioValidationError("A amplitude do WAV está fora do intervalo aceito.")
    mono = samples.mean(axis=1, dtype=np.float32)
    return AudioData(mono, rate, channels, duration)


def select_segment(audio: AudioData, start: float, end: float) -> np.ndarray:
    if not math.isfinite(start) or not math.isfinite(end):
        raise AudioValidationError("Início e fim devem ser números finitos.")
    if start < 0 or end <= start or end > audio.duration_s + 1e-6:
        raise AudioValidationError("Selecione um intervalo válido dentro da duração do áudio.")
    duration = end - start
    if duration < MIN_SEGMENT_S - 1e-9 or duration > MAX_SEGMENT_S + 1e-9:
        raise AudioValidationError("O trecho deve durar entre 0,1 e 20 segundos.")
    first, last = int(round(start * audio.sample_rate)), int(round(end * audio.sample_rate))
    selected = audio.samples[first:last]
    if len(selected) < 2 or not np.isfinite(selected).all():
        raise AudioValidationError("O trecho não possui amostras válidas suficientes.")
    if np.sqrt(np.mean(selected.astype(np.float64) ** 2)) < 1e-7:
        raise AudioValidationError("O trecho está silencioso ou sua amplitude é insuficiente para análise.")
    return selected


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    n_fft, n_mels = FEATURE_CONFIG["n_fft"], FEATURE_CONFIG["n_mels"]
    upper_mel = 2595 * np.log10(1 + (TARGET_RATE / 2) / 700)
    mel = np.linspace(0, upper_mel, n_mels + 2)
    hz = 700 * (10 ** (mel / 2595) - 1)
    frequencies = np.fft.rfftfreq(n_fft, 1 / TARGET_RATE)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz[index:index + 3]
        filters[index] = np.maximum(0, np.minimum((frequencies - left) / (center - left), (right - frequencies) / (right - center)))
    return filters


def extract_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Matriz 64x256 em [0,1], preservando o ciclo inteiro no eixo temporal."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise AudioValidationError("O trecho deve ser um vetor de amostras finitas.")
    if not isinstance(sample_rate, (int, np.integer)) or not 2000 <= sample_rate <= 192000:
        raise AudioValidationError("Taxa de amostragem inválida.")
    divisor = math.gcd(int(sample_rate), TARGET_RATE)
    if sample_rate != TARGET_RATE:
        values = signal.resample_poly(values, TARGET_RATE // divisor, int(sample_rate) // divisor).astype(np.float32)
    peak = float(np.max(np.abs(values)))
    if peak < 1e-8:
        raise AudioValidationError("O trecho não contém sinal mensurável.")
    values = values / peak
    n_fft = FEATURE_CONFIG["n_fft"]
    if len(values) < n_fft:
        values = np.pad(values, (0, n_fft - len(values)))
    _, _, transform = signal.stft(values, fs=TARGET_RATE, window="hann", nperseg=n_fft,
                                  noverlap=n_fft - FEATURE_CONFIG["hop_length"], boundary="zeros", padded=True)
    power = np.abs(transform).astype(np.float32) ** 2
    mel_power = mel_filterbank() @ power
    log_mel = 10 * np.log10(np.maximum(mel_power, np.finfo(np.float32).tiny))
    log_mel -= float(log_mel.max())
    log_mel = np.clip(log_mel, -80, 0)
    resized = zoom(log_mel, (1, FEATURE_CONFIG["time_bins"] / log_mel.shape[1]), order=1, prefilter=False)
    if resized.shape != (64, 256):
        raise RuntimeError("Dimensão interna inesperada no espectrograma.")
    return ((resized + 80) / 80).astype(np.float32)


def baseline_features(features: np.ndarray) -> np.ndarray:
    """Resumo cepstral: média/desvio de 20 coeficientes, sobre a mesma entrada."""
    cepstrum = dct(features * 80 - 80, type=2, axis=0, norm="ortho")[:20]
    return np.concatenate([cepstrum.mean(axis=1), cepstrum.std(axis=1)]).astype(np.float32)


def quality_warnings(samples: np.ndarray, sample_rate: int) -> list[str]:
    warnings = []
    if sample_rate != TARGET_RATE:
        warnings.append(f"Sinal reamostrado de {sample_rate:,} Hz para 4.000 Hz.")
    clipping = float(np.mean(np.abs(samples) >= 0.98))
    if clipping > 0.01:
        warnings.append("Possível saturação: mais de 1% das amostras próximas do limite de amplitude.")
    if len(samples) / sample_rate < 1:
        warnings.append("Trecho curto: verifique se contém um ciclo respiratório completo.")
    warnings.append("Seleção do ciclo é manual; qualidade clínica e origem do áudio não são verificadas automaticamente.")
    return warnings


def spectrogram_png(features: np.ndarray) -> str:
    values = np.clip(np.flipud(features), 0, 1)
    # Paleta sequencial local, sem bibliotecas gráficas ou recursos externos.
    stops = np.array([[12, 28, 41], [21, 78, 91], [29, 142, 142], [110, 202, 166], [246, 232, 153]], dtype=float)
    position = values * (len(stops) - 1)
    low = np.minimum(position.astype(int), len(stops) - 2)
    fraction = (position - low)[..., None]
    rgb = (stops[low] * (1 - fraction) + stops[low + 1] * fraction).astype(np.uint8)
    plot = Image.fromarray(rgb).resize((768, 192), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (832, 238), "#f4f8f7")
    canvas.paste(plot, (52, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 12), "2 kHz", fill="#24434b")
    draw.text((22, 190), "0", fill="#24434b")
    draw.text((52, 214), "Inicio do ciclo", fill="#24434b")
    draw.text((710, 214), "Fim do ciclo", fill="#24434b")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

