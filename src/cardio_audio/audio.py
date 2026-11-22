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
from scipy.ndimage import uniform_filter1d, zoom

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DURATION_S = 120.0
MIN_SEGMENT_S = 0.1
MAX_SEGMENT_S = 20.0
TARGET_RATE = 4000
FEATURE_CONFIG = {
    "version": "logmel-bp-v3",
    "sample_rate": TARGET_RATE,
    "fmin_hz": 80.0,
    "fmax_hz": 1200.0,
    "bandpass_order": 4,
    "n_fft": 256,
    "hop_length": 32,
    "n_mels": 64,
    "time_bins": 256,
    "min_dynamic_span_db": 25.0,
    "low_percentile": 5.0,
    "reference_percentile": 99.5,
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
    if payload.startswith(b"OggS"):
        raise AudioValidationError("Arquivo OGG ou Opus. Para analisar, escolha uma gravação de ausculta em WAV.")
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
def _bandpass_sos() -> np.ndarray:
    return signal.butter(
        int(FEATURE_CONFIG["bandpass_order"]),
        [float(FEATURE_CONFIG["fmin_hz"]), float(FEATURE_CONFIG["fmax_hz"])],
        btype="bandpass",
        fs=TARGET_RATE,
        output="sos",
    )


def _apply_bandpass(values: np.ndarray) -> np.ndarray:
    sos = _bandpass_sos()
    padlen = 3 * (2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
    if len(values) <= padlen:
        return values.astype(np.float32)
    return signal.sosfiltfilt(sos, values).astype(np.float32)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    n_fft = int(FEATURE_CONFIG["n_fft"])
    n_mels = int(FEATURE_CONFIG["n_mels"])
    fmin = float(FEATURE_CONFIG["fmin_hz"])
    fmax = float(FEATURE_CONFIG["fmax_hz"])
    lower_mel = 2595.0 * np.log10(1.0 + fmin / 700.0)
    upper_mel = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel = np.linspace(lower_mel, upper_mel, n_mels + 2)
    hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / TARGET_RATE)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz[index:index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-6),
                (right - frequencies) / max(right - center, 1e-6),
            ),
        )
    # Normalização de área Slaney para evitar ênfase artificial nas bandas altas.
    enorm = 2.0 / np.maximum(hz[2:] - hz[:-2], 1e-6)
    filters *= enorm[:, None].astype(np.float32)
    return filters


def extract_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Matriz 64x256 em [0,1] filtrada na faixa respiratória (80-1800 Hz)."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise AudioValidationError("O trecho deve ser um vetor de amostras finitas.")
    if not isinstance(sample_rate, (int, np.integer)) or not 2000 <= sample_rate <= 192000:
        raise AudioValidationError("Taxa de amostragem inválida.")
    divisor = math.gcd(int(sample_rate), TARGET_RATE)
    if sample_rate != TARGET_RATE:
        values = signal.resample_poly(values, TARGET_RATE // divisor, int(sample_rate) // divisor).astype(np.float32)
    peak_raw = float(np.max(np.abs(values)))
    if peak_raw < 1e-8:
        raise AudioValidationError("O trecho não contém sinal mensurável.")
    # Filtro passa-faixa respiratório remove ruído sub-80 Hz (batimentos cardíacos/manuseio).
    filtered = _apply_bandpass(values)
    rms = float(np.sqrt(np.mean(filtered.astype(np.float64) ** 2)))
    if rms < 1e-8:
        # Fallback para sinais puros fora da banda principal (ex.: fixtures de teste).
        filtered = values
        rms = float(np.sqrt(np.mean(filtered.astype(np.float64) ** 2)))
    if rms < 1e-8:
        raise AudioValidationError("O trecho não contém sinal mensurável.")
    filtered = (filtered / rms).astype(np.float32)
    n_fft = int(FEATURE_CONFIG["n_fft"])
    if len(filtered) < n_fft:
        filtered = np.pad(filtered, (0, n_fft - len(filtered)))
    _, _, transform = signal.stft(
        filtered,
        fs=TARGET_RATE,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - int(FEATURE_CONFIG["hop_length"]),
        boundary="zeros",
        padded=True,
    )
    power = (transform.real ** 2 + transform.imag ** 2).astype(np.float32)
    mel_power = mel_filterbank() @ power
    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-12))
    # Normalização adaptativa por percentis harmoniza contraste entre estetoscópios com e sem ANR.
    low_ref = float(np.percentile(log_mel, float(FEATURE_CONFIG["low_percentile"])))
    high_ref = float(np.percentile(log_mel, float(FEATURE_CONFIG["reference_percentile"])))
    span = max(high_ref - low_ref, float(FEATURE_CONFIG["min_dynamic_span_db"]))
    normalized = np.clip((log_mel - (high_ref - span)) / span, 0.0, 1.0)
    resized = zoom(normalized, (1.0, int(FEATURE_CONFIG["time_bins"]) / normalized.shape[1]), order=1, prefilter=False)
    if resized.shape != (64, 256):
        raise RuntimeError("Dimensão interna inesperada no espectrograma.")
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def baseline_features(features: np.ndarray) -> np.ndarray:
    """Resumo cepstral: média/desvio de 20 coeficientes, sobre a mesma entrada."""
    cepstrum = dct(features * 65.0 - 60.0, type=2, axis=0, norm="ortho")[:20]
    return np.concatenate([cepstrum.mean(axis=1), cepstrum.std(axis=1)]).astype(np.float32)


def quality_warnings(samples: np.ndarray, sample_rate: int) -> list[str]:
    warnings = []
    if sample_rate != TARGET_RATE:
        warnings.append(f"Sinal reamostrado de {sample_rate:,} Hz para 4.000 Hz com filtro pulmonar (80–1.200 Hz).")
    clipping = float(np.mean(np.abs(samples) >= 0.98))
    if clipping > 0.01:
        warnings.append("Possível saturação: mais de 1% das amostras próximas do limite de amplitude.")
    if len(samples) / sample_rate < 1:
        warnings.append("Trecho curto: verifique se contém um ciclo respiratório completo.")
    warnings.append("Seleção do ciclo é manual; qualidade clínica e origem do áudio não são verificadas automaticamente.")
    return warnings


def _draw_axes(canvas: Image.Image, fmin: float, fmax: float, footer_left: str, footer_mid: str, footer_right: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((61, 15, 62 + 768, 16 + 216), outline="#1d4d49", width=1)
    top_label = f"{fmax / 1000:.1f} kHz".replace(".", ",")
    ticks = [(16, top_label), (88, "650 Hz"), (156, "300 Hz"), (222, f"{int(fmin)} Hz")]
    for y, lbl in ticks:
        draw.line((56, y, 61, y), fill="#5b8b87", width=1)
        draw.text((8, max(6, y - 5)), lbl, fill="#a8bdbb")
    if footer_left:
        draw.text((62, 240), footer_left, fill="#a8bdbb")
    if footer_mid:
        draw.text((425, 240), footer_mid, fill="#7c9694")
    if footer_right:
        draw.text((760, 240), footer_right, fill="#a8bdbb")


def spectrogram_png(features: np.ndarray) -> str:
    values = np.clip(np.flipud(features), 0.0, 1.0)
    # Paleta sequencial escura local harmonizada com a interface.
    stops = np.array([
        [3, 19, 19],
        [8, 56, 56],
        [20, 128, 120],
        [95, 214, 188],
        [218, 245, 168],
        [248, 200, 255],
    ], dtype=float)
    position = values * (len(stops) - 1)
    low = np.minimum(position.astype(int), len(stops) - 2)
    fraction = (position - low)[..., None]
    rgb = (stops[low] * (1.0 - fraction) + stops[low + 1] * fraction).astype(np.uint8)
    plot = Image.fromarray(rgb).resize((768, 216), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (856, 264), "#011514")
    canvas.paste(plot, (62, 16))
    _draw_axes(
        canvas,
        float(FEATURE_CONFIG["fmin_hz"]),
        float(FEATURE_CONFIG["fmax_hz"]),
        "Inicio do ciclo",
        "Meio",
        "Fim do ciclo",
    )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def contrast_png(features: np.ndarray) -> str:
    """Mapa de contraste acústico (após CMN): ciano destaca cristas tonais (sibilos); âmbar destaca transientes (estertores)."""
    s = np.asarray(features, dtype=np.float32)
    cmn = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 0.08)
    smooth_f = uniform_filter1d(cmn, size=9, axis=0, mode="nearest")
    fc = np.maximum(cmn - smooth_f, 0.0)
    wheeze_map = np.clip(fc / max(float(np.percentile(fc, 99.0)), 1e-4), 0.0, 1.0)
    tdiff = np.abs(np.diff(cmn, axis=1, prepend=cmn[:, :1]))
    crackle_map = np.clip(tdiff / max(float(np.percentile(tdiff, 99.0)), 1e-4), 0.0, 1.0)
    w = np.flipud(wheeze_map)
    c = np.flipud(crackle_map)
    r = np.clip(4.0 + c * 250.0 + w * 40.0, 0.0, 255.0)
    g = np.clip(22.0 + c * 185.0 + w * 220.0, 0.0, 255.0)
    b = np.clip(22.0 + c * 40.0 + w * 235.0, 0.0, 255.0)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    plot = Image.fromarray(rgb).resize((768, 216), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (856, 264), "#011514")
    canvas.paste(plot, (62, 16))
    _draw_axes(
        canvas,
        float(FEATURE_CONFIG["fmin_hz"]),
        float(FEATURE_CONFIG["fmax_hz"]),
        "",
        "",
        "Fim do ciclo",
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((62, 240, 72, 250), fill="#4be6eb")
    draw.text((78, 239), "Ciano: cristas tonais (sibilos)", fill="#a8bdbb")
    draw.rectangle((310, 240, 320, 250), fill="#ffbf3c")
    draw.text((326, 239), "Ambar: transientes (estertores)", fill="#a8bdbb")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
