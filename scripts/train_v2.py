"""Treinamento reprodutível v2.0: Ensemble 5-fold por paciente (RespiraSEBandNet + RespiraDilatedNet + SpectralGatedMLP).

Protocolo científico estrito:
- Os 49 pacientes do teste oficial (2.756 ciclos) NUNCA são usados no treinamento,
  na seleção de época, na ponderação do ensemble nem na calibração de temperatura/viés.
- Toda seleção e calibração ocorre exclusivamente via validação cruzada 5-fold agrupada
  por paciente nos 77 pacientes de treino/validação (4.025 ciclos).
- Todos os pesos do ensemble (15 submodelos) são empacotados em um único arquivo models/cnn.pt
  carregável com weights_only=True e checksum SHA-256 verificado.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cardio_audio import __version__
from cardio_audio.audio import CLASS_NAMES, FEATURE_CONFIG
from cardio_audio.metrics import classification_metrics, patient_bootstrap
from cardio_audio.model import (
    EnsembleRespiraNet,
    ExtendedCMNStats,
    RespiraDilatedNet,
    RespiraSEBandNet,
    SpectralGatedMLP,
)
from cardio_audio.predictor import Predictor
from train import make_predictions, prepare_features, save_json


def augment_spec(batch: torch.Tensor, gen: torch.Generator, lowpass_prob: float = 0.35) -> torch.Tensor:
    B, _, F_bins, T_bins = batch.shape
    # 1. Deslocamento circular temporal (invariância à posição do evento no ciclo)
    shift = int(torch.randint(-T_bins // 4, T_bins // 4 + 1, (1,), generator=gen).item())
    out = torch.roll(batch, shifts=shift, dims=-1)
    # 2. Variação suave de curva de resposta / ganho / contraste gama entre estetoscópios
    f_grid = torch.linspace(-1.0, 1.0, F_bins).view(1, 1, F_bins, 1)
    c1 = (torch.rand((B, 1, 1, 1), generator=gen) - 0.5) * 0.28
    c2 = (torch.rand((B, 1, 1, 1), generator=gen) - 0.5) * 0.18
    gamma = torch.rand((B, 1, 1, 1), generator=gen) * 0.50 + 0.75
    out = (out.clamp_min(1e-4).pow(gamma) + c1 * f_grid + c2 * (f_grid ** 2 - 0.33)).clamp(0.0, 1.0)
    # 3. Simulação de redução ativa de ruído (ANR suave como Littmann 3200) vs chiado de microfone (AKG)
    smooth_mask = (torch.rand((B, 1, 1, 1), generator=gen) < 0.35).float()
    if smooth_mask.any():
        smoothed = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        alpha = torch.rand((B, 1, 1, 1), generator=gen) * 0.55 + 0.20
        out = out * (1.0 - smooth_mask * alpha) + smoothed * (smooth_mask * alpha)
    # 4. Simulação física de atenuação de agudos
    if lowpass_prob > 0:
        apply_lp = torch.rand((B,), generator=gen) < lowpass_prob
        if apply_lp.any():
            f_idx = torch.arange(F_bins, dtype=torch.float32).view(1, 1, F_bins, 1)
            cutoffs = torch.randint(42, 60, (B, 1, 1, 1), generator=gen).float()
            slope = torch.rand((B, 1, 1, 1), generator=gen) * 0.06 + 0.03
            damp = torch.clamp(1.0 - (f_idx - cutoffs) * slope, min=0.15, max=1.0)
            out = out * (1.0 - apply_lp.view(B, 1, 1, 1).float() * (1.0 - damp))
    # 5. Ruído térmico de microfone (variando SNR)
    noise_std = torch.rand((B, 1, 1, 1), generator=gen) * 0.04
    out = (out + torch.randn(out.shape, generator=gen) * noise_std).clamp(0.0, 1.0)
    # 6. SpecAugment preenchido pela média (evita bordas artificiais confundidas com estertores)
    means = out.mean(dim=(2, 3), keepdim=True)
    fw = torch.randint(0, 7, (B, 1), generator=gen)
    fs = (torch.rand((B, 1), generator=gen) * (F_bins - fw + 1)).long()
    f_pos = torch.arange(F_bins).unsqueeze(0)
    out = torch.where(((f_pos >= fs) & (f_pos < fs + fw)).view(B, 1, F_bins, 1), means, out)
    tw = torch.randint(0, 26, (B, 1), generator=gen)
    ts = (torch.rand((B, 1), generator=gen) * (T_bins - tw + 1)).long()
    t_pos = torch.arange(T_bins).unsqueeze(0)
    out = torch.where(((t_pos >= ts) & (t_pos < ts + tw)).view(B, 1, 1, T_bins), means, out)
    return out


def eval_logits(model: nn.Module, features: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outs = []
    with torch.inference_mode():
        for i in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[i:i + batch_size]).unsqueeze(1)
            outs.append(model(xb).numpy())
    return np.concatenate(outs, axis=0)


def extract_cmn_stats_matrix(features: np.ndarray, batch_size: int = 256) -> np.ndarray:
    ext = ExtendedCMNStats()
    ext.eval()
    outs = []
    with torch.inference_mode():
        for i in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[i:i + batch_size]).unsqueeze(1)
            outs.append(ext(xb).numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def extract_augmented_cmn_stats(features: np.ndarray, n_views: int, seed: int) -> np.ndarray:
    """Gera visões de descritores CMN sob aumento físico de estetoscópio (ANR/EQ/low-pass/SpecAugment)."""
    ext = ExtendedCMNStats()
    ext.eval()
    gen = torch.Generator().manual_seed(seed)
    views = []
    with torch.inference_mode():
        for _ in range(n_views):
            chunks = []
            for i in range(0, len(features), 256):
                xb = torch.from_numpy(features[i:i + 256]).unsqueeze(1)
                chunks.append(ext(augment_spec(xb, gen, lowpass_prob=0.35)).numpy())
            views.append(np.concatenate(chunks, axis=0).astype(np.float32))
    return np.stack(views, axis=0)


def train_single_fold(
    model_fn,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    class_power: float,
    lowpass_prob: float,
    seed: int,
    tag: str,
) -> tuple[dict, np.ndarray, int, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = model_fn()
    counts = np.bincount(y_tr, minlength=4).astype(np.float32)
    cw = torch.tensor((len(y_tr) / (4.0 * np.maximum(counts, 1.0))) ** class_power, dtype=torch.float32)
    cw = cw / cw.mean()

    y_bin_tr = np.stack(((y_tr == 1) | (y_tr == 3), (y_tr == 2) | (y_tr == 3)), axis=1).astype(np.float32)
    pos_rate = y_bin_tr.mean(axis=0)
    pos_weight = torch.tensor(((1.0 - pos_rate) / np.maximum(pos_rate, 1e-4)) ** 0.45, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr).unsqueeze(1),
            torch.from_numpy(y_tr.astype(np.int64)),
            torch.from_numpy(y_bin_tr),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )

    best_score = -1.0
    best_state = None
    best_logits = None
    best_epoch = 0
    t0 = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb, ybin in loader:
            xb = augment_spec(xb, gen, lowpass_prob=lowpass_prob)
            y_oh = F.one_hot(yb, 4).float() * 0.92 + 0.02
            if float(torch.rand(1, generator=gen)) < 0.50:
                lam = float(np.random.beta(0.4, 0.4))
                perm = torch.randperm(len(xb), generator=gen)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                y_oh = lam * y_oh + (1.0 - lam) * y_oh[perm]
                ybin = lam * ybin + (1.0 - lam) * ybin[perm]
            optimizer.zero_grad(set_to_none=True)
            logits, bin_logits = model.forward_with_aux(xb)
            loss_4 = -(y_oh * F.log_softmax(logits, dim=1) * cw.unsqueeze(0)).sum(dim=1).mean()
            loss_b = F.binary_cross_entropy_with_logits(bin_logits, ybin, pos_weight=pos_weight)
            loss = loss_4 + 0.5 * loss_b
            if not torch.isfinite(loss):
                raise RuntimeError("Perda não finita durante treinamento v2.0.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        scheduler.step()

        if epoch >= max(4, epochs // 3):
            va_logits = eval_logits(model, X_va)
            m = classification_metrics(y_va, va_logits.argmax(axis=1))
            score = 0.45 * m["macro_f1"] + 0.30 * (m["icbhi_score"] or 0.0) + 0.25 * m["accuracy"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_logits = va_logits.copy()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    m_best = classification_metrics(y_va, best_logits.argmax(axis=1))
    print(
        f"  [{tag}] ep={best_epoch:2d}/{epochs} ({time.monotonic()-t0:5.1f}s) | "
        f"val Acc={m_best['accuracy']*100:5.2f}% MacroF1={m_best['macro_f1']:.4f} ICBHI={m_best['icbhi_score']*100:5.2f}%",
        flush=True,
    )
    return best_state, best_logits, best_epoch, best_score


def train_stats_mlp_fold(
    V_clean_tr: np.ndarray,
    V_aug_tr: np.ndarray,
    y_tr: np.ndarray,
    V_clean_va: np.ndarray,
    y_va: np.ndarray,
    *,
    epochs: int,
    seed: int,
    tag: str,
) -> tuple[dict, np.ndarray, int]:
    """Treina SpectralGatedMLP sobre descritores CMN físicos (192 dims) SEM Mixup destrutivo entre classes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = SpectralGatedMLP(in_dim=V_clean_tr.shape[1], hidden=192, drop=0.35, include_extractor=False)
    counts = np.bincount(y_tr, minlength=4).astype(np.float32)
    cw = torch.tensor((len(y_tr) / (4.0 * np.maximum(counts, 1.0))) ** 0.55, dtype=torch.float32)
    cw = cw / cw.mean()

    y_bin_tr = np.stack(((y_tr == 1) | (y_tr == 3), (y_tr == 2) | (y_tr == 3)), axis=1).astype(np.float32)
    pos_rate = y_bin_tr.mean(axis=0)
    pos_weight = torch.tensor(((1.0 - pos_rate) / np.maximum(pos_rate, 1e-4)) ** 0.50, dtype=torch.float32)

    # Combina visão limpa + visões fisicamente aumentadas (mesmos pacientes de treino!)
    V_pool = np.concatenate([V_clean_tr] + [V_aug_tr[v] for v in range(len(V_aug_tr))], axis=0)
    y_pool = np.tile(y_tr, 1 + len(V_aug_tr))
    yb_pool = np.tile(y_bin_tr, (1 + len(V_aug_tr), 1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(V_pool),
            torch.from_numpy(y_pool.astype(np.int64)),
            torch.from_numpy(yb_pool),
        ),
        batch_size=96,
        shuffle=True,
        generator=gen,
    )

    va_tensor = torch.from_numpy(V_clean_va)
    best_score = -1.0
    best_state = None
    best_logits = None
    best_epoch = 0
    t0 = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        for vb, yb, ybin in loader:
            vb = vb + torch.randn(vb.shape, generator=gen) * 0.08
            optimizer.zero_grad(set_to_none=True)
            logits, bin_logits = model.forward_vec(vb)
            loss = (
                F.cross_entropy(logits, yb, weight=cw, label_smoothing=0.04)
                + 0.5 * F.binary_cross_entropy_with_logits(bin_logits, ybin, pos_weight=pos_weight)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Perda não finita em SpectralGatedMLP.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        scheduler.step()

        if epoch >= max(8, epochs // 3):
            model.eval()
            with torch.inference_mode():
                va_logits = model.forward_vec(va_tensor)[0].numpy()
            m = classification_metrics(y_va, va_logits.argmax(axis=1))
            score = 0.50 * m["macro_f1"] + 0.30 * (m["icbhi_score"] or 0.0) + 0.20 * m["accuracy"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_logits = va_logits.copy()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    m_best = classification_metrics(y_va, best_logits.argmax(axis=1))
    print(
        f"  [{tag}] ep={best_epoch:2d}/{epochs} ({time.monotonic()-t0:5.1f}s) | "
        f"val Acc={m_best['accuracy']*100:5.2f}% MacroF1={m_best['macro_f1']:.4f} ICBHI={m_best['icbhi_score']*100:5.2f}%",
        flush=True,
    )
    return best_state, best_logits, best_epoch


def calibrate_oof_3way(
    oof_se: np.ndarray,
    oof_di: np.ndarray,
    oof_st: np.ndarray,
    y_oof: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Calibra pesos das 3 famílias, temperaturas (por NLL) e viés suave estritamente nos 4.025 ciclos OOF."""
    y_t = torch.from_numpy(y_oof.astype(np.int64))
    se_t = torch.from_numpy(oof_se)
    di_t = torch.from_numpy(oof_di)
    st_t = torch.from_numpy(oof_st)

    # 1. Temperatura ótima por família via NLL estrito
    best_temp_cnn, best_nll_cnn = 1.0, float("inf")
    for tc in (0.65, 0.75, 0.85, 0.95, 1.05, 1.20):
        p = 0.5 * torch.softmax(se_t / tc, dim=1) + 0.5 * torch.softmax(di_t / tc, dim=1)
        nll = float(F.nll_loss(torch.log(p.clamp_min(1e-7)), y_t).item())
        if nll < best_nll_cnn:
            best_nll_cnn, best_temp_cnn = nll, tc

    best_temp_st, best_nll_st = 1.0, float("inf")
    for ts in (0.75, 0.90, 1.00, 1.15, 1.30, 1.50, 1.75):
        p = torch.softmax(st_t / ts, dim=1)
        nll = float(F.nll_loss(torch.log(p.clamp_min(1e-7)), y_t).item())
        if nll < best_nll_st:
            best_nll_st, best_temp_st = nll, ts

    p_se = torch.softmax(se_t / best_temp_cnn, dim=1)
    p_di = torch.softmax(di_t / best_temp_cnn, dim=1)
    p_st = torch.softmax(st_t / best_temp_st, dim=1)

    # 2. Pesos das 3 famílias otimizando combinação equilibrada de NLL + Macro-F1 + Acurácia + ICBHI no OOF
    best_w = np.array([0.32, 0.33, 0.35], dtype=np.float32)
    best_combo = -1e9
    counts = np.bincount(y_oof, minlength=4).astype(np.float64)
    for w_st in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        rem = 1.0 - w_st
        for frac_se in (0.35, 0.45, 0.50, 0.55, 0.65):
            w_se = round(rem * frac_se, 4)
            w_di = round(rem - w_se, 4)
            probs = w_se * p_se + w_di * p_di + w_st * p_st
            nll = float(F.nll_loss(torch.log(probs.clamp_min(1e-7)), y_t).item())
            pred = probs.argmax(dim=1).numpy()
            cm = np.bincount(4 * y_oof + pred, minlength=16).reshape(4, 4)
            tp = np.diag(cm).astype(np.float64)
            denom = counts + cm.sum(axis=0)
            mf1 = float(np.mean(np.divide(2.0 * tp, denom, out=np.zeros(4), where=denom > 0)))
            acc = float(tp.sum() / len(y_oof))
            icb = 0.5 * (float(cm[0, 0] / counts[0]) + float(np.trace(cm[1:, 1:]) / counts[1:].sum()))
            score = 0.40 * mf1 + 0.30 * acc + 0.20 * icb - 0.15 * nll
            if score > best_combo:
                best_combo = score
                best_w = np.array([w_se, w_di, w_st], dtype=np.float32)

    probs = float(best_w[0]) * p_se + float(best_w[1]) * p_di + float(best_w[2]) * p_st
    nll_ens = float(F.nll_loss(torch.log(probs.clamp_min(1e-7)), y_t).item())
    log_p = torch.log(probs.clamp_min(1e-7)).numpy()

    # 3. Viés conservador regularizado (|b| <= 0.36) estritamente no OOF
    best_obj = -1e9
    best_bias = np.zeros(4, dtype=np.float32)
    grid = np.round(np.arange(-0.36, 0.361, 0.06), 3)
    for b1, b2, b3 in product(grid, repeat=3):
        bias = np.array([0.0, b1, b2, b3], dtype=np.float32)
        pred = (log_p + bias).argmax(axis=1)
        cm = np.bincount(4 * y_oof + pred, minlength=16).reshape(4, 4)
        tp = np.diag(cm).astype(np.float64)
        denom = counts + cm.sum(axis=0)
        mf1 = float(np.mean(np.divide(2.0 * tp, denom, out=np.zeros(4), where=denom > 0)))
        bal = float(np.mean(tp / counts))
        acc = float(tp.sum() / len(y_oof))
        icb = 0.5 * (float(cm[0, 0] / counts[0]) + float(np.trace(cm[1:, 1:]) / counts[1:].sum()))
        reg = 0.025 * float(bias @ bias)
        obj = 0.25 * acc + 0.40 * mf1 + 0.20 * icb + 0.15 * bal - reg
        if obj > best_obj:
            best_obj = obj
            best_bias = bias.copy()

    print(
        f"Calibração OOF 3-famílias (77 pacientes / {len(y_oof)} ciclos): "
        f"weights={best_w.round(3).tolist()}, temp_cnn={best_temp_cnn:.2f}, temp_stats={best_temp_st:.2f}, "
        f"nll={nll_ens:.4f}, class_bias={best_bias.round(3).tolist()}",
        flush=True,
    )
    return best_w, best_temp_cnn, best_temp_st, best_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-seband", type=int, default=16)
    parser.add_argument("--epochs-dilated", type=int, default=16)
    parser.add_argument("--epochs-stats", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-cnn-from",
        type=Path,
        default=None,
        help="Opcional: reutiliza os 5 folds SEBand + 5 folds Dilated já treinados no mesmo protocolo determinístico.",
    )
    args = parser.parse_args()

    started = time.monotonic()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)

    manifest_path = ROOT / "data" / "manifest.csv"
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str})
    assert frame.groupby("patient_id").split.nunique().max() == 1, "Sobreposição de pacientes detectada!"
    assert frame.groupby("audio_sha256").split.nunique().max() == 1, "Sobreposição de áudio detectada!"

    fingerprint = hashlib.sha256(manifest_bytes + json.dumps(FEATURE_CONFIG, sort_keys=True).encode()).hexdigest()
    cache_path = ROOT / "data" / "processed" / f"features_{fingerprint[:16]}.npz"
    print(f"Preparando features determinísticas ({FEATURE_CONFIG['version']})...", flush=True)
    features, _ = prepare_features(frame, cache_path, fingerprint)
    print(f"Features prontas: {features.shape} ({time.monotonic()-started:.1f}s)", flush=True)

    non_test_idx = np.flatnonzero(frame["split"].to_numpy() != "test")
    test_idx = np.flatnonzero(frame["split"].to_numpy() == "test")
    X_nt = features[non_test_idx]
    y_nt = frame.iloc[non_test_idx]["label"].to_numpy(dtype=np.int64)
    groups_nt = frame.iloc[non_test_idx]["patient_id"].to_numpy()

    folds = list(StratifiedGroupKFold(n_splits=5).split(np.zeros((len(y_nt), 1)), y_nt, groups_nt))
    print(f"5-fold StratifiedGroupKFold sobre {len(non_test_idx)} ciclos ({len(np.unique(groups_nt))} pacientes)...", flush=True)

    se_states, di_states, st_states = [], [], []
    se_epochs, di_epochs, st_epochs = [], [], []
    oof_se = np.zeros((len(y_nt), 4), dtype=np.float32)
    oof_di = np.zeros((len(y_nt), 4), dtype=np.float32)
    oof_st = np.zeros((len(y_nt), 4), dtype=np.float32)

    if args.reuse_cnn_from is not None:
        print(f"Reutilizando pesos CNN 5-fold (SEBand + Dilated) de {args.reuse_cnn_from}...", flush=True)
        raw_sd = torch.load(args.reuse_cnn_from, map_location="cpu", weights_only=True)
        for fi, (_, va) in enumerate(folds):
            se_prefix = f"seband.{fi}."
            di_prefix = f"dilated.{fi}."
            se_sd = {k[len(se_prefix):]: v.clone() for k, v in raw_sd.items() if k.startswith(se_prefix)}
            di_sd = {k[len(di_prefix):]: v.clone() for k, v in raw_sd.items() if k.startswith(di_prefix)}
            m_se_fold = RespiraSEBandNet()
            m_se_fold.load_state_dict(se_sd, strict=True)
            oof_se[va] = eval_logits(m_se_fold, X_nt[va])
            se_states.append(se_sd)

            m_di_fold = RespiraDilatedNet()
            m_di_fold.load_state_dict(di_sd, strict=True)
            oof_di[va] = eval_logits(m_di_fold, X_nt[va])
            di_states.append(di_sd)
        old_meta = json.loads((ROOT / "models" / "model.json").read_text(encoding="utf-8"))
        se_epochs = list(old_meta.get("seband_best_epochs", [args.epochs_seband] * 5))
        di_epochs = list(old_meta.get("dilated_best_epochs", [args.epochs_dilated] * 5))
    else:
        for fi, (tr, va) in enumerate(folds):
            st, lg, ep, _ = train_single_fold(
                RespiraSEBandNet, X_nt[tr], y_nt[tr], X_nt[va], y_nt[va],
                epochs=args.epochs_seband, batch_size=args.batch_size, lr=2.2e-3,
                class_power=0.35, lowpass_prob=0.35, seed=args.seed + fi * 17,
                tag=f"SEBand F{fi+1}/5",
            )
            se_states.append(st)
            oof_se[va] = lg
            se_epochs.append(ep)

        for fi, (tr, va) in enumerate(folds):
            st, lg, ep, _ = train_single_fold(
                RespiraDilatedNet, X_nt[tr], y_nt[tr], X_nt[va], y_nt[va],
                epochs=args.epochs_dilated, batch_size=args.batch_size, lr=2.2e-3,
                class_power=0.35, lowpass_prob=0.35, seed=args.seed + 100 + fi * 17,
                tag=f"Dilated F{fi+1}/5",
            )
            di_states.append(st)
            oof_di[va] = lg
            di_epochs.append(ep)

    print("Extraindo descritores CMN físicos (192 dims) + 2 visões aumentadas de estetoscópio...", flush=True)
    V_clean_nt = extract_cmn_stats_matrix(X_nt)
    V_aug_nt = extract_augmented_cmn_stats(X_nt, n_views=2, seed=args.seed + 777)
    for fi, (tr, va) in enumerate(folds):
        st, lg, ep = train_stats_mlp_fold(
            V_clean_nt[tr], V_aug_nt[:, tr], y_nt[tr], V_clean_nt[va], y_nt[va],
            epochs=args.epochs_stats, seed=args.seed + 200 + fi * 17,
            tag=f"StatsMLP F{fi+1}/5",
        )
        st_states.append(st)
        oof_st[va] = lg
        st_epochs.append(ep)

    m_se = classification_metrics(y_nt, oof_se.argmax(1))
    m_di = classification_metrics(y_nt, oof_di.argmax(1))
    m_st = classification_metrics(y_nt, oof_st.argmax(1))
    print(f"\nOOF SEBand   5-fold: Acc={m_se['accuracy']*100:.2f}% | MacroF1={m_se['macro_f1']:.4f} | ICBHI={m_se['icbhi_score']*100:.2f}%", flush=True)
    print(f"OOF Dilated  5-fold: Acc={m_di['accuracy']*100:.2f}% | MacroF1={m_di['macro_f1']:.4f} | ICBHI={m_di['icbhi_score']*100:.2f}%", flush=True)
    print(f"OOF StatsMLP 5-fold: Acc={m_st['accuracy']*100:.2f}% | MacroF1={m_st['macro_f1']:.4f} | ICBHI={m_st['icbhi_score']*100:.2f}%", flush=True)

    best_w, best_tc, best_ts, best_bias = calibrate_oof_3way(oof_se, oof_di, oof_st, y_nt)
    oof_probs = (
        float(best_w[0]) * torch.softmax(torch.from_numpy(oof_se) / best_tc, dim=1)
        + float(best_w[1]) * torch.softmax(torch.from_numpy(oof_di) / best_tc, dim=1)
        + float(best_w[2]) * torch.softmax(torch.from_numpy(oof_st) / best_ts, dim=1)
    )
    oof_logits_cal = torch.log(oof_probs.clamp_min(1e-7)) + torch.from_numpy(best_bias)
    oof_metrics = classification_metrics(y_nt, oof_logits_cal.argmax(dim=1).numpy())
    print(
        f"OOF Ensemble 3-famílias: Acc={oof_metrics['accuracy']*100:.2f}% | MacroF1={oof_metrics['macro_f1']:.4f} | "
        f"BalAcc={oof_metrics['balanced_accuracy']*100:.2f}% | ICBHI={oof_metrics['icbhi_score']*100:.2f}%",
        flush=True,
    )

    # Empacota os 15 submodelos em um único nn.Module congelado
    ensemble = EnsembleRespiraNet(n_seband=5, n_dilated=5, n_stats=5)
    for mod, st in zip(ensemble.seband, se_states, strict=True):
        mod.load_state_dict(st, strict=True)
    for mod, st in zip(ensemble.dilated, di_states, strict=True):
        mod.load_state_dict(st, strict=True)
    for mod, st in zip(ensemble.stats_mlp, st_states, strict=True):
        mod.load_state_dict(st, strict=True)
    ensemble.branch_weights.copy_(torch.from_numpy(best_w))
    ensemble.temperature.fill_(best_tc)
    ensemble.stats_temperature.fill_(best_ts)
    ensemble.class_bias.copy_(torch.from_numpy(best_bias))
    ensemble.eval()

    models_dir = ROOT / "models"
    weights_path = models_dir / "cnn.pt"
    torch.save(ensemble.state_dict(), weights_path)
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    metadata = {
        "version": __version__,
        "name": "Ensemble RespiraNet v2.0 (invariante a estetoscópio)",
        "task": "pulmonar_4_classes",
        "classes": CLASS_NAMES,
        "feature_config": FEATURE_CONFIG,
        "architecture_id": "respira_ensemble_v2",
        "architecture": "EnsembleRespiraNet: 5x RespiraSEBandNet + 5x RespiraDilatedNet + 5x SpectralGatedMLP (CMN 192-d), invariante a estetoscópio + cabeçalho multitarefa",
        "parameters": sum(p.numel() for p in ensemble.parameters()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "cv_folds": 5,
        "best_epoch": int(round(np.mean(se_epochs + di_epochs + st_epochs))),
        "seband_best_epochs": se_epochs,
        "dilated_best_epochs": di_epochs,
        "stats_mlp_best_epochs": st_epochs,
        "epochs_executed": max(args.epochs_seband, args.epochs_dilated, args.epochs_stats),
        "maximum_epochs": max(args.epochs_seband, args.epochs_dilated, args.epochs_stats),
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": 0.0022,
        "branch_weights": [float(x) for x in best_w],
        "temperature": float(best_tc),
        "stats_temperature": float(best_ts),
        "decision_offsets": [float(x) for x in best_bias],
        "manifest_sha256": manifest_hash,
        "weights_sha256": weights_hash,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "training_seconds": round(time.monotonic() - started, 2),
        "clinical_validation": False,
        "arrhythmia_available": False,
        "cache_quantization": "none; float32 in training and inference",
    }
    save_json(models_dir / "model.json", metadata)

    # Verifica carregamento seguro via Predictor antes da avaliação final
    predictor = Predictor(models_dir)

    # ===== AVALIAÇÃO ÚNICA NO TESTE RESERVADO (49 PACIENTES / 2.756 CICLOS) =====
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    y_test = test_frame["label"].to_numpy(dtype=np.int64)
    test_pred = make_predictions(predictor.model, features[test_idx], batch_size=64)
    test_metrics = classification_metrics(y_test, test_pred)
    test_metrics["macro_f1_ci95"] = patient_bootstrap(y_test, test_pred, test_frame["patient_id"])

    # Preserva histórico v1.0 e v1.1 e adiciona métricas por dispositivo
    comparison = json.loads((models_dir / "metrics.json").read_text(encoding="utf-8"))
    ref_csv = pd.read_csv(ROOT / "reports" / "test_predictions.csv", dtype={"patient_id": str})
    keys = ["patient_id", "recording_id", "start_s", "end_s", "label", "split"]
    pd.testing.assert_frame_equal(test_frame[keys], ref_csv[keys], check_dtype=False)

    if "cnn_v1_1" not in comparison:
        comparison["cnn_v1_1"] = comparison["cnn"]
    if "cnn_v1_1_prediction" not in ref_csv.columns:
        ref_csv["cnn_v1_1_prediction"] = ref_csv["cnn_prediction"]
    v11_pred = ref_csv["cnn_v1_1_prediction"].to_numpy(dtype=np.int64)

    devices_test = test_frame["recording_id"].str.split("_").str[-1].to_numpy()
    per_device = {}
    for dev in sorted(np.unique(devices_test)):
        m_dev = devices_test == dev
        per_device[str(dev)] = {
            "cycles": int(m_dev.sum()),
            "accuracy": float(np.mean(y_test[m_dev] == test_pred[m_dev])),
            "accuracy_v1_1": float(np.mean(y_test[m_dev] == v11_pred[m_dev])),
        }
    test_metrics["per_device"] = per_device

    comparison["cnn"] = {"validation": oof_metrics, "test": test_metrics}
    comparison["selection"] = {
        "criterion": "5fold_patient_disjoint_oof_3family_nll_and_balanced_objective",
        "best_epoch": metadata["best_epoch"],
        "seed": args.seed,
        "test_used_for_selection": False,
        "feature_cache": f"float32, {FEATURE_CONFIG['version']}",
    }
    save_json(models_dir / "metrics.json", comparison)

    ref_csv["cnn_prediction"] = test_pred
    ref_csv.to_csv(ROOT / "reports" / "test_predictions.csv", index=False)

    dataset_json = json.loads((models_dir / "dataset.json").read_text(encoding="utf-8"))
    dataset_json["manifest_sha256"] = manifest_hash
    save_json(models_dir / "dataset.json", dataset_json)

    print("\n================= RESULTADO FINAL NO TESTE RESERVADO (2.756 CICLOS) =================", flush=True)
    v11_t = comparison["cnn_v1_1"]["test"]
    print(f"v1.1 BandCNN  | Acc={v11_t['accuracy']*100:5.2f}% | MacroF1={v11_t['macro_f1']:.4f} | BalAcc={v11_t['balanced_accuracy']*100:5.2f}% | ICBHI={v11_t['icbhi_score']*100:5.2f}%", flush=True)
    print(f"v2.0 Ensemble | Acc={test_metrics['accuracy']*100:5.2f}% | MacroF1={test_metrics['macro_f1']:.4f} | BalAcc={test_metrics['balanced_accuracy']*100:5.2f}% | ICBHI={test_metrics['icbhi_score']*100:5.2f}%", flush=True)
    print("Por estetoscópio:", json.dumps(per_device, indent=2), flush=True)
    print("Por categoria:", [(r["label"], round(r["sensitivity"] * 100, 1), round(r["f1"], 3)) for r in test_metrics["per_class"]], flush=True)
    print(f"Tempo total: {metadata['training_seconds']:.1f}s", flush=True)

    import write_report_v2
    write_report_v2.main()


if __name__ == "__main__":
    main()
    os._exit(0)
