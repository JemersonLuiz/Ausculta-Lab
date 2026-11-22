"""Arquiteturas neurais puras em PyTorch; pesos somente são gerados pelo treinamento documentado."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in (16, 32, 64):
            layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(),
                                        nn.Dropout(0.3), nn.Linear(64 * 4 * 8, 64),
                                        nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(values))


class BandCNN(nn.Module):
    """Extrai padrões locais e resume energia típica e picos por faixa."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=(2, 4), padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2 * 64 * 4 * 2, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 4),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features((values - 0.5) * 3)
        average = F.adaptive_avg_pool2d(features, (4, 2))
        peaks = F.adaptive_max_pool2d(features, (4, 2))
        return self.classifier(torch.cat((average, peaks), dim=1).flatten(1))


def _instance_z(t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mu = t.mean(dim=(2, 3), keepdim=True)
    sd = t.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return ((t - mu) / sd).clamp(-5.0, 5.0)


class AcousticFrontEnd(nn.Module):
    """Converte o espectrograma (B, 1, 64, 256) em 4 canais estritamente invariantes ao estetoscópio.

    Calcula CMN antes do contraste em frequência e da derivada temporal, e aplica
    normalização por instância em cada canal para anular tanto a curva de EQ/ressonância
    do microfone quanto diferenças de piso de ruído (ex.: ANR do Littmann 3200 vs AKGC417L).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cmn_raw = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 0.08)
            cmn = _instance_z(cmn_raw)
            smooth_f = F.avg_pool2d(cmn, kernel_size=(9, 1), stride=1, padding=(4, 0), count_include_pad=False)
            fcon = _instance_z(cmn - smooth_f)
            c_pad = F.pad(cmn, (1, 1, 0, 0), mode="replicate")
            tdiff = _instance_z(c_pad[:, :, :, 2:] - c_pad[:, :, :, :-2])
            env = x.mean(dim=2, keepdim=True)
            env = ((env - env.mean(dim=-1, keepdim=True)) / env.std(dim=-1, keepdim=True).clamp_min(0.03)).clamp(-5.0, 5.0)
            return torch.cat((cmn, fcon, tdiff, env.expand_as(x)), dim=1)


class VectorizedSpectralStats(nn.Module):
    """Descritores acústicos físicos invariantes ao dispositivo (128 dimensões, sobre canais CMN z-normalizados)."""

    def __init__(self, n_mels: int = 64, n_dct: int = 16):
        super().__init__()
        n = torch.arange(n_mels, dtype=torch.float32)
        k = torch.arange(n_dct, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(math.pi * (n + 0.5) * k / n_mels) * math.sqrt(2.0 / n_mels)
        basis[0] *= math.sqrt(0.5)
        self.register_buffer("dct_basis", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cmn_4d = _instance_z((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 0.08))
            smooth_f = F.avg_pool2d(cmn_4d, kernel_size=(9, 1), stride=1, padding=(4, 0), count_include_pad=False)
            fcon_4d = F.relu(_instance_z(cmn_4d - smooth_f))
            tdiff_4d = _instance_z(cmn_4d[:, :, :, 1:] - cmn_4d[:, :, :, :-1]).abs()

            cmn = cmn_4d.squeeze(1)
            fcon = fcon_4d.squeeze(1)
            tdiff = tdiff_4d.squeeze(1)

            cep_cmn = torch.matmul(self.dct_basis, cmn)
            cep_abs = torch.matmul(self.dct_basis, fcon)
            cep_feats = [cep_cmn.std(dim=-1), cep_cmn.amax(dim=-1) - cep_cmn.amin(dim=-1), cep_abs.mean(dim=-1)]

            ft = fcon.amax(dim=1)
            ft_pow = (ft * ft).mean(dim=-1).sqrt()
            ft_sust = F.avg_pool1d(ft.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
            tonal = torch.stack((
                ft.mean(dim=-1), ft.std(dim=-1), ft_pow, ft.amax(dim=-1),
                ft_sust.amax(dim=-1),
                (ft_sust > 0.8).float().mean(dim=-1),
                (ft_sust > 1.3).float().mean(dim=-1),
                (ft > 2.2).float().mean(dim=-1),
            ), dim=1)

            tr = tdiff.mean(dim=1)
            tr_max = tdiff.amax(dim=1)
            tr_mu = tr.mean(dim=-1, keepdim=True)
            tr_sd = tr.std(dim=-1, keepdim=True).clamp_min(1e-4)
            tr_kurt = ((tr - tr_mu) / tr_sd).pow(4).mean(dim=-1).clamp_max(25.0) / 25.0
            trans = torch.stack((
                tr.mean(dim=-1), tr.std(dim=-1), (tr * tr).mean(dim=-1).sqrt(),
                tr.amax(dim=-1) / tr.mean(dim=-1).clamp_min(1e-4),
                tr_max.mean(dim=-1), tr_max.std(dim=-1), tr_max.amax(dim=-1), tr_kurt,
            ), dim=1)

            batch_size = x.shape[0]
            sf = fcon.view(batch_size, 8, 8, -1)
            st = tdiff.view(batch_size, 8, 8, -1)
            sc = cmn.view(batch_size, 8, 8, -1)
            band_feats = torch.cat((
                sf.mean(dim=(2, 3)), sf.amax(dim=(2, 3)), (sf > 1.0).float().mean(dim=(2, 3)),
                st.mean(dim=(2, 3)), st.amax(dim=(2, 3)), st.std(dim=(2, 3)),
                sc.std(dim=(2, 3)), sc.amax(dim=(2, 3)) - sc.amin(dim=(2, 3)),
            ), dim=1)
            return torch.cat(cep_feats + [tonal, trans, band_feats], dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class MultiTaskHead(nn.Module):
    """Cabeçalho estruturado com detectores compartilhados de estertores/sibilos + interação 4 classes."""

    def __init__(self, in_dim: int, hidden: int = 128, drop: float = 0.28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(drop * 0.8),
        )
        self.bin_head = nn.Linear(hidden, 2)
        self.joint_head = nn.Linear(hidden, 4)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(feat)
        bin_logits = self.bin_head(h)
        joint_res = self.joint_head(h)
        c, w = bin_logits[:, 0], bin_logits[:, 1]
        lc1, lc0 = F.logsigmoid(c), F.logsigmoid(-c)
        lw1, lw0 = F.logsigmoid(w), F.logsigmoid(-w)
        fact = torch.stack((lc0 + lw0, lc1 + lw0, lc0 + lw1, lc1 + lw1), dim=1)
        return fact + joint_res, bin_logits


class RespiraSEBandNet(nn.Module):
    """Rede híbrida multi-escala invariante ao estetoscópio + ramo estatístico acústico."""

    def __init__(self):
        super().__init__()
        self.frontend = AcousticFrontEnd()
        self.stats = VectorizedSpectralStats()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=5, stride=(2, 4), padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.dir_h = nn.Conv2d(24, 16, kernel_size=(3, 7), padding=(1, 3), bias=False)
        self.dir_v = nn.Conv2d(24, 16, kernel_size=(7, 3), padding=(3, 1), bias=False)
        self.dir_i = nn.Conv2d(24, 16, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.dir_bn = nn.BatchNorm2d(48)
        self.dir_se = SEBlock(48)

        self.b2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            SEBlock(64),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.12),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            SEBlock(96),
        )
        self.attn = nn.Conv2d(96, 1, 1)
        self.stats_mlp = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        self.head = MultiTaskHead(96 + 384 + 384 + 64, hidden=128, drop=0.30)

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(self.frontend(x))
        h = self.dir_se(F.silu(self.dir_bn(torch.cat((self.dir_h(h), self.dir_v(h), self.dir_i(h)), dim=1))))
        h = self.b3(self.b2(h))
        w = torch.softmax(self.attn(h).flatten(2), dim=-1)
        p_attn = (h.flatten(2) * w).sum(dim=-1)
        p_avg = F.adaptive_avg_pool2d(h, (4, 1)).flatten(1)
        p_max = F.adaptive_max_pool2d(h, (4, 1)).flatten(1)
        st = self.stats_mlp(self.stats(x))
        return self.head(torch.cat((p_attn, p_avg, p_max, st), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)[0]


class RespiraDilatedNet(nn.Module):
    """Arquitetura complementar 2D com convoluções dilatadas e pooling multi-faixa."""

    def __init__(self):
        super().__init__()
        self.frontend = AcousticFrontEnd()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=5, stride=(2, 4), padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(24, 40, 3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU(),
            SEBlock(40),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(40, 64, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            SEBlock(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 88, 3, padding=(1, 2), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(88),
            nn.SiLU(),
            SEBlock(88),
        )
        self.head = MultiTaskHead(88 * 4 * 2 * 2, hidden=128, drop=0.30)

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.c3(self.c2(self.c1(self.stem(self.frontend(x)))))
        p_avg = F.adaptive_avg_pool2d(h, (4, 2)).flatten(1)
        p_max = F.adaptive_max_pool2d(h, (4, 2)).flatten(1)
        return self.head(torch.cat((p_avg, p_max), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)[0]


class ExtendedCMNStats(nn.Module):
    """Descritores acústicos físicos 100% invariantes ao estetoscópio (192 dims, estritamente sobre canais CMN)."""

    def __init__(self):
        super().__init__()
        self.base = VectorizedSpectralStats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base128 = self.base(x)
            cmn_4d = _instance_z((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 0.08))
            smooth_f = F.avg_pool2d(cmn_4d, kernel_size=(9, 1), stride=1, padding=(4, 0), count_include_pad=False)
            fcon_4d = F.relu(_instance_z(cmn_4d - smooth_f))
            tdiff_4d = _instance_z(cmn_4d[:, :, :, 1:] - cmn_4d[:, :, :, :-1]).abs()

            b_sz = x.shape[0]
            cmn = cmn_4d.squeeze(1)
            fcon = fcon_4d.squeeze(1)
            tdiff = tdiff_4d.squeeze(1)

            fq = fcon.view(b_sz, 4, 16, 4, 64)
            tq = tdiff[:, :, :252].view(b_sz, 4, 16, 4, 63)

            env = x.squeeze(1).mean(dim=1)
            env = ((env - env.mean(dim=-1, keepdim=True)) / env.std(dim=-1, keepdim=True).clamp_min(0.03)).clamp(-5.0, 5.0)
            eq = env.view(b_sz, 8, 32)

            ft = fcon.amax(dim=1)[:, :255]
            tr = tdiff.mean(dim=1)
            tr_max = tdiff.amax(dim=1)
            cooc = torch.stack((
                (ft * tr).mean(dim=-1),
                (ft * tr).amax(dim=-1),
                ((ft > 1.2) & (tr_max > 2.0)).float().mean(dim=-1),
                ((ft > 1.5) & (tr > 0.8)).float().mean(dim=-1),
                ft.amax(dim=-1) * tr.amax(dim=-1) * 0.1,
                ft.mean(dim=-1) * tr_max.mean(dim=-1),
                (ft > 1.0).float().mean(dim=-1),
                (tr_max > 2.2).float().mean(dim=-1),
                fcon[:, :, :128].mean(dim=(1, 2)) - fcon[:, :, 128:].mean(dim=(1, 2)),
                tdiff[:, :, :128].mean(dim=(1, 2)) - tdiff[:, :, 128:].mean(dim=(1, 2)),
                cmn[:, :, :128].std(dim=(1, 2)) - cmn[:, :, 128:].std(dim=(1, 2)),
                env[:, :128].mean(dim=-1) - env[:, 128:].mean(dim=-1),
                env.amax(dim=-1),
                env.amin(dim=-1),
                (env > 0.5).float().mean(dim=-1),
                (env < -0.5).float().mean(dim=-1),
            ), dim=1)

            extra = torch.cat((
                fq.mean(dim=(2, 4)).flatten(1),
                tq.mean(dim=(2, 4)).flatten(1),
                eq.mean(dim=-1),
                eq.std(dim=-1),
                cooc,
            ), dim=1)
            return torch.cat((base128, extra), dim=1)


class SpectralGatedMLP(nn.Module):
    """Rede residual sobre descritores CMN físicos (192 dims) com cabeça multitarefa sem Mixup destrutivo."""

    def __init__(self, in_dim: int = 192, hidden: int = 192, drop: float = 0.35, include_extractor: bool = True):
        super().__init__()
        self.stats = ExtendedCMNStats() if include_extractor else None
        self.bn = nn.BatchNorm1d(in_dim)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(drop))
        self.res1 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.bin_head = nn.Linear(hidden, 2)
        self.joint_head = nn.Linear(hidden, 4)

    def forward_vec(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(self.bn(vec))
        h = F.silu(h + self.res1(h))
        h = F.silu(h + self.res2(h))
        bin_logits = self.bin_head(h)
        joint_res = self.joint_head(h)
        c, w = bin_logits[:, 0], bin_logits[:, 1]
        lc1, lc0 = F.logsigmoid(c), F.logsigmoid(-c)
        lw1, lw0 = F.logsigmoid(w), F.logsigmoid(-w)
        fact = torch.stack((lc0 + lw0, lc1 + lw0, lc0 + lw1, lc1 + lw1), dim=1)
        return fact + joint_res, bin_logits

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vec = self.stats(x) if self.stats is not None else x
        return self.forward_vec(vec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)[0]


class EnsembleRespiraNet(nn.Module):
    """Ensemble calibrado 5-fold (RespiraSEBandNet + RespiraDilatedNet + SpectralGatedMLP) em PyTorch puro."""

    def __init__(self, n_seband: int = 5, n_dilated: int = 5, n_stats: int = 5):
        super().__init__()
        self.seband = nn.ModuleList([RespiraSEBandNet() for _ in range(n_seband)])
        self.dilated = nn.ModuleList([RespiraDilatedNet() for _ in range(n_dilated)])
        self.stats_ext = ExtendedCMNStats()
        self.stats_mlp = nn.ModuleList([SpectralGatedMLP(192, include_extractor=False) for _ in range(n_stats)])
        self.register_buffer("branch_weights", torch.tensor([0.35, 0.35, 0.30], dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("stats_temperature", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("class_bias", torch.zeros(4, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp_min(0.2)
        st_temp = self.stats_temperature.clamp_min(0.2)
        w = self.branch_weights.clamp_min(0.0)
        w = w / w.sum().clamp_min(1e-6)

        probs = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
        if len(self.seband) > 0 and float(w[0]) > 0.0:
            p_se = sum(torch.softmax(m(x) / temp, dim=1) for m in self.seband) / len(self.seband)
            probs = probs + w[0] * p_se
        if len(self.dilated) > 0 and float(w[1]) > 0.0:
            p_di = sum(torch.softmax(m(x) / temp, dim=1) for m in self.dilated) / len(self.dilated)
            probs = probs + w[1] * p_di
        if len(self.stats_mlp) > 0 and float(w[2]) > 0.0:
            vec = self.stats_ext(x)
            p_st = sum(torch.softmax(m.forward_vec(vec)[0] / st_temp, dim=1) for m in self.stats_mlp) / len(self.stats_mlp)
            probs = probs + w[2] * p_st
        return torch.log(probs.clamp_min(1e-7)) + self.class_bias


def build_model(architecture: str = "small_cnn_v1") -> nn.Module:
    if architecture == "small_cnn_v1":
        return SmallCNN()
    if architecture == "band_cnn_v2":
        return BandCNN()
    if architecture == "respira_seband_v2":
        return RespiraSEBandNet()
    if architecture == "respira_dilated_v2":
        return RespiraDilatedNet()
    if architecture == "respira_stats_mlp_v2":
        return SpectralGatedMLP(in_dim=192, include_extractor=True)
    if architecture == "respira_ensemble_v2":
        return EnsembleRespiraNet(n_seband=5, n_dilated=5, n_stats=5)
    raise ValueError(f"Arquitetura incompatível: {architecture}")
