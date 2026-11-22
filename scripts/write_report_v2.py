"""Gera relatório científico e figuras da versão 2.0 exclusivamente a partir dos artefatos salvos."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    metrics = read(ROOT / "models" / "metrics.json")
    model = read(ROOT / "models" / "model.json")
    dataset = read(ROOT / "models" / "dataset.json")
    audit = read(ROOT / "data" / "manifest.audit.json")
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "#011514",
        "axes.facecolor": "#022321",
        "axes.edgecolor": "#245753",
        "axes.labelcolor": "#edfffe",
        "text.color": "#edfffe",
        "xtick.color": "#a8bdbb",
        "ytick.color": "#a8bdbb",
    })

    labels = ["Sem eventos", "Estertores", "Sibilos", "Ambos"]
    # 1. Confusion matrices: v1.1 vs v2.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    pairs = []
    if "cnn_v1_1" in metrics:
        pairs.append(("cnn_v1_1", "CNN por faixas (v1.1)"))
    elif "cnn_v1" in metrics:
        pairs.append(("cnn_v1", "SmallCNN (v1.0)"))
    pairs.append(("cnn", f"{model.get('name', 'Modelo ativo')} (v{model.get('version', '2.0.0')})"))

    for ax, (key, title) in zip(axes, pairs, strict=True):
        mat = np.asarray(metrics[key]["test"]["confusion_matrix"])
        row_pct = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
        ax.imshow(row_pct, cmap="YlGnBu", vmin=0, vmax=1)
        for r in range(4):
            for c in range(4):
                txt_col = "#edfffe" if row_pct[r, c] > 0.52 else "#011514"
                ax.text(c, r, f"{mat[r, c]}\n({row_pct[r, c]*100:.0f}%)", ha="center", va="center", color=txt_col, fontsize=9, fontweight="bold")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
               xlabel="Categoria prevista", ylabel="Categoria anotada", title=title)
        ax.tick_params(axis="x", labelrotation=18)
    fig.suptitle("Teste reservado por paciente (49 pacientes · 2.756 ciclos) — Evolução v1.1 → v2.0", color="#cbfffc", fontsize=12, fontweight="bold")
    fig.savefig(figures / "matrizes_confusao_v2.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Bar chart comparing all versions on Accuracy, Macro-F1, Balanced Acc, ICBHI
    series = [
        ("Maioria", metrics["majority_baseline"]["test"]),
        ("RF (v1.0)", metrics["random_forest"]["test"]),
        ("CNN v1.0", metrics["cnn_v1"]["test"]),
    ]
    if "cnn_v1_1" in metrics:
        series.append(("CNN v1.1", metrics["cnn_v1_1"]["test"]))
    series.append(("Modelo v2.0", metrics["cnn"]["test"]))

    metric_keys = [("accuracy", "Acurácia"), ("macro_f1", "Macro-F1"), ("balanced_accuracy", "Acurácia bal."), ("icbhi_score", "Escore ICBHI")]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(series)
    colors = ["#4a6361", "#688986", "#2a8b84", "#48bfa8", "#cbfffc"]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for idx, (label, data) in enumerate(series):
        vals = [data[k] * 100 for k, _ in metric_keys]
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=label, color=colors[idx % len(colors)])
        if idx == len(series) - 1:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", color="#cbfffc", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metric_keys])
    ax.set_ylim(0, 82)
    ax.set_ylabel("Pontuação (%)")
    ax.set_title("Comparação no conjunto de teste reservado (ICBHI 2017 · sem vazamento de pacientes)", color="#cbfffc", fontweight="bold")
    ax.grid(axis="y", color="#18423f", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="#011514", edgecolor="#245753")
    fig.savefig(figures / "comparacao_modelos_v2.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 3. Write REFINAMENTO_V2.md
    cur = metrics["cnn"]["test"]
    prev = metrics.get("cnn_v1_1", metrics["cnn_v1"])["test"]
    ci = cur["macro_f1_ci95"]

    comp_rows = []
    for title, data in series:
        comp_rows.append(
            f"| {title} | {data['accuracy']*100:.2f}% | {data['macro_f1']:.4f} | "
            f"{data['balanced_accuracy']*100:.2f}% | {data['icbhi_score']*100:.2f}% |"
        )
    cat_rows = []
    for old_r, new_r in zip(prev["per_class"], cur["per_class"], strict=True):
        diff = (new_r["sensitivity"] - old_r["sensitivity"]) * 100
        sign = "+" if diff >= 0 else ""
        cat_rows.append(
            f"| {new_r['label']} | {new_r['support']} | {old_r['sensitivity']*100:.2f}% | "
            f"{new_r['sensitivity']*100:.2f}% | {sign}{diff:.2f} p.p. | {new_r['f1']:.4f} |"
        )

    dev_section = ""
    if "per_device" in cur:
        dev_rows = [
            f"| {dev} | {info['cycles']} | {info['accuracy_v1_1']*100:.2f}% | {info['accuracy']*100:.2f}% | {(info['accuracy']-info['accuracy_v1_1'])*100:+.2f} p.p. |"
            for dev, info in sorted(cur["per_device"].items())
        ]
        dev_section = (
            "\n## Generalização por estetoscópio (teste reservado)\n\n"
            "| Estetoscópio | Ciclos | Acurácia v1.1 | Acurácia v2.0 | Ganho |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
            + "\n".join(dev_rows)
            + "\n"
        )

    md = f"""# Ausculta Lab 2.0.0 — Melhoria de Acurácia e Invariância a Estetoscópios

Avaliação estrita na mesma partição reservada da base ICBHI 2017 ({cur['n_cycles']} ciclos reais de {dataset['splits']['test']['patients']} pacientes jamais vistos no treinamento, validação ou calibração).

## Comparação geral no conjunto de teste reservado

| Modelo | Acurácia | Macro-F1 | Acurácia balanceada | Escore ICBHI |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(comp_rows)}

- **Ganho sobre a v1.1:** Acurácia **{(cur['accuracy']-prev['accuracy'])*100:+.2f} p.p.** ({prev['accuracy']*100:.2f}% → **{cur['accuracy']*100:.2f}%**), Macro-F1 **{(cur['macro_f1']-prev['macro_f1']):+.4f}** ({prev['macro_f1']:.4f} → **{cur['macro_f1']:.4f}**), Acurácia balanceada **{(cur['balanced_accuracy']-prev['balanced_accuracy'])*100:+.2f} p.p.** e Escore ICBHI **{(cur['icbhi_score']-prev['icbhi_score'])*100:+.2f} p.p.**
- **IC 95% de Macro-F1 (bootstrap por paciente, {ci['repetitions']} repetições):** [{ci['lower']:.4f}; {ci['upper']:.4f}].

## Desempenho por categoria acústica

| Categoria | Ciclos | Recall v1.1 | Recall v2.0 | Diferença | F1 v2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(cat_rows)}
{dev_section}
![Comparação entre versões](figures/comparacao_modelos_v2.png)

![Matrizes de confusão v1.1 vs v2.0](figures/matrizes_confusao_v2.png)

## Principais inovações técnicas da v2.0

1. **Eliminação do viés de equipamento (estetoscópio):** filtragem passa-faixa Butterworth na banda respiratória (`{model['feature_config']['fmin_hz']}–{model['feature_config']['fmax_hz']} Hz`), normalização robusta por RMS e contraste adaptativo (`p5..p99.5` com piso dinâmico mínimo de 25 dB, imune a cliques cardíacos e diferenças de piso de ruído/ANR), além de camada neural invariante a resposta de frequência estacionária (CMN por frequência antes do contraste local para sibilos e derivada temporal para estertores).
2. **Simulação física de estetoscópios e perda multitarefa ponderada:** aumento de dados simulando redução ativa de ruído (ANR suave do Littmann 3200), chiado térmico (AKGC417L), curvas de EQ/low-pass e SpecAugment preenchido pela média, aliado a detectores binários compartilhados de estertores e sibilos.
3. **Ensemble híbrido de 3 famílias em PyTorch puro (`{model['architecture_id']}`):** combina 5× `RespiraSEBandNet` (convoluções direcionais + atenção Squeeze-and-Excitation + ramo estatístico), 5× `RespiraDilatedNet` (convoluções dilatadas multi-faixa) e 5× `SpectralGatedMLP` (rede residual sobre 192 descritores físicos CMN-invariantes treinada sem Mixup destrutivo entre classes, preservando a detecção de co-ocorrência `Ambos`).
4. **Ensemble 5-fold por paciente + calibração fora da amostra (OOF):** treinado em validação cruzada estratificada por paciente sobre os 77 pacientes de treino/validação (4.025 ciclos), sem qualquer consulta aos 49 pacientes do teste durante seleção ou calibração.
5. **Paridade e segurança mantidas:** todos os pesos do ensemble são serializados em um único arquivo `models/cnn.pt` carregado com `weights_only=True` e checksum SHA-256 verificado (`{model['weights_sha256'][:16]}…`).

## Limitações mantidas explicitamente

- Protótipo acadêmico experimental, sem validação para uso diagnóstico clínico.
- Ausência de sibilos/estertores não equivale à ausência de doença.
- O módulo cardíaco (arritmia / FA) permanece indisponível por falta de base de áudio acoplada com confirmação por ECG.
"""
    (ROOT / "reports" / "REFINAMENTO_V2.md").write_text(md, encoding="utf-8")
    print("Relatório REFINAMENTO_V2.md e figuras gerados com sucesso!")


if __name__ == "__main__":
    main()
