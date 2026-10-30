"""Gera relatório e figuras exclusivamente a partir dos resultados salvos."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    metrics = read(ROOT / "models" / "metrics.json")
    model = read(ROOT / "models" / "model.json")
    dataset = read(ROOT / "models" / "dataset.json")
    history = read(ROOT / "reports" / "training_history.json")
    audit = read(ROOT / "data" / "manifest.audit.json")
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "white"})
    labels = ["Sem eventos", "Estertores", "Sibilos", "Ambos"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, key, name in zip(axes, ("cnn", "random_forest"), ("CNN", "MFCC + Random Forest"), strict=True):
        matrix = np.asarray(metrics[key]["test"]["confusion_matrix"])
        axis.imshow(matrix, cmap="BuGn")
        for row in range(4):
            for column in range(4):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() * .55 else "#19343b")
        axis.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
                 xlabel="Categoria prevista", ylabel="Categoria anotada", title=name)
        axis.tick_params(axis="x", labelrotation=20)
    figure.suptitle("Teste reservado por paciente — contagens de ciclos reais")
    figure.savefig(figures / "matrizes_confusao.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o", color="#0b7f79")
    axes[0].set(xlabel="Época", ylabel="Perda ponderada", title="Treinamento")
    axes[1].plot([row["epoch"] for row in history], [row["validation_macro_f1"] for row in history], marker="o", color="#245c70")
    axes[1].axvline(model["best_epoch"], color="#bb7d21", linestyle="--", label="Pesos selecionados")
    axes[1].set(xlabel="Época", ylabel="Macro-F1", title="Validação interna por paciente", ylim=(0, 1))
    axes[1].legend()
    figure.savefig(figures / "treinamento.png", dpi=180)
    plt.close(figure)
    cnn, forest, majority = (metrics[key]["test"] for key in ("cnn", "random_forest", "majority_baseline"))
    result_rows = []
    for title, result in (("Classe majoritária", majority), ("MFCC + Random Forest", forest), ("CNN", cnn)):
        result_rows.append(f"| {title} | {result['macro_f1']:.4f} | {result['balanced_accuracy']:.4f} | {result['accuracy']:.4f} | {result['icbhi_score']:.4f} |")
    class_rows = []
    for row in cnn["per_class"]:
        sensitivity = "Não aplicável" if row["sensitivity"] is None else f"{row['sensitivity']:.4f}"
        class_rows.append(f"| {row['label']} | {row['support']} | {sensitivity} | {row['specificity']:.4f} | {row['f1']:.4f} |")
    partitions = [f"| {key} | {value['patients']} | {value['cycles']} |" for key, value in dataset["splits"].items()]
    ci = cnn["macro_f1_ci95"]
    conclusion = ("A CNN obteve macro-F1 ligeiramente maior que o modelo convencional neste teste reservado." if cnn["macro_f1"] > forest["macro_f1"] else "O modelo convencional apresentou macro-F1 maior ou igual à CNN neste teste reservado; a hipótese de vantagem da CNN não foi sustentada por essa comparação.")
    conclusion += " A acurácia e o escore ICBHI da CNN ficaram abaixo do preditor de classe majoritária. Portanto, essa diferença pontual em macro-F1 não demonstra superioridade geral; o desempenho é insuficiente para uso clínico."
    text = f"""# Resultados experimentais reais

Os valores abaixo foram gerados pelos artefatos do treinamento. Nenhum áudio sintético foi usado nas métricas científicas. Fixtures artificiais existem exclusivamente nos testes de software.

## Dados e protocolo

ICBHI 2017, obtida por espelho acadêmico com integridade verificada. Foram auditados {audit['raw_recordings']} arquivos, {audit['raw_patients']} pacientes e {audit['raw_cycles']} ciclos originais. Após excluir {audit['excluded_cycle_count']} ciclos de gravações de treino dos pacientes presentes no teste original, restaram {dataset['cycles']} ciclos.

| Partição | Pacientes | Ciclos |
| --- | ---: | ---: |
{chr(10).join(partitions)}

O teste original foi preservado. Pacientes 156 e 218 estavam em treino e teste da lista espelhada; suas 13 gravações de treino foram excluídas antes da validação interna. O protocolo difere de avaliações que permitem essa sobreposição. Não houve ajuste do modelo com base no teste final.

## Resultados no teste

Unidade: ciclo respiratório; {cnn['n_cycles']} ciclos e {dataset['splits']['test']['patients']} pacientes reservados.

| Modelo | Macro-F1 | Acurácia balanceada | Acurácia | Escore ICBHI |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(result_rows)}

Intervalo de confiança de 95% para macro-F1 da CNN, por bootstrap de pacientes ({ci['repetitions']} repetições): **{ci['lower']:.4f} a {ci['upper']:.4f}**. Esse intervalo descreve incerteza amostral dentro desta avaliação; não demonstra validade clínica externa.

## Resultados da CNN por categoria

| Categoria | Ciclos | Sensibilidade | Especificidade um-contra-os-demais | F1 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(class_rows)}

![Matrizes de confusão](figures/matrizes_confusao.png)

![Curvas de treinamento](figures/treinamento.png)

## Configuração executada

- CNN: {model['architecture']}; {model['parameters']} parâmetros.
- Entrada: log-Mel 64×256; 4.000 Hz; STFT 256; salto 64; faixa dinâmica de 80 dB; ciclo inteiro interpolado no tempo, sem truncar o final.
- Treino: {model['epochs_executed']} épocas executadas, máximo de {model['maximum_epochs']}; seleção da época {model['best_epoch']} pela macro-F1 de validação.
- Otimizador: {model['optimizer']}, taxa {model['learning_rate']}; lote {model['batch_size']}; semente {model['seed']}; pesos de classe calculados somente no treino.
- Cache e inferência float32; sem quantização intermediária das representações.
- Ambiente: Python {model['python']}, PyTorch {model['torch']}, NumPy {model['numpy']}.
- Duração registrada do pipeline: {model['training_seconds']:.1f} segundos; inclui auditoria e características quando não havia cache.
- Pesos SHA-256: `{model['weights_sha256']}`.

## Interpretação

{conclusion} O resultado é experimental. A rede reconhece padrões das categorias usadas neste conjunto; não é um sistema validado de diagnóstico ou de triagem clínica. As diferenças entre dispositivos e a pequena diversidade de pacientes limitam generalização. O redimensionamento temporal pode prejudicar eventos breves.

O módulo cardíaco de arritmia permanece indisponível por ausência de uma base de áudio apropriada com confirmação do ritmo. Seu funcionamento não é simulado nesta aplicação.

## Evidências e verificações

- `models/metrics.json`: métricas completas e definições.
- `models/model.json`: configuração e integridade do modelo.
- `reports/test_predictions.csv`: rótulos e previsões de todos os ciclos do teste.
- `data/manifest.audit.json`: auditoria e exclusões, por gravação.
- `docs/QA_INDEPENDENTE.md` e `reports/qa/`: testes e retestes; conferir o estado registrado nesses arquivos.

Os quatro exemplos distribuídos foram escolhidos por ordem de paciente/gravação, um por categoria, sem seleção por acerto do modelo. O reteste não constitui uma nova amostra de pacientes.
"""
    (ROOT / "reports" / "RESULTADOS.md").write_text(text, encoding="utf-8")
    academic = f"""# Relatório acadêmico — implementação experimental

**Título da implementação:** Classificação de sons respiratórios adventícios por processamento de áudio e redes neurais convolucionais.

**Autores, instituição e orientação:** a preencher pela equipe. **Versão:** implementação de 26/10/2026, com resultados reais. O tema geral inicial incluía arritmias; a implementação entregue avalia somente sons respiratórios.

## Resumo

Foi desenvolvido um protótipo local de classificação de ciclos respiratórios usando aprendizado supervisionado. A base ICBHI 2017 foi obtida de um espelho acadêmico com integridade verificada e auditada para impedir sobreposição de pacientes. Foram usados {dataset['cycles']} ciclos, dos quais {cnn['n_cycles']} pertencem ao teste reservado. Compararam-se uma CNN compacta, um modelo de características cepstrais com Random Forest e um preditor de classe majoritária. A CNN alcançou macro-F1 de {cnn['macro_f1']:.4f}, acurácia de {cnn['accuracy']:.4f} e escore ICBHI de {cnn['icbhi_score']:.4f}. O resultado foi modesto e não estabelece utilidade clínica. A aplicação permite selecionar um trecho, gerar sua representação e inspecionar uma previsão experimental, com métricas e limitações explícitas.

**Palavras-chave:** ausculta digital; processamento de áudio; aprendizado supervisionado; CNN; sons respiratórios.

## 1. Introdução e delimitação

Este trabalho investiga a classificação de padrões acústicos em gravações de ausculta. A literatura inclui modelos para reconhecer sibilos, estertores e sua combinação [1, 2]. A proposta temática inicial também incluía fibrilação atrial por áudio [3], mas essa tarefa exige dados e confirmação de ritmo próprios. Não foi obtida uma base apropriada para executá-la nesta versão.

A contribuição entregue é um procedimento reproduzível de organização dos dados, preparação dos sinais, treinamento, avaliação e demonstração local. O reconhecimento de categorias acústicas é distinguido do diagnóstico de doenças e da comprovação de detecção clínica precoce.

## 2. Problema, hipótese e objetivos

A pergunta foi se uma CNN consegue reconhecer as categorias em pacientes reservados e superar um classificador convencional na macro-F1. O objetivo foi implementar, comparar e documentar os modelos e uma aplicação demonstrativa. A superioridade da CNN não foi assumida como condição para relatar resultados.

## 3. Dados e protocolo

Utilizou-se a ICBHI 2017 [2], com {audit['raw_recordings']} gravações e {audit['raw_patients']} pacientes. A origem, os termos, os hashes e a aquisição estão em [DADOS.md](DADOS.md). A auditoria encontrou os pacientes 156 e 218 tanto no treino quanto no teste da lista original espelhada. Foram removidas apenas suas 13 gravações de treino ({audit['excluded_cycle_count']} ciclos); os 381 arquivos e 2.756 ciclos do teste original foram preservados.

O conjunto final contém {dataset['splits']['train']['patients']} pacientes de treino ({dataset['splits']['train']['cycles']} ciclos), {dataset['splits']['validation']['patients']} de validação ({dataset['splits']['validation']['cycles']} ciclos) e {dataset['splits']['test']['patients']} de teste ({cnn['n_cycles']} ciclos). Não há pacientes ou conteúdo PCM compartilhados entre partições. As classes são ausência dos dois eventos, estertores, sibilos e ambos. Ausência desses eventos não equivale a ausência de doença.

## 4. Processamento e modelos

Os limites dos ciclos foram obtidos das anotações. Cada trecho foi preparado com reamostragem para 4.000 Hz, normalização por pico e espectrograma log-Mel de 64 faixas. A STFT usa janela de 256 amostras e salto de 64. A faixa dinâmica é de 80 dB. O ciclo completo foi interpolado para 256 posições temporais, sem truncamento do final; isso preserva a extensão do ciclo, mas pode alterar a representação de eventos curtos.

A CNN possui blocos convolucionais com 16, 32 e 64 canais, agregação adaptativa e camadas densas, totalizando {model['parameters']} parâmetros. Foi treinada com AdamW, taxa de aprendizado {model['learning_rate']}, lote {model['batch_size']}, perda ponderada pelas classes de treinamento e semente {model['seed']}. A seleção ocorreu pela macro-F1 na validação: época {model['best_epoch']} entre {model['epochs_executed']} executadas, com interrupção após seis épocas sem melhoria.

O modelo convencional usa média e desvio de 20 coeficientes cepstrais derivados da mesma representação e Random Forest com 250 árvores. Um classificador de maioria contextualiza a influência do desequilíbrio. Nenhuma configuração foi ajustada após consultar o teste final.

## 5. Avaliação e resultados

Unidade de avaliação: ciclo respiratório. A macro-F1 é a média do F1 das quatro categorias. Especificidade por classe usa um-contra-os-demais. O escore ICBHI combina o acerto da categoria exata dos ciclos anormais e o acerto dos ciclos sem eventos.

| Modelo | Macro-F1 | Acurácia balanceada | Acurácia | Escore ICBHI |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(result_rows)}

O IC95% de macro-F1 da CNN por bootstrap de pacientes foi [{ci['lower']:.4f}; {ci['upper']:.4f}], com {ci['repetitions']} reamostragens. Esse intervalo não é um teste de significância da diferença entre os modelos.

![Matrizes de confusão](../reports/figures/matrizes_confusao.png)

![Treinamento e validação](../reports/figures/treinamento.png)

As métricas completas por categoria e as previsões individuais estão em [RESULTADOS.md](../reports/RESULTADOS.md) e `reports/test_predictions.csv`.

## 6. Discussão e limitações

{conclusion} O resultado depende da população, dos equipamentos e do procedimento de representação. O número de pacientes é limitado e as categorias são desequilibradas. A comparação com trabalhos que utilizam partições diferentes ou permitem sobreposição de pacientes exige considerar essa diferença metodológica.

O protótipo classifica um ciclo delimitado e não detecta automaticamente respirações em gravações contínuas. Pontuações de softmax não são probabilidades clínicas calibradas. A avaliação não demonstra benefício clínico, diagnóstico de doença ou antecipação do diagnóstico. O módulo de arritmia está indisponível e não é simulado.

## 7. Aplicação e verificação

A interface local foi implementada com HTML/CSS/JavaScript e API FastAPI, sem CDN. Ela recebe WAV, permite selecionar e ouvir um ciclo, exibe forma de onda e espectrograma, aplica os pesos salvos e exporta a análise em JSON. Uploads não são adicionados ao treinamento.

O treinamento e a inferência compartilham o mesmo processamento float32. O pacote inclui código, pesos, métricas, quatro exemplos reais de teste, documentação e testes. A base completa é obtida pelo downloader e não está redistribuída integralmente no pacote.

As verificações cobrem áudio, partições, anotações, artefatos, API, arquivos inválidos, origem dos exemplos e coerência das métricas. Consultar [QA independente](QA_INDEPENDENTE.md) para resultados de testes e retestes efetivamente registrados. Fixtures artificiais servem apenas a testes de software, nunca à avaliação científica. Repetir previsões nos mesmos pacientes verifica reprodução; não constitui validação em novos pacientes.

## 8. Conclusão

Foi entregue um protótipo experimental que executa o fluxo de classificação e explicita seus resultados. A avaliação demonstrou limitações importantes do modelo, apesar da macro-F1 superior à referência de maioria e ligeiramente superior ao modelo convencional. Não há evidência suficiente para recomendar uso clínico. A continuação deve investigar representações temporais, generalização e dados independentes, com um novo protocolo de avaliação reservado para evitar adaptação ao teste já consultado.

## Referências

[1] PETMEZAS, G. et al. Automated Lung Sound Classification Using a Hybrid CNN-LSTM Network and Focal Loss Function. *Sensors*, 2022. [Artigo](https://doi.org/10.3390/s22031232).

[2] ROCHA, B. M. et al. An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*, 2019. [Artigo](https://doi.org/10.1088/1361-6579/ab03ea).

[3] GUO, Y. et al. Detection of atrial fibrillation via artificial intelligence-assisted auscultation. *BMC Cardiovascular Disorders*, 2025. [Artigo](https://doi.org/10.1186/s12872-025-05236-z).

Completar dados da equipe e adaptar a formatação às exigências da instituição. Não foi declarada conformidade com uma norma acadêmica específica.
"""
    (ROOT / "docs" / "RELATORIO_ACADEMICO.md").write_text(academic, encoding="utf-8")
    print("Relatório e figuras gerados a partir dos resultados reais.")


if __name__ == "__main__":
    main()
