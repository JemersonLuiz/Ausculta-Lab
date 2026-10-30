# Experimento 001 — execução real

**ID:** ICBHI2017-CNN-v1-seed42. **Data:** 26/10/2026. **Estado:** treinado e avaliado; verificação científica e de software registrada.

## Objetivo e dados

Comparar uma CNN compacta com MFCC + Random Forest em ciclos respiratórios de pacientes reservados. A origem e os termos estão em [DADOS.md](DADOS.md). O manifesto foi auditado e 13 gravações de treino de pacientes presentes no teste foram excluídas: 117 ciclos. O teste original permaneceu completo, sem compartilhar pacientes ou conteúdo com o desenvolvimento.

| Partição | Pacientes | Gravações | Ciclos |
| --- | ---: | ---: | ---: |
| Treino | 62 | 467 | 3.594 |
| Validação | 15 | 59 | 431 |
| Teste | 49 | 381 | 2.756 |

## Execução

Representação float32 compartilhada pelo treino e pela inferência: 4 kHz, log-Mel 64×256, STFT 256, salto 64, faixa dinâmica 80 dB e interpolação do ciclo completo. CNN com 16/32/64 canais e 154.804 parâmetros, AdamW a 0,0007, lote 32, semente 42 e perda ponderada por frequências somente de treino. Foram executadas 12 de no máximo 18 épocas, selecionando a época 6 pela macro-F1 de validação, com paciência de seis épocas.

O comparador usa média/desvio de 20 coeficientes cepstrais e Random Forest com 250 árvores. O teste não foi usado para ajustar configurações. Ambiente: Windows, Python 3.13.2, PyTorch 2.14.1+cpu e NumPy 2.5.4; demais versões no lock.

## Resultados reservados

| Modelo | Macro-F1 | Acurácia | Escore ICBHI |
| --- | ---: | ---: | ---: |
| CNN | 0,3216 | 0,4358 | 0,4210 |
| MFCC + Random Forest | 0,3098 | 0,5145 | 0,4793 |
| Classe majoritária | 0,1821 | 0,5729 | 0,5000 |

IC95% de macro-F1 da CNN por bootstrap de pacientes, 500 reamostragens: 0,2737–0,3686. A pequena diferença de macro-F1 não comprova superioridade geral ou estatística. A sensibilidade para “ambos” foi 0,1399. O desempenho não sustenta uso clínico.

## Evidências

- Configuração e pesos: `models/model.json` e `models/cnn.pt`.
- Pesos SHA-256: `aa69b383ae7b5a1cd4585a0dcc8d9e3a8ad6244cd3b84a3bb2d5db1110d03650`.
- Métricas, histórico e previsões: `models/metrics.json`, `reports/training_history.json` e `reports/test_predictions.csv`.
- Auditoria: `data/manifest.audit.json`.
- Duas rodadas de 105 testes e reteste dos 2.756 ciclos: [QA independente](QA_INDEPENDENTE.md) e `reports/qa/reteste_completo.json`. Features dentro de quatro epsilons float32; previsões exatamente reproduzidas.

## Decisão

Entregar como demonstração acadêmica com limitações e erros explícitos. O módulo cardíaco não foi implementado por falta de dados adequados. A continuação deverá prever avaliação independente para evitar adaptar o modelo ao teste já consultado. A aprovação dos testes de software não transforma esse resultado em validação clínica.
