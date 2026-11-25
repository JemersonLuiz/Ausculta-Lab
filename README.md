<div align="center">

# Ausculta Lab

**Protótipo acadêmico local para classificação de ciclos respiratórios** a partir de gravações de ausculta pulmonar, com processamento de áudio, redes neurais convolucionais e interface web totalmente offline.

[![Python](https://img.shields.io/badge/Python-3.11%20--%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Versão](https://img.shields.io/badge/vers%C3%A3o-2.0.0-blue)](#versionamento)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Testes](https://img.shields.io/badge/testes-pytest-0A9EDC?logo=pytest&logoColor=white)](tests/)
[![Plataforma](https://img.shields.io/badge/plataforma-Windows-0078D4?logo=windows&logoColor=white)](#instalação-no-windows)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#limitações)
[![Offline](https://img.shields.io/badge/interface-100%25%20offline-success)](web/)

</div>

---

## Sumário

- [Visão geral](#visão-geral)
- [Principais recursos](#principais-recursos)
- [Resultados no conjunto de teste reservado](#resultados-no-conjunto-de-teste-reservado)
- [Versionamento](#versionamento)
- [Instalação no Windows](#instalação-no-windows)
- [Uso](#uso)
- [Testes e reteste](#testes-e-reteste)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Documentação](#documentação)
- [Reprodução dos experimentos](#reprodução-dos-experimentos)
- [Limitações](#limitações)

---

## Visão geral

O Ausculta Lab classifica ciclos respiratórios em quatro categorias acústicas: **sem sibilos ou estertores**, **estertores**, **sibilos** e **ambos**. O pipeline converte o áudio em espectrogramas log-Mel filtrados na faixa pulmonar e os encaminha a um modelo de *ensemble* com cabeçalho multitarefa, que estima separadamente a presença de estertores e de sibilos.

Todo o processamento ocorre no computador local. Nenhum áudio é enviado a serviços externos ou armazenado como dado de treinamento.

O escopo entregue é **pulmonar**. O módulo de arritmia cardíaca não está implementado e aparece como indisponível na interface.

## Principais recursos

**Processamento e modelo**

- Filtragem passa-faixa na banda pulmonar e espectrogramas log-Mel (`logmel-bp-v3`, 80–1.200 Hz).
- *Ensemble* **RespiraNet v2.0** com 15 redes (5× `RespiraSEBandNet`, 5× `RespiraDilatedNet`, 5× `SpectralGatedMLP`) e cabeçalho multitarefa, com cerca de 3,57 milhões de parâmetros.
- Invariância a estetoscópios por normalização espectral (CMN de 192 dimensões).
- Pesos distribuídos em `models/cnn.pt`, carregados com `weights_only=True`.

**Interface web (`web/`)**

- Forma de onda interativa com seleção por arraste, alças móveis e *playhead* animado.
- Estúdio de escuta via Web Audio API: filtro pulmonar em tempo real, repetição em *loop*, ganho 2×, reprodução a 0,75× e sugestão automática de ciclos em gravações longas.
- Exemplos reais do conjunto de teste (16 ciclos) com comparação automática à referência anotada.
- Indicadores independentes de estertores e sibilos, abas de espectrograma, mapa de contraste acústico e detalhes técnicos.
- Painel de desempenho com comparação entre modelos, matriz de confusão e métricas por categoria.
- Histórico da sessão e exportação do resultado em JSON.
- Sem CDN: todos os recursos são servidos localmente.

**Qualidade e reprodutibilidade**

- Avaliação estrita por paciente, com o conjunto de teste reservado usado uma única vez.
- Intervalos de confiança por *bootstrap* (500 repetições, semente 42).
- Suíte de testes unitários e de integração com artefatos reais, além de relatórios de QA em `reports/qa/`.

## Resultados no conjunto de teste reservado

Avaliação no conjunto de teste reservado da base ICBHI 2017, com **2.756 ciclos de 49 pacientes**, sem vazamento entre partições.

| Métrica | v1.0 | v1.1.0 | **v2.0.0** |
|---|---:|---:|---:|
| Acurácia | 43,58% | 45,46% | **55,12%** |
| Acurácia balanceada | 32,99% | 37,47% | **39,80%** |
| Macro-F1 | 0,3216 | 0,3611 | **0,4034** |
| Escore ICBHI | 0,4210 | 0,4450 | **0,5291** |

Na versão 2.0.0, o Macro-F1 tem intervalo de confiança de 95% entre 0,3299 e 0,4595 (*bootstrap* por paciente). No estetoscópio Littmann 3200, a acurácia passou de 22,13% (v1.1.0) para 43,38% (v2.0.0).

Os relatórios completos estão em [REFINAMENTO_V2.md](reports/REFINAMENTO_V2.md), [REFINAMENTO_CNN.md](reports/REFINAMENTO_CNN.md) e [RESULTADOS.md](reports/RESULTADOS.md).

## Versionamento

O projeto segue o [Versionamento Semântico 2.0.0](https://semver.org/lang/pt-BR/), no formato `MAJOR.MINOR.PATCH`:

- **MAJOR:** mudanças incompatíveis de modelo, de pré-processamento ou de contrato da API. Alterações de pesos que mudam a previsão para o mesmo áudio entram aqui.
- **MINOR:** novas funcionalidades ou refinamentos de modelo compatíveis com a versão anterior.
- **PATCH:** correções sem alteração de comportamento científico.

A versão é mantida em três lugares, que devem permanecer sincronizados: `pyproject.toml`, `src/cardio_audio/__init__.py` e `models/model.json`. A API expõe a versão em `GET /api/status`.

| Versão | Modelo | Destaques |
|---|---|---|
| **2.0.0** | Ensemble RespiraNet v2.0 | Invariância a estetoscópios, pré-processamento `logmel-bp-v3`, cabeçalho multitarefa e interface redesenhada |
| **1.1.0** | BandCNN por faixas | Refinamento por faixas de frequência |
| **1.0** | SmallCNN e Random Forest (MFCC) | Experimento original e referências de comparação |

## Instalação no Windows

**Requisitos**

- Windows 10 ou 11, 64 bits.
- Python **3.13 de 64 bits** (ambiente em que a versão foi verificada). São aceitas versões de 3.11 a 3.14.
- Conexão com a internet apenas na instalação, para baixar as bibliotecas, incluindo o PyTorch.

**Passos**

1. Extraia o projeto para uma **pasta curta**, por exemplo `C:\AuscultaLab`. Não execute a partir de dentro do ZIP. O instalador alerta sobre caminhos acima de 90 caracteres, pois o PyTorch pode ultrapassar o limite de caminhos do Windows.
2. Execute `INSTALAR.bat` na primeira utilização.
3. Execute `INICIAR.bat`. A interface abre em **http://127.0.0.1:8765**.
4. Mantenha a janela do servidor aberta. Use `Ctrl+C` para encerrar.

Se a porta 8765 estiver ocupada, use outra porta:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Iniciar.ps1 -Port 8767
```

Após a instalação, a classificação funciona sem internet, com os pesos e os exemplos que acompanham o pacote.

## Uso

1. Escolha um dos 16 exemplos reais ou envie um arquivo WAV de ausculta, mono ou estéreo, de até **20 MB** e **120 segundos**.
2. Ouça a gravação e delimite um ciclo de **0,1 a 20 segundos** pela forma de onda ou pelos campos de início e fim.
3. Confirme que o trecho contém apenas ausculta respiratória e clique em **Analisar ciclo respiratório**.

Alterar o trecho invalida a previsão anterior. As pontuações indicam a preferência do modelo entre as quatro classes, não uma probabilidade clínica.

**Atalhos de teclado:** `Espaço` ouve o trecho, `Enter` analisa e `[` / `]` trocam o exemplo.

**Restrições de entrada:** arquivos OGG/Opus, como mensagens de voz do WhatsApp, são recusados. Um WAV tecnicamente válido de fala pode ser aceito, mas o resultado estará fora do escopo do modelo.

## Testes e reteste

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --require-real-artifacts
.\.venv\Scripts\python.exe scripts\retest.py
```

O primeiro comando exige os artefatos reais incluídos no pacote. Fixtures sintéticas são usadas apenas nos testes de software, nunca nas métricas científicas. As evidências das rodadas de QA ficam em `reports/qa/`.

## Estrutura do projeto

```text
Ausculta-Lab/
├── src/
│   └── cardio_audio/        # Pacote Python
│       ├── api.py           # Servidor FastAPI e rotas /api/*
│       ├── audio.py         # Leitura de WAV, filtragem e espectrogramas
│       ├── data.py          # Manifesto, partições e exemplos
│       ├── metrics.py       # Métricas, intervalos de confiança e ICBHI
│       ├── model.py         # Arquiteturas da rede neural
│       └── predictor.py     # Inferência e carregamento dos pesos
├── web/                     # Interface web offline
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── signal-orb.svg
├── models/                  # Pesos e artefatos da versão ativa
│   ├── cnn.pt               # Ensemble RespiraNet v2.0
│   ├── model.json           # Metadados e pré-processamento
│   ├── metrics.json         # Métricas de validação e teste
│   ├── baseline.joblib      # Modelo convencional (Random Forest)
│   └── dataset.json
├── examples/                # 16 ciclos reais do teste e proveniência
├── scripts/                 # Download, treino, relatórios e reteste
│   ├── download_data.py
│   ├── train_v2.py
│   ├── write_report_v2.py
│   ├── retest.py
│   └── ...
├── tests/                   # Testes unitários e de integração
├── reports/                 # Relatórios, figuras e evidências de QA
│   ├── REFINAMENTO_V2.md
│   ├── REFINAMENTO_CNN.md
│   ├── RESULTADOS.md
│   ├── figures/
│   └── qa/
├── docs/                    # Documentação técnica, acadêmica e de experimentos
├── data/                    # manifest.csv e auditoria (bases brutas não são versionadas)
├── INSTALAR.bat · Instalar.ps1
├── INICIAR.bat  · Iniciar.ps1
├── run.py                   # Ponto de entrada do servidor
├── pyproject.toml           # Metadados e dependências
├── requirements.lock.txt    # Versões fixadas do ambiente
├── PROJETO.md               # Título, problema, metodologia e cronograma
└── README.md
```

## Documentação

| Documento | Conteúdo |
|---|---|
| [Índice da documentação](docs/README.md) | Organização e critérios de revisão |
| [Como funciona](docs/COMO_FUNCIONA.md) | Obtenção dos áudios, aprendizado, avaliação e uso |
| [PROJETO.md](PROJETO.md) | Problema, objetivos, hipótese, metodologia e referências |
| [Documentação técnica](docs/DOCUMENTACAO_TECNICA.md) | Arquitetura, dados, modelos, configuração e reprodução |
| [Manual de uso](docs/MANUAL_USO.md) | Roteiro da demonstração |
| [Dados e proveniência](docs/DADOS.md) | Aquisição, integridade, termos e correção das partições |
| [Relatório acadêmico](docs/RELATORIO_ACADEMICO.md) | Texto-base para o trabalho |
| [Ensemble RespiraNet v2.0](reports/REFINAMENTO_V2.md) | Relatório científico da versão 2.0 |
| [CNN por faixas (v1.1)](reports/REFINAMENTO_CNN.md) | Refinamento da versão 1.1 |
| [Resultados da v1.0](reports/RESULTADOS.md) | Métricas e interpretação do experimento original |
| [Experimento 001](docs/EXPERIMENTO_001.md) | Ficha preenchida com configuração, dados e decisões |
| [QA independente](docs/QA_INDEPENDENTE.md) | Testes, achados e retestes |

## Reprodução dos experimentos

**Versão 2.0.0**, com validação cruzada 5-fold por paciente nos 77 pacientes de treino e validação, calibração *out-of-fold* e avaliação única nos 49 pacientes de teste:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\train_v2.py --seed 42
.\.venv\Scripts\python.exe scripts\write_report_v2.py
.\.venv\Scripts\python.exe scripts\retest.py --full
```

**Versão 1.1.0:** `scripts\refine_model.py`, conforme o protocolo em [REFINAMENTO_CNN.md](reports/REFINAMENTO_CNN.md).

**Versão 1.0:** use uma cópia separada do projeto. Obtenha a base seguindo [DADOS.md](docs/DADOS.md) e depois execute:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\train.py --epochs 18 --batch-size 32 --seed 42
.\.venv\Scripts\python.exe scripts\write_report.py
.\.venv\Scripts\python.exe scripts\retest.py --full
```

O treino regenera os artefatos. Faça uma cópia dos pesos e relatórios distribuídos antes de repetir um experimento. O modo `--full` exige a base e o cache locais. O download completo ocupa cerca de 2 GB compactados, além dos arquivos extraídos.

**Observação sobre a partição:** 13 gravações de treino (117 ciclos) de dois pacientes que também aparecem no teste foram excluídas para impedir vazamento. Comparações com trabalhos que usam outra divisão devem considerar essa diferença.

## Limitações

- **Protótipo experimental, sem validação clínica.** Os resultados não devem orientar diagnóstico ou conduta médica.
- Escopo restrito a quatro categorias pulmonares. O módulo cardíaco não está implementado.
- Os dados de treino e teste vêm de uma única base pública (ICBHI 2017), com 49 pacientes no teste.
- Desempenho variável entre estetoscópios: a acurácia no Meditron (62,53%) é maior que no AKGC417L (56,21%) e no Littmann 3200 (43,38%).
- A interface é local e não possui autenticação, pois foi concebida para demonstração acadêmica.

---

**Mantenedor:** Jemerson Luiz · Projeto acadêmico
