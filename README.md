# Ausculta Lab — aplicação acadêmica local

Classificação de ciclos respiratórios com processamento de áudio e uma CNN, acompanhada de modelo convencional, resultados rastreáveis e testes.

**Versão 2.0.0:** novo modelo **Ensemble RespiraNet v2.0** invariante a estetoscópios (`logmel-bp-v3` 80–1.200 Hz + 5× `RespiraSEBandNet` + 5× `RespiraDilatedNet` + 5× `SpectralGatedMLP`, empacotado em um único `models/cnn.pt` carregado com `weights_only=True`) e **interface web local redesenhada** (100% offline, sem CDN). No teste reservado por paciente (49 pacientes · 2.756 ciclos, sem vazamento), a v2.0 evoluiu frente à v1.1 em **acurácia (45,46% → 55,12%, +9,65 p.p.)**, **Macro-F1 (0,3611 → 0,4034, +0,0423)**, **acurácia balanceada (37,47% → 39,80%, +2,33 p.p.)** e **escore ICBHI (44,50% → 52,91%, +8,41 p.p.)**, praticamente dobrando o acerto no estetoscópio Littmann 3200 (22,13% → 43,38%). Consulte o [relatório completo da versão 2.0](reports/REFINAMENTO_V2.md) e o [histórico da versão 1.1](reports/REFINAMENTO_CNN.md).

**Destaques da interface v2.0 (`web/`):**
- Forma de onda interativa com seleção por arraste, alças móveis, *playhead* animado ao vivo e ajuste fino (±0,1 s).
- Estúdio de escuta via Web Audio API: filtro pulmonar em tempo real (80–1.600 Hz), repetição em *loop*, ganho 2×, reprodução lenta (0,75×) e sugestão automática de ciclos em gravações longas.
- Carregamento em 1 clique por categoria, comparação automática com a referência anotada nos 16 exemplos, indicadores independentes de estertores e sibilos, abas de **Espectrograma**, **Mapa de contraste acústico** e **Detalhes técnicos**, histórico da sessão e exportação/cópia do resumo.
- Painel **Desempenho & Validação Científica** com gráfico comparativo interativo (Maioria, RF, CNN v1.0, CNN v1.1 e Ensemble v2.0), matriz de confusão interativa e tabela por categoria.

**Escopo entregue:** pulmonar, quatro categorias acústicas. O módulo de arritmia não está implementado e aparece como indisponível. O modelo é experimental e não possui validação clínica. O planejamento inicial de um mês foi mantido na documentação como contexto; a demonstração foi implementada como aplicação web local, sem hospedagem externa.

## Iniciar no Windows

1. Extraia o ZIP completo para uma **pasta curta**, por exemplo `C:\AuscultaLab` ou `C:\Users\SEU_USUARIO\AuscultaLab`. Não execute dentro do ZIP. O instalador detecta caminhos de projeto acima de 90 caracteres, pois bibliotecas do PyTorch podem ultrapassar o limite de caminhos deste Windows.
2. Tenha Python **3.13 de 64 bits** instalado (ambiente efetivamente verificado).
3. Execute `INSTALAR.bat` na primeira utilização. A instalação requer internet e baixa bibliotecas, incluindo PyTorch.
4. Execute `INICIAR.bat`. A interface abre em **http://127.0.0.1:8765**.
5. Mantenha a janela do servidor aberta. Use Ctrl+C para encerrar.

Após instalar, a classificação funciona localmente com os pesos distribuídos e os 16 exemplos reais. A demonstração usa os arquivos que acompanham o pacote.

Se a porta estiver ocupada: `powershell -NoProfile -ExecutionPolicy Bypass -File .\Iniciar.ps1 -Port 8767`.

## Uso

Carregue um exemplo ou um WAV de ausculta mono ou estéreo de até 20 MB e 120 segundos. Ouça o áudio, delimite um ciclo de 0,1 a 20 segundos, confirme a seleção e clique em analisar. Alterar o trecho invalida a previsão anterior. As pontuações descrevem a preferência do modelo entre as quatro classes.

Os exemplos trazem o rótulo anotado para comparação. Foram escolhidos por uma regra fixa, sem selecionar os que o modelo acerta. A aplicação não envia uploads a serviços externos nem os mantém como novos dados de treinamento.

Mensagens em OGG ou Opus recebem uma instrução para escolher um WAV de ausculta. A análise aceita um WAV tecnicamente válido de fala e pode gerar uma categoria fora do escopo. A confirmação do trecho orienta o usuário a utilizar somente ausculta respiratória.

## Executar testes e reteste

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --require-real-artifacts
.\.venv\Scripts\python.exe scripts\retest.py
```

O primeiro comando exige os artefatos reais incluídos no pacote. Fixtures sintéticas são usadas somente nos testes de software, nunca nas métricas científicas. As evidências de ambas as rodadas ficam em `reports/qa/`.

## Documentos

- [Índice da documentação](docs/README.md): organização dos documentos, cronograma de escrita e critérios de revisão.
- [Como funciona](docs/COMO_FUNCIONA.md): explicação da obtenção dos áudios, aprendizado, avaliação e uso do modelo.
- [PROJETO.md](PROJETO.md): título, problema, objetivos, hipótese, metodologia, dados, avaliação, cronograma e referências.
- [Relatório acadêmico](docs/RELATORIO_ACADEMICO.md): texto-base para o trabalho e estrutura para métodos, resultados e discussão.
- [Documentação técnica](docs/DOCUMENTACAO_TECNICA.md): arquitetura, dados, modelos, configuração e reprodução.
- [Manual de uso](docs/MANUAL_USO.md): roteiro previsto da demonstração e instruções a conferir após a implementação.
- [Ficha de experimento](docs/REGISTRO_EXPERIMENTO.md): registro completo de cada execução real.
- [Experimento 001 executado](docs/EXPERIMENTO_001.md): ficha preenchida com configuração, dados, resultados e decisões.
- [Dados e proveniência](docs/DADOS.md): aquisição, integridade, termos e correção das partições.
- [Ensemble RespiraNet v2.0](reports/REFINAMENTO_V2.md): relatório científico da versão 2.0, métricas por categoria/estetoscópio e figuras comparativas.
- [CNN por faixas (v1.1)](reports/REFINAMENTO_CNN.md): comparação de acertos por categoria, decisões do experimento e teste com fala.
- [Resultados da versão 1.0](reports/RESULTADOS.md): métricas, figuras e interpretação do experimento original.
- [QA independente](docs/QA_INDEPENDENTE.md): testes, achados e retestes.
- [docs/PLANO_IMPLEMENTACAO.md](docs/PLANO_IMPLEMENTACAO.md): tarefas e critérios de aceite para construir o protótipo.
- [docs/manifesto.template.csv](docs/manifesto.template.csv): cabeçalho do futuro catálogo de segmentos; contém somente o cabeçalho, sem pacientes ou exemplos fictícios.
- [docs/experimentos.template.csv](docs/experimentos.template.csv): cabeçalho para registrar experimentos reais; ainda sem resultados.

## Organização

```text
projeto-triagem-cardiopulmonar/
|-- PROJETO.md
|-- README.md
|-- docs/
|   |-- README.md
|   |-- COMO_FUNCIONA.md
|   |-- PLANO_IMPLEMENTACAO.md
|   |-- RELATORIO_ACADEMICO.md
|   |-- DOCUMENTACAO_TECNICA.md
|   |-- MANUAL_USO.md
|   |-- REGISTRO_EXPERIMENTO.md
|   |-- manifesto.template.csv
|   `-- experimentos.template.csv
|-- data/
|   |-- raw/          # arquivos originais, após obtenção autorizada
|   `-- processed/    # dados derivados
|-- src/cardio_audio/ # dados, áudio, modelos, métricas, inferência e API
|-- scripts/         # aquisição, treino, relatório e reteste
|-- web/             # interface sem CDN
|-- tests/           # testes unitários e integração real
|-- examples/        # 16 ciclos reais de teste e proveniência
|-- models/          # pesos, modelo convencional e resultados
|-- requirements.lock.txt
|-- INICIAR.bat
`-- reports/         # previsões, métricas, gráficos e evidências
```

## Reproduzir os experimentos

Para reproduzir a **versão 2.0** (validação cruzada 5-fold por paciente nos 77 pacientes de treino/validação, calibração OOF e avaliação única nos 49 pacientes de teste):

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\train_v2.py --seed 42
.\.venv\Scripts\python.exe scripts\write_report_v2.py
.\.venv\Scripts\python.exe scripts\retest.py --full
```

O refinamento da versão 1.1 usa `scripts/refine_model.py` (protocolo em [REFINAMENTO_CNN.md](reports/REFINAMENTO_CNN.md)).

Para refazer a versão 1.0, use uma cópia separada. Obtenha a base seguindo [DADOS.md](docs/DADOS.md), preservando a origem e as condições de pesquisa. Depois:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\train.py --epochs 18 --batch-size 32 --seed 42
.\.venv\Scripts\python.exe scripts\write_report.py
.\.venv\Scripts\python.exe scripts\retest.py --full
```

O treino regenera os artefatos. Faça uma cópia dos pesos e relatórios distribuídos antes de repetir o experimento. O modo `--full` exige a base/cache locais; o reteste comum funciona com o pacote de entrega. O download completo ocupa aproximadamente 2 GB compactados, além dos arquivos extraídos e do cache.

O teste original foi preservado, mas 13 gravações de treino (117 ciclos) de dois pacientes presentes no teste foram excluídas para impedir vazamento. A comparação com trabalhos que usam outra divisão requer considerar essa diferença.

O planejamento não pressupõe treinamento conjunto dos sinais do coração e do pulmão ou gravações dos mesmos pacientes nas duas bases.

## Planejamento inicial de um mês

1. Semana 1: dados, auditoria, separação por paciente e espectrogramas; introdução, objetivos e descrição dos dados.
2. Semana 2: modelo convencional e primeira CNN; métodos, ambiente e registros de treinamento.
3. Semana 3: seleção pela validação, teste reservado e análise de erros; resultados, gráficos e ficha do modelo.
4. Semana 4: demonstração, relatório e apresentação; manual conferido e revisão de reprodução.

A interface entregue é local e serve à demonstração do experimento. Aplicativo móvel, coleta própria, detecção automática de ciclos e validação clínica permanecem fora desta versão.
