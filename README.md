# Ausculta Lab — aplicação acadêmica local

Classificação de ciclos respiratórios com processamento de áudio e uma CNN, acompanhada de modelo convencional, resultados rastreáveis e testes.

**Implementado:** interface local para WAV, seleção manual do ciclo, player, forma de onda, CNN, espectrograma, exportação JSON e consulta de evidências. Os dados reais foram obtidos e auditados. Consulte [resultados](reports/RESULTADOS.md) e [verificações independentes](docs/QA_INDEPENDENTE.md) para o estado final dos experimentos e retestes.

**Escopo entregue:** pulmonar, quatro categorias acústicas. O módulo de arritmia não está implementado e aparece como indisponível. O modelo é experimental e não possui validação clínica. O planejamento inicial de um mês foi mantido na documentação como contexto; a demonstração foi implementada como aplicação web local, sem hospedagem externa.

## Iniciar no Windows

1. Extraia o ZIP completo para uma **pasta curta**, por exemplo `C:\AuscultaLab` ou `C:\Users\SEU_USUARIO\AuscultaLab`. Não execute dentro do ZIP. O instalador detecta caminhos de projeto acima de 90 caracteres, pois bibliotecas do PyTorch podem ultrapassar o limite de caminhos deste Windows.
2. Tenha Python **3.13 de 64 bits** instalado (ambiente efetivamente verificado).
3. Execute `INSTALAR.bat` na primeira utilização. A instalação requer internet e baixa bibliotecas, incluindo PyTorch.
4. Execute `INICIAR.bat`. A interface abre em **http://127.0.0.1:8765**.
5. Mantenha a janela do servidor aberta. Use Ctrl+C para encerrar.

Após instalar, a classificação funciona localmente com os pesos distribuídos e os quatro exemplos reais. A base completa não precisa ser baixada para usar a demonstração.

Se a porta estiver ocupada: `powershell -NoProfile -ExecutionPolicy Bypass -File .\Iniciar.ps1 -Port 8767`.

## Uso

Carregue um exemplo de teste ou um WAV mono/estéreo de até 20 MB e 120 segundos. Delimite um ciclo de 0,1 a 20 segundos, ouça o trecho, confirme a seleção e clique em analisar. A aplicação mostra a categoria prevista e pontuações softmax não calibradas; não são probabilidades clínicas de doença. Alterar o trecho invalida a previsão anterior.

Os exemplos trazem o rótulo anotado para comparação. Foram escolhidos por uma regra fixa, sem selecionar os que o modelo acerta. A aplicação não envia uploads a serviços externos nem os mantém como novos dados de treinamento.

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
- [Resultados reais](reports/RESULTADOS.md): métricas, figuras e interpretação.
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
|-- examples/        # quatro ciclos reais de teste e proveniência
|-- models/          # pesos, modelo convencional e resultados
|-- requirements.lock.txt
|-- INICIAR.bat
`-- reports/         # previsões, métricas, gráficos e evidências
```

## Reproduzir o experimento completo

Obtenha a base seguindo [DADOS.md](docs/DADOS.md), preservando a origem e as condições de pesquisa. Depois:

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
