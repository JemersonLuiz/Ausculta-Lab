# Documentação técnica — Ausculta Lab

**Instalação em Windows:** extrair em caminho curto. A verificação de entrega encontrou falha real do pip/PyTorch em uma extração muito profunda. Instalar.ps1 agora verifica o comprimento do diretório antes de instalar, com limite conservador de 90 caracteres e orientação para uma pasta curta. Nenhuma chave de registro ou política do Windows é modificada.

Documentação da implementação local. O código efetivo substitui os nomes de módulos sugeridos no planejamento inicial. Resultados numéricos e estado das execuções devem ser lidos nos artefatos de treinamento e QA, sem interpretar a existência de um teste como evidência de que foi aprovado.

## 1. Componentes implementados

| Componente | Responsabilidade |
| --- | --- |
| src/cardio_audio/data.py | Descoberta WAV/TXT, rótulos, auditoria e partições por paciente |
| src/cardio_audio/audio.py | Validação de áudio, seleção do trecho, log-Mel, resumo cepstral e imagem |
| src/cardio_audio/model.py | SmallCNN com quatro saídas |
| src/cardio_audio/predictor.py | Conferência de configuração/hash, carga de pesos e inferência |
| src/cardio_audio/metrics.py | Métricas multiclasse, escore ICBHI e bootstrap por paciente |
| src/cardio_audio/api.py | API FastAPI e disponibilização da interface local |
| web/ | Interface HTML/CSS/JavaScript, seleção de arquivo/ciclo e apresentação dos resultados |
| scripts/download_data.py | Obtenção HTTPS com checksum, retomada por partes e extração verificada |
| scripts/train.py | Treinamento, seleção, avaliação, metadados e exportação dos exemplos |
| scripts/retest.py | Repetição da integração e comparação com o teste salvo |

O frontend é servido estaticamente pelo FastAPI; não há etapa de compilação frontend para o usuário. O servidor iniciado por run.py aceita conexões no endereço 127.0.0.1.

## 2. Dados e protocolo executado

A ICBHI completa foi obtida e auditada; fonte e termos constam em [DADOS.md](DADOS.md). O manifesto bruto tem 6.898 ciclos. A lista consultada apresenta os pacientes 156/218 em treino e teste. O treinamento chama build_manifest com official_overlap_policy='exclude_train', excluindo 13 gravações de treino e seus 117 ciclos; a opção padrão da função é 'error'.

As 381 gravações oficiais de teste foram preservadas. Os dados elegíveis somam 6.781 ciclos: treino 3.594/62 pacientes; validação 431/15; teste 2.756/49. Não foram encontrados outros ciclos inválidos ou PCM duplicado nos registros elegíveis. O protocolo corrigido deve acompanhar qualquer comparação com outros resultados.

O CSV data/manifest.csv contém dataset_id, audio_path, annotation_path, patient_id, recording_id, start_s, end_s, label, cycle_id, split, audio_sha256, sample_rate_hz, channels e duration_s. O JSON data/manifest.audit.json registra raw_cycles, valid_cycles, exclusões por gravação, impacto em ciclos e contagens por partição. Os caminhos locais do manifesto podem exigir reconstrução em outro computador; a matriz de previsões exportada evita caminhos absolutos.

## 3. Contrato do processamento

O backend aceita WAV verdadeiro de até 20 × 1024² bytes, duração de até 120 s, um ou dois canais e taxa entre 2.000 e 192.000 Hz. O trecho deve estar dentro da gravação, durar 0,1–20 s e apresentar RMS mono de pelo menos 1e-7.

O áudio é convertido para mono e reamostrado para 4.000 Hz por reamostragem polifásica. O trecho é normalizado pela amplitude máxima. A STFT usa Hann, n_fft=256 e hop_length=64. Um banco de 64 filtros Mel produz energia logarítmica relativa ao máximo, limitada a [-80,0] dB. Interpolação linear redimensiona o eixo temporal para 256 posições; a matriz final tem 64 × 256 elementos float32 em [0,1].

A política preserva o ciclo inteiro e não recorta automaticamente seu final, mas modifica a escala temporal representada. Essa limitação aparece na interface. O cache mantém float32, sem quantização, e treinamento e inferência chamam a mesma função.

## 4. Modelos e treinamento

A CNN possui blocos convolucionais 16/32/64, kernels 3 × 3 com padding, BatchNorm, ReLU e MaxPool2d(2). A classificação usa AdaptiveAvgPool2d(4,8), camada densa de 64 unidades, dropout 0,3/0,2 e quatro saídas.

O treinamento padrão usa semente 42, lotes de 32, AdamW com taxa 0,0007 e weight_decay 0,0001, perda CrossEntropyLoss ponderada pelas frequências de treino e corte de norma do gradiente em 5. A seleção utiliza macro-F1 da validação; o limite padrão é 18 épocas e a paciência é seis épocas sem melhora. Na execução entregue foram realizadas 12 épocas, com a época 6 selecionada; a rede tem 154.804 parâmetros. A ficha efetiva consta em models/model.json.

O classificador convencional aplica DCT à representação log-Mel, resume média e desvio de 20 coeficientes (40 características) e usa Random Forest com 250 árvores. Há também a referência de classe mais frequente. O classificador da interface é a CNN.

O carregador confere configuração, ordem das classes e SHA-256 dos pesos. Usa torch.load com weights_only=True, modo de avaliação e inferência sem cálculo de gradientes. A versão executada, parâmetros e configuração estão nos metadados do modelo.

## 5. API local

| Método e rota | Função |
| --- | --- |
| GET /api/status | Estado do modelo, metadados, resultados registrados, limites e descrição dos dados |
| GET /api/examples | Lista dos exemplos distribuídos sem expor caminhos arbitrários |
| GET /api/examples/{id}/audio | WAV do exemplo registrado |
| POST /api/analyze | Envio multipart de file, start e end; retorna previsão, segmento, avisos e espectrograma |
| GET /api/docs | Documentação interativa gerada pelo FastAPI |

A análise retorna prediction.label/index/scores, metadados do áudio e do segmento, quality.warnings e spectrogram_png em data URL. Os escores são softmax não calibrados. O endpoint não recebe o rótulo correto do arquivo.

Erros relevantes: 413 para envio acima do limite, 422 para áudio/intervalo inválido, 503 para modelo indisponível ou incompatível e 404 para exemplo inexistente. A aplicação restringe hosts/origens ao uso local. Os uploads não são gravados permanentemente pela aplicação.

## 6. Instalação e utilização

O ambiente de referência utiliza Python 3.13 de 64 bits no Windows. As versões são fixadas em requirements.lock.txt. Execute INSTALAR.bat ou Instalar.ps1; depois INICIAR.bat ou Iniciar.ps1. O instalador cria .venv e instala o pacote local em modo editável.

Comandos manuais após instalar, dentro da pasta do projeto:

    .\.venv\Scripts\python.exe run.py --port 8765
    .\.venv\Scripts\python.exe -m pytest --require-real-artifacts
    .\.venv\Scripts\python.exe scripts/retest.py

A suíte com --require-real-artifacts falha se faltarem os artefatos reais requeridos, em vez de silenciosamente ignorar esses testes. O reteste padrão utiliza os pesos e os exemplos distribuídos e pode ser executado sem baixar novamente toda a base.

## 7. Refazer a pesquisa

Para utilizar a aplicação, não é necessário retreinar. Para produzir um novo experimento, obter a base e executar o treinamento:

    .\.venv\Scripts\python.exe scripts/download_data.py --help
    .\.venv\Scripts\python.exe scripts/download_data.py --output data/raw/icbhi --workers 6 --chunk-mb 8
    .\.venv\Scripts\python.exe scripts/train.py --epochs 18 --batch-size 32 --seed 42
    .\.venv\Scripts\python.exe scripts/retest.py --full

O downloader também aceita --skip-extract. O treinamento permite --data-root, --manifest e --output. Reexecutar o treino pode substituir os pesos/metadados no diretório de saída; preservar uma cópia se for necessário manter o experimento anterior.

O reteste --full exige os WAVs locais, o manifesto exato, o cache float32 correspondente e as previsões salvas daquele experimento. Confere o hash do manifesto, reabre as 381 gravações originais de teste e reextrai os 2.756 ciclos. Compara as representações finitas recalculadas ao cache com rtol=0 e atol=4*eps(float32), aproximadamente 4,7684e-7, para arredondamento numérico; aplica o modelo e exige previsões idênticas às salvas. As métricas recalculadas são comparadas com tolerância de 1e-12. Não é afirmada identidade bit a bit das representações. A evidência completa fica em reports/qa/reteste_completo.json e a leve em reports/qa/reteste_integracao.json. Para um novo treinamento, os artefatos usados pelo reteste deverão corresponder ao novo experimento.

A base completa, o cache e .venv não integram o pacote de entrega. O downloader pode precisar de aproximadamente 6 GB durante aquisição/extração e das ferramentas de download descritas em DADOS.md. Resultados bit a bit idênticos não são garantidos entre diferentes versões/equipamentos; registrar essas diferenças.

## 8. Métricas e artefatos

A unidade de avaliação é o ciclo respiratório. São calculados macro-F1, acurácia, acurácia balanceada, matriz de confusão, sensibilidade/especificidade/precisão/F1 por classe e escore ICBHI. A especificidade por classe é um-contra-os-demais. O escore ICBHI combina o acerto da classe sem eventos com o acerto da categoria exata nos ciclos com eventos. O intervalo de confiança do macro-F1 usa bootstrap por paciente.

| Artefato | Conteúdo |
| --- | --- |
| models/cnn.pt | Pesos da CNN |
| models/model.json | Arquitetura, configuração, versões, época e hash dos pesos |
| models/metrics.json | Resultados CNN, Random Forest e referência, separados por conjunto |
| models/dataset.json | Contagens, protocolo e hash do manifesto |
| models/baseline.joblib | Classificador convencional para reprodução da pesquisa |
| examples/index.json e examples/ORIGEM.md | Proveniência dos quatro ciclos reais distribuídos |
| reports/test_predictions.csv | Rótulos e previsões de todo o teste |
| reports/training_history.json | Perda e macro-F1 por época |
| reports/RESULTADOS.md | Interpretação dos resultados efetivamente obtidos |
| reports/qa/ | Evidências das rodadas de testes e retestes |
| docs/QA_INDEPENDENTE.md | Revisão independente dos testes |

Os valores numéricos finais devem ser lidos nesses artefatos; esta especificação não substitui os resultados ou os registros de execução.

Resumo da execução entregue, arredondado a quatro casas: macro-F1 de teste CNN 0,3216, Random Forest 0,3098 e referência majoritária 0,1821. A acurácia de teste da CNN é 0,4358, inferior à da Random Forest, 0,5145. A avaliação por classe e o intervalo de confiança devem acompanhar a interpretação; a diferença em macro-F1 não estabelece superioridade estatística ou utilidade clínica.

## 9. Limites

O protótipo não tem validação para uso assistencial. O ciclo é delimitado manualmente, a origem pulmonar de um WAV não é verificada automaticamente, e o redimensionamento temporal pode afetar eventos curtos. Ausência de sibilos/estertores não equivale a ausência de doença. O módulo de arritmia não está implementado por falta de dados adequados.
