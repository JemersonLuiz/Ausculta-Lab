# Plano de implementação e estado da entrega

O cronograma de quatro semanas abaixo é o planejamento inicial, preservado como referência acadêmica. A solicitação posterior de implementação resultou em uma aplicação local FastAPI com interface web, substituindo a demonstração exclusivamente em notebook. Os nomes de módulos desta versão correspondem ao código existente.

## 1. Planejamento inicial de um mês

| Semana inicialmente prevista | Trabalho |
| --- | --- |
| 1 | Obter dados, verificar rótulos, separar pacientes e gerar representações |
| 2 | Treinar modelo convencional e CNN e registrar validação |
| 3 | Fixar configuração, avaliar teste reservado e analisar erros |
| 4 | Consolidar demonstração, relatório, documentação e apresentação |

A escrita acompanha os experimentos. Dados, metodologia, resultados, execução e limitações têm documentos próprios no [índice](README.md).

## 2. Recursos implementados

| Etapa | Implementação e evidência |
| --- | --- |
| Aquisição | scripts/download_data.py, origem Dataverse, hashes e DADOS.md |
| Ingestão e auditoria | src/cardio_audio/data.py e data/manifest.audit.json |
| Separação | Teste oficial preservado, 13 gravações de treino sobrepostas excluídas e validação interna por paciente |
| Representação | src/cardio_audio/audio.py, log-Mel 64 × 256 float32 com ciclo inteiro interpolado |
| Referências | Resumo cepstral + Random Forest e classe majoritária em scripts/train.py |
| CNN | src/cardio_audio/model.py, filtros 16/32/64; treino e seleção por validação |
| Avaliação | src/cardio_audio/metrics.py e artefatos de resultados em models/ e reports/ |
| Aplicação | src/cardio_audio/api.py e interface em web/ |
| Inicialização | INSTALAR.bat/Instalar.ps1 e INICIAR.bat/Iniciar.ps1 |
| Testes e reteste | tests/, scripts/retest.py e evidências em reports/qa |

A existência do código não deve ser confundida com aprovação de uma execução. O resultado de cada rodada e o estado final constam no relatório de QA e nos artefatos correspondentes.

## 3. Decisões efetivamente adotadas

- Tarefa principal: quatro categorias respiratórias por ciclo.
- Fonte: dados reais da ICBHI obtidos por espelho rastreável com checksum.
- Correção prévia ao treino: pacientes 156 e 218 tinham gravações em ambos os lados da lista oficial; excluir somente suas 13 gravações de treino e manter as 381 de teste.
- Conjuntos: treino 3.594 ciclos/62 pacientes; validação 431/15; teste 2.756/49.
- Seleção da CNN: macro-F1 da validação; teste não usado para escolher a época.
- Demonstração: aplicação web local com ciclo delimitado manualmente.
- Exemplos: quatro ciclos reais de teste, um por classe, selecionados sem favorecer acertos.
- Componente cardíaco: indisponível por falta de dados apropriados; não simular resultado.
- Distribuição: código, modelo, quatro exemplos, testes e documentação; sem base completa, cache ou .venv.

## 4. Fluxo de reprodução da entrega

Após extrair o pacote, instalar e abrir:

    .\INSTALAR.bat
    .\INICIAR.bat

Com o ambiente instalado, executar a suíte e o reteste:

    .\.venv\Scripts\python.exe -m pytest --require-real-artifacts
    .\.venv\Scripts\python.exe scripts/retest.py

Para refazer a pesquisa completa, obter os dados, treinar e então executar o reteste com cache:

    .\.venv\Scripts\python.exe scripts/download_data.py --help
    .\.venv\Scripts\python.exe scripts/download_data.py --output data/raw/icbhi
    .\.venv\Scripts\python.exe scripts/train.py --epochs 18
    .\.venv\Scripts\python.exe scripts/retest.py --full

A modalidade --full reabre os 381 WAVs originais de teste e reextrai seus 2.756 ciclos, comparando features finitas float32 com o cache real (rtol=0, atol=4*eps(float32), aproximadamente 4,7684e-7) e exigindo previsões idênticas e métricas dentro de 1e-12. O pacote leve não contém a base completa ou o cache; essa modalidade exige os artefatos locais do mesmo experimento.

## 5. Critérios de conclusão e evidências

1. Código de processamento e previsão produz saídas coerentes e rejeita entradas inválidas.
2. Metadados e checksum correspondem ao modelo entregue.
3. Pacientes e conteúdo não atravessam as partições corrigidas.
4. Métricas podem ser recalculadas a partir das previsões de teste.
5. API e inferência direta concordam nos exemplos reais e na repetição.
6. Instalação, inicialização e interface são verificadas na versão final.
7. Resultados, limites e falhas encontradas são documentados.
8. O pacote extraído é conferido antes da entrega.

Os registros de execução estão em reports/qa; a revisão independente está em docs/QA_INDEPENDENTE.md. A interpretação dos resultados está em reports/RESULTADOS.md. O número de testes aprovados e eventuais pendências devem ser lidos desses registros finais.

## 6. Dicionário do manifesto implementado

Cada linha é um ciclo respiratório:

| Campo | Definição |
| --- | --- |
| dataset_id | Base de origem |
| patient_id | Identificador técnico do paciente |
| recording_id | Gravação de origem |
| audio_path / annotation_path | Localização dos arquivos obtidos |
| start_s / end_s | Limites do ciclo na gravação |
| label | 0 sem eventos, 1 estertores, 2 sibilos, 3 ambos |
| cycle_id | Identificador da gravação e linha da anotação |
| split | train, validation ou test |
| audio_sha256 | Hash do PCM decodificado, incluindo taxa/canais |
| sample_rate_hz / channels | Parâmetros da gravação original |
| duration_s | Duração da gravação original |

O modelo de CSV criado durante o planejamento é um esquema ilustrativo anterior. O manifesto efetivamente utilizado é gerado pelo importador com os campos acima; a auditoria associada documenta exclusões e sobreposição.

## 7. Continuação futura

Coleta própria, aplicativo móvel, uso assistencial, detecção automática de respirações contínuas e módulo de arritmia permanecem como trabalhos futuros. A inclusão de FA exigirá áudio cardíaco com confirmação de ritmo e protocolo próprio; a base pulmonar não fornece essa supervisão.
