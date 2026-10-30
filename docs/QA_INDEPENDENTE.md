# Revisão e testes independentes

Revisão realizada em 26/10/2026 por agente dedicado, separado da implementação do processamento, da interface e da ingestão. Este documento registra verificações executadas e seus limites; aprovação de testes não constitui garantia de ausência de falhas nem validação clínica.

## Escopo dos testes

- Decodificação de WAV real mono/estéreo, duração, taxa, canais, conteúdo finito e limites de tamanho.
- Rejeição de arquivos vazios, corrompidos, falsamente identificados como WAV e trechos inválidos, silenciosos ou com valores NaN/inf.
- Representação determinística, paridade entre preparação e inferência, e preservação de evidência no fim do ciclo.
- Mapeamento de anotações e separação por paciente, inclusive conteúdo PCM duplicado associado a identificadores diferentes.
- Política de sobreposição do split oficial: erro por padrão e exclusão explícita apenas do treino quando solicitada.
- API sem modelo, sucesso de contrato com substituto de teste, controle de origem, cabeçalhos, travessia de caminhos e uploads transitórios.
- Integridade de pesos, compatibilidade de categorias e processamento, metadados inválidos e carregamento seguro.
- Métricas comparadas com exemplos calculados manualmente e repetibilidade de bootstrap por paciente.
- Segurança de extração ZIP, tamanho/checksum e reextração de arquivo com CRC incorreto.

As ondas artificiais e os pesos aleatórios existentes nos testes são exclusivamente fixtures de software. Não entram no treinamento científico, não fundamentam métricas do modelo e não são apresentados como dados de pacientes. O substituto de preditor é usado apenas para testar o contrato HTTP sem depender de um treinamento.

## Resultados já executados

| Rodada | Resultado observado | Evidência |
| --- | --- | --- |
| Primeira suíte | 82 testes aprovados em 17,49 s | `reports/qa/primeira_rodada.xml` |
| Revisão unitária após ampliar métricas e política de split | 93 testes aprovados em 8,44 s | `reports/qa/revisao_unitaria.xml` |
| Testes adicionais do downloader, sem rede | 6 testes aprovados em 0,16 s | `reports/qa/download_seguranca.xml` |
| Suíte completa com artefatos reais — rodada 1 | 105 testes aprovados, zero falhas e zero casos pulados, em 7,72 s | `reports/qa/integracao_real_rodada1.xml` |
| Reteste completo, novo processo — rodada 2 | 105 testes aprovados, zero falhas e zero casos pulados, em 7,64 s | `reports/qa/integracao_real_rodada2.xml` |
| ZIP extraído em pasta nova e Python novo | Instalação do lock, importação da cópia, pip check, 105 testes sem falhas/skips, reteste leve e HTTP real aprovados | [pacote_extraido.json](../reports/qa/pacote_extraido.json) e `reports/qa/pacote_logs/` |

Os tempos são os informados pelo pytest. A revisão unitária e os seis casos do downloader foram execuções distintas, não uma execução única de 99 casos.

As duas rodadas completas usaram o comando obrigatório com artefatos reais, em processos separados e sem alteração de código ou pesos entre elas. Elas incluem todos os casos anteriores e a integração científica e HTTP descrita abaixo. Os XMLs registram a duração técnica total de cada suíte, que pode diferir ligeiramente do resumo exibido pelo pytest.

## Achados tratados durante a revisão

1. Metadados JSON de tipo diferente de objeto poderiam causar exceção não convertida em indisponibilidade do modelo. A implementação foi ajustada e os testes incluem lista, texto, número e nulo.
2. Pesos ilegíveis com checksum correspondente poderiam gerar erro de desserialização fora do tratamento esperado. O carregamento agora trata essa condição; `weights_only=True` permanece obrigatório.
3. Categorias numéricas fora de 0–3 ou não inteiras poderiam ser convertidas/ignoradas indevidamente nas métricas. A implementação passou a rejeitar explicitamente esses valores.
4. A aplicação local recebeu restrição de Host além da verificação de Origin. Isso reduz a superfície de acesso indevido ao servidor local.
5. A primeira instalação em uma extração muito profunda falhou ao instalar cabeçalhos do PyTorch por limite de caminho do Windows. A tentativa foi preservada. O instalador final rejeita antecipadamente pastas com mais de 90 caracteres e recomenda uma pasta curta, sem alterar configurações do Windows. Esse bloqueio foi executado em uma pasta de 118 caracteres: retornou erro informativo antes de iniciar pip. O pacote foi então verificado em extração curta direta e ambiente inteiramente novo.

Os casos correspondentes estavam aprovados nas rodadas acima. Nenhum desses ajustes implica que outros problemas sejam impossíveis.

## Conferência independente da base real

A lista por gravação destinada ao teste foi comparada diretamente ao manifesto produzido, normalizando os quatro campos anteriores ao equipamento conforme a correção documentada. O teste preserva exatamente as **381 gravações** da lista oficial.

Também foram recalculadas as interseções por identificador de paciente e por SHA-256 do PCM: nenhuma atravessa treino, validação e teste. O manifesto tem **6.781 ciclos**, correspondentes aos 6.898 originais menos as 117 exclusões preventivas de treino dos pacientes sobrepostos 156 e 218. Os detalhes de origem e da política estão em [DADOS.md](DADOS.md) e `data/manifest.audit.json`.

## Integração e reteste executados

Os testes de integração em `tests/test_real_artifacts.py` verificam hash e metadados do modelo, recalculam macro-F1 e matriz de confusão a partir das predições reservadas, conferem as partições e a origem dos exemplos e comparam a saída da API à inferência dos pesos salvos. A repetição da mesma entrada deve conservar a previsão.

Para impedir que a ausência dos artefatos seja interpretada como teste aprovado, executar:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --require-real-artifacts
```

A opção falha se os artefatos reais necessários não estiverem presentes. Na invocação padrão, esses casos podem ser pulados enquanto o treinamento não terminou; um resultado com casos pulados não será registrado aqui como integração concluída.

Na execução efetiva, as métricas foram recalculadas a partir de `reports/test_predictions.csv`, sem depender da função de métricas do projeto: macro-F1 e matriz de confusão foram comparadas com as funções do scikit-learn. O teste também confirmou a igualdade entre as predições da API e do modelo carregado diretamente em todos os exemplos distribuídos, inclusive em duas chamadas repetidas de cada exemplo.

O script complementar `scripts/retest.py --full`, revisado separadamente, reconstrói as representações lendo os WAVs originais de teste e compara features, predições e métricas. Sua evidência é registrada pelo processo de integração da entrega; os 105 testes deste relatório não devem ser confundidos com o reprocessamento completo desse script.

**Resultado complementar conferido:** `reports/qa/reteste_integracao.json` registra aprovação do reprocessamento dos 2.756 ciclos provenientes das 381 gravações reservadas. Todas as predições coincidiram exatamente com as salvas; macro-F1, acurácia, acurácia balanceada e escore ICBHI foram comparados com tolerância absoluta de 1e-12.

As features não foram idênticas bit a bit: a diferença absoluta máxima observada foi **1,9371509552e-7**. A comparação utiliza `rtol=0` e tolerância absoluta **4,7683715820e-7**, correspondente a quatro vezes o epsilon de float32 para a escala normalizada [0,1]. Essa margem pequena é compatível com arredondamentos de ponto flutuante e ordem de operações; não é usada para relaxar a comparação de categorias previstas. A identidade exata das predições foi mantida como requisito. Não houve alteração de pesos ou seleção de hiperparâmetros após conhecer o teste reservado.

## Validação independente do pacote distribuível

O verificador [scripts/verify_package.py](../scripts/verify_package.py) foi incluído no pacote e executado depois de congelar código, interface, modelos, requisitos e instaladores. Conferiu CRC, caminhos seguros do ZIP e os 89 arquivos declarados em `CHECKSUMS.sha256` do candidato avaliado.

A extração final foi feita diretamente em uma pasta curta e nova no TEMP, sem junction ou alteração de registro do Windows. Dentro dela foi criado outro ambiente `.venv`, usando Python **3.13.2** de 64 bits. A instalação de `requirements.lock.txt` foi efetivamente executada (69,44 s nesta máquina), seguida de `pip install --no-deps -e .` e `pip check`.

A prova de importação registra `cardio_audio.__file__` dentro da pasta `src` da cópia e `sys.prefix` no ambiente novo, com o ambiente original e `PYTHONPATH` excluídos dessa execução. Nesse ambiente foram executados **105 testes**, sem falhas, erros ou casos pulados, e o reteste leve de inferência real.

Depois foi iniciado o servidor real por `run.py` na porta local 8771. Página, status e arquivos estáticos responderam HTTP 200; o modelo estava disponível; duas análises do mesmo exemplo deram resultados idênticos; um arquivo inválido recebeu HTTP 422. O verificador encerrou somente seu próprio servidor e confirmou esse encerramento.

O JSON guarda o SHA-256 do ZIP candidato e o fingerprint de cada arquivo relevante de código, UI, pesos, requisitos, instaladores, exemplos e artefatos de avaliação. Fonte e cópia coincidiram no início e no fim da verificação. Documentos e relatórios posteriores podem ser adicionados ao ZIP final sem mudança funcional, desde que o fingerprint desses arquivos relevantes continue igual e os checksums/CRC finais sejam conferidos.

Os logs, a prova de importação, o JUnit e o resultado do reteste da cópia foram preservados no caminho portátil `reports/qa/pacote_logs/`; o resultado não depende de o destinatário ter acesso à pasta TEMP desta máquina. A falha inicial de caminho longo está preservada em `pacote_extraido_tentativa1.json` e no log correspondente. Uma tentativa intermediária foi cancelada para substituir a estratégia de junction por extração curta direta, não sendo contada como teste aprovado.

O conjunto completo de áudios e o cache de treinamento não são distribuídos no ZIP. O reteste leve e os testes com artefatos inclusos foram executados nessa condição. Já `scripts/retest.py --full` exige o experimento local coerente com WAVs, manifesto e cache, ou a reconstrução/treinamento documentados em outro computador.

## Qualidade observada do modelo

A aprovação do software não significa que suas classificações sejam todas corretas. A avaliação reservada tem 2.756 ciclos. As métricas salvas, conferidas independentemente, são:

| Modelo | Macro-F1 | Acurácia | Acurácia balanceada | Escore ICBHI |
| --- | ---: | ---: | ---: | ---: |
| CNN | 0,3216 | 0,4358 | 0,3299 | 0,4210 |
| Random Forest | 0,3098 | 0,5145 | 0,3118 | 0,4793 |
| Classe majoritária | 0,1821 | 0,5729 | 0,2500 | 0,5000 |

A CNN tem macro-F1 pontualmente maior, mas acurácia e escore ICBHI menores que os modelos comparados. A diferença pontual de macro-F1 não demonstra, por si só, superioridade estatística. O desempenho é modesto, com erros relevantes de classificação; não fundamenta uso clínico ou promessa de triagem confiável. Não foram ajustados pesos ou hiperparâmetros em resposta ao teste reservado.

## Avisos e limites

Foi observado aviso de depreciação de Starlette/TestClient com a dependência httpx. O cenário de métrica com apenas uma classe verdadeira também produz um aviso do scikit-learn; o teste valida a saída esperada para esse cenário deliberadamente incompleto. Os avisos não representam falhas funcionais observadas, mas devem permanecer identificados.

A revisão de software não demonstra sensibilidade clínica, utilidade de triagem, reconhecimento de arritmias ou funcionamento em qualquer equipamento. A qualidade estatística será relatada a partir do teste real reservado. O componente de arritmia permanece indisponível, e a seleção de um ciclo respiratório continua manual.
