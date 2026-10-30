# Dados reais, origem e protocolo

## Fonte e condições de uso

O experimento utiliza a ICBHI 2017 Respiratory Sound Database. Os áudios e rótulos são dados reais de pesquisa, não gerados pelo projeto. A referência original é Rocha et al., *Physiological Measurement* (2019), DOI [10.1088/1361-6579/ab03ea](https://doi.org/10.1088/1361-6579/ab03ea).

A [página dos organizadores](https://bhichallenge.med.auth.gr/) informa disponibilidade gratuita para pesquisa e solicita a citação da publicação da base. Essa declaração não deve ser interpretada como licença irrestrita para qualquer finalidade comercial ou de redistribuição.

Na aquisição, o certificado TLS do servidor original estava expirado. O download do ZIP usa HTTPS verificado no [espelho Harvard Dataverse, DOI 10.7910/DVN/HT6PKI](https://doi.org/10.7910/DVN/HT6PKI), versão 1.0, publicada em 30/05/2023. O espelho declara CC0 em seus metadados, mas essa informação não é tratada como substituição das condições da fonte original. O arquivo não será integralmente embutido no pacote da aplicação.

## Integridade do arquivo

O downloader confere o tamanho e o MD5 publicado no Dataverse antes da extração e registra também SHA-256. Não desativa a verificação de certificados HTTPS. A extração verifica CRC e rejeita caminhos fora da pasta de destino.

| Campo | Valor publicado no espelho |
| --- | --- |
| Arquivo | ICBHI_final_database.zip |
| Identificador Dataverse | 7127117 |
| Tamanho | 1.978.998.275 bytes |
| MD5 | f6a5d0ea40bbcd25e17754f9d5ddae48 |
| Registro local de aquisição | data/raw/icbhi/provenance.json, gerado após download completo |

O script `scripts/download_data.py` suporta retomada por partes. O progresso é registrado em `data/raw/icbhi/download_progress.json`. O conjunto completo permanece local, fora do pacote de entrega.

## Formato e interpretação

Cada WAV está associado a um TXT com início do ciclo, fim do ciclo, indicador de estertores e indicador de sibilos. A classificação é acústica por ciclo. O mapeamento implementado é `label = estertores + 2 * sibilos`: 0 sem esses eventos, 1 estertores, 2 sibilos, 3 ambos. Não usamos diagnósticos clínicos como substitutos desses rótulos.

O manifesto preserva identificador da gravação, paciente, limites do ciclo e origem do áudio. Conteúdo PCM decodificado recebe hash SHA-256 para identificar duplicatas mesmo quando o cabeçalho do arquivo difere.

## Partições e auditoria

A lista oficial de partições é obtida de um [espelho com commit fixado](https://raw.githubusercontent.com/jaygshah/ICBHI/c88d99c53f1da0094bf9fec65fdd79a1c0f5f7ad/ICBHI_challenge_train_test.txt). A autoria desse repositório não é confundida com a organização da base: ele é utilizado como espelho da lista, com origem e checksum registrados. A lista é conferida contra os identificadores realmente obtidos.

Foi detectada uma inconsistência real na lista: os pacientes **156 e 218** têm gravações em treino e teste. A lista contém 539 gravações destinadas ao treino e 381 ao teste. A lista de treino foi comparada com a referência independente do [repositório de pesquisadores nttcslab/m2d](https://github.com/nttcslab/m2d/blob/master/app/icbhi_sprs/files_icbhi2017.csv); os registros coincidem após normalizar a correção de equipamento do identificador 226_1b1_Pl_sc.

O padrão do importador é rejeitar essa sobreposição. O experimento utiliza explicitamente `official_overlap_policy='exclude_train'`: preserva integralmente as **381 gravações de teste** e exclui somente **13 gravações de treino** dos dois pacientes sobrepostos, correspondentes a **117 ciclos**. Nenhum arquivo de teste é transferido para treino. Essa correção foi definida antes do treinamento, para impedir vazamento, e não em função de métricas.

Vinte por cento dos pacientes restantes destinados ao treino são reservados para validação, com ordenação e semente 42. A correção da sobreposição e a divisão interna são próprias do projeto. O protocolo não deve ser comparado ingenuamente com avaliações que mantêm pacientes sobrepostos.

Além da política explícita acima, o importador rejeita PCM idêntico associado a pacientes diferentes; duplicatas do mesmo paciente são removidas e registradas. Áudios ilegíveis e linhas com valores não finitos, rótulos inválidos, limites invertidos ou fora da duração são excluídos do manifesto com motivos explícitos no arquivo `.audit.json` correspondente. Ciclos fora de 0,1–20 s ou com RMS mono inferior a 1e-7 também são excluídos, alinhando os critérios de treino e inferência. Os originais permanecem preservados.

Diferenças de nomes de equipamento descritas na correção oficial são tratadas comparando os quatro primeiros campos da gravação, sem modificar identificação do paciente, visita, posição e modo.

## Relatórios gerados

`build_manifest` produz o CSV de ciclos válidos e um JSON de auditoria com contagens por partição, distribuição das classes, exclusões, remoção de duplicatas e verificações de sobreposição. Os números finais devem ser lidos desses arquivos após a execução, e não inferidos de valores esperados da publicação.

### Auditoria efetivamente executada em 26/10/2026

O ZIP foi integralmente obtido e conferido: MD5 publicado correspondente e SHA-256 **e34e9913c43ee9da62af4faa7335d8eda7ce9883c315b69b07cfcbde28650e4c**. Foram encontrados 920 pares WAV/TXT e 126 pacientes. Após a exclusão preventiva das 13 gravações de treino sobrepostas, o manifesto contém 6.781 ciclos de 907 gravações.

| Partição | Pacientes | Gravações | Ciclos | Sem eventos | Estertores | Sibilos | Ambos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Treino | 62 | 467 | 3.594 | 1.833 | 1.130 | 388 | 243 |
| Validação | 15 | 59 | 431 | 220 | 56 | 98 | 57 |
| Teste | 49 | 381 | 2.756 | 1.579 | 649 | 385 | 143 |

Não foram encontrados outros ciclos inválidos nem duplicatas de PCM entre os registros elegíveis. Não existe sobreposição de paciente ou conteúdo de áudio entre as partições produzidas. A lista individual das exclusões fica em `data/manifest.audit.json`, com origem, motivo e impacto. O CSV mantém caminhos locais para os arquivos; em outro computador, reconstruir o manifesto após o download.

## Limite do componente cardíaco

Esta aquisição cobre somente os sons respiratórios. Não há rótulos de fibrilação atrial nesses arquivos. O módulo cardíaco não pode ser treinado ou apresentado como validado com esta base.
