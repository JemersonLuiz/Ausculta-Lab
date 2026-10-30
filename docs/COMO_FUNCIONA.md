# Como funciona o Ausculta Lab

Atualizado em 26/10/2026 para a aplicação implementada. O sistema é uma aplicação local com servidor Python/FastAPI e interface HTML/CSS/JavaScript. O componente respiratório utiliza uma CNN sobre espectrogramas. A parte de arritmia permanece indisponível por falta de dados de áudio com confirmação de ritmo.

## 1. Dados efetivamente obtidos

A base real ICBHI 2017 foi obtida pelo espelho Harvard Dataverse, com verificação HTTPS e conferência do checksum publicado. A origem, as condições de pesquisa e os hashes estão em [DADOS.md](DADOS.md). O experimento não utiliza gravações sintéticas para produzir métricas de pesquisa.

A base original contém 920 pares de áudio e anotação. O importador associa cada ciclo ao paciente, à gravação e à categoria anotada. O rótulo é estertores + 2 × sibilos: 0 sem esses eventos, 1 estertores, 2 sibilos e 3 ambos. Diagnósticos e nomes dos arquivos não entram como características da CNN.

## 2. Correção necessária nas partições

A lista de partições consultada tinha os pacientes 156 e 218 em treino e teste. O procedimento estrito rejeita essa sobreposição. No experimento, foi escolhida explicitamente a política de excluir as 13 gravações de treino desses pacientes, correspondentes a 117 ciclos, preservando as 381 gravações de teste.

O manifesto corrigido tem 6.781 ciclos: 3.594 de treino, 431 de validação e 2.756 de teste. Os pacientes ficam separados: 62, 15 e 49, respectivamente. Não há pacientes ou PCM duplicado atravessando as partições produzidas. Esse protocolo corrigido deve ser identificado ao comparar os resultados com outros estudos.

## 3. Transformação do áudio

O processamento compartilhado entre treinamento e inferência faz:

1. Leitura do WAV e conferência de integridade e limites.
2. Seleção de um ciclo entre 0,1 e 20 segundos.
3. Conversão de estéreo para mono, quando necessária.
4. Reamostragem para 4.000 Hz e normalização pela amplitude máxima do trecho.
5. STFT com janela Hann de 256 amostras e passo de 64.
6. Banco de 64 bandas Mel e escala logarítmica, limitada a uma faixa de 80 dB.
7. Interpolação linear do eixo temporal para 256 posições, preservando a extensão completa do ciclo.

O resultado é uma matriz float32 de 64 × 256 em [0,1]. Não há quantização no cache das características. Interpolar o tempo evita cortar o final de ciclos longos, mas pode alterar a representação de eventos curtos; essa limitação é registrada na aplicação.

## 4. Aprendizado da CNN

A rede tem três blocos com 16, 32 e 64 filtros, cada um com convolução 3 × 3, normalização de lote, ReLU e redução espacial. A saída passa por agregação adaptativa 4 × 8, camadas densas e dropout, produzindo quatro pontuações.

O treinamento compara a previsão com o rótulo anotado, calcula a perda e ajusta os parâmetros por gradientes. O script utiliza AdamW, taxa inicial 0,0007, perda ponderada por classe e semente 42. O comando padrão permite até 18 épocas; a interrupção antecipada ocorre após seis épocas sem melhora. A melhor época é escolhida pelo macro-F1 da validação, antes da avaliação do teste.

O modelo convencional compara um resumo cepstral de 40 números com uma Random Forest de 250 árvores. Um preditor da classe mais frequente fornece outra referência. O treinamento real executou 12 épocas e selecionou a época 6 pela validação.

No teste reservado, o macro-F1 foi 0,3216 para a CNN, 0,3098 para a Random Forest e 0,1821 para a referência majoritária. O desempenho da CNN é limitado: sua acurácia de 0,4358 foi inferior à da Random Forest, 0,5145. A pequena vantagem em macro-F1 não demonstra superioridade geral nem adequação clínica. As métricas completas ficam em models/metrics.json e no relatório reports/RESULTADOS.md.

## 5. Utilização depois do treinamento

Para abrir o sistema, instale as dependências com INSTALAR.bat e execute INICIAR.bat. O navegador acessa http://127.0.0.1:8765.

É possível carregar um dos quatro recortes reais de teste ou selecionar um WAV próprio. O usuário delimita uma respiração, confirma o trecho e solicita a análise. A interface mostra a forma de onda, a categoria prevista, as quatro pontuações, o espectrograma e avisos técnicos. O resultado pode ser salvo em JSON.

A CNN usa os parâmetros salvos; a utilização não altera seu treinamento. A classificação do trecho é automática, mas a localização dos ciclos em gravações contínuas é manual.

## 6. Como verificamos o funcionamento

A suíte de testes cobre processamento, anotações, partições, métricas, carregamento do modelo e API. A verificação com artefatos reais compara a API com a inferência direta e repete as mesmas entradas. Os sinais artificiais de testes unitários são apenas fixtures de software e não entram nas métricas do experimento.

O script scripts/retest.py repete a integração com os exemplos. Com --full, reabre os 381 WAVs originais do teste, reextrai as representações dos 2.756 ciclos e as compara ao cache float32 do treinamento com tolerância absoluta de quatro epsilons float32 (4,7684e-7), sem tolerância relativa e rejeitando valores não finitos. Em seguida, exige previsões idênticas e métricas iguais dentro de 1e-12. Essa modalidade requer a base local e o cache exato do experimento. Os relatórios leve e completo são salvos separadamente em reports/qa/.

Os resultados e evidências devem ser consultados em reports/RESULTADOS.md, reports/qa e docs/QA_INDEPENDENTE.md. Aprovação de testes de software não equivale a validação clínica.

## 7. Limites desta entrega

O sistema classifica padrões acústicos de um ciclo respiratório. “Sem sibilos/estertores” não significa pessoa saudável. As pontuações softmax não são probabilidades clínicas calibradas.

Não existe módulo de arritmia funcionando nesta versão. Implementá-lo exige outro conjunto de áudios e confirmação do ritmo no período correspondente à gravação. O modelo pulmonar não fornece esse resultado.

## 8. Conteúdo do pacote

A entrega inclui código, instaladores, modelo, metadados, quatro exemplos reais, testes e documentação. A base completa, o cache de treinamento e o ambiente .venv não são incluídos. É possível usar o modelo e os exemplos sem baixar a base novamente; para refazer o treinamento, seguir a [documentação técnica](DOCUMENTACAO_TECNICA.md).
