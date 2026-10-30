# Projeto de pesquisa — Triagem cardiopulmonar por áudio e machine learning

Plano inicial revisado para um mês em 23/10/2026; implementação registrada em 26/10/2026.

**Estado da execução:** aplicação local FastAPI com interface HTML/JS implementada, dados ICBHI reais adquiridos e auditados, CNN e modelo convencional treinados. O experimento utilizou 6.781 ciclos após excluir 117 ciclos de treino para impedir sobreposição de pacientes. A CNN obteve macro-F1 de 0,3216, com desempenho limitado e sem validação clínica. O módulo cardíaco permanece indisponível. Consulte [resultados executados](reports/RESULTADOS.md), [relatório acadêmico](docs/RELATORIO_ACADEMICO.md) e [QA independente](docs/QA_INDEPENDENTE.md). As seções abaixo preservam a formulação e o planejamento; os documentos citados registram os procedimentos efetivamente realizados.

**Título informado:** Detecção precoce de padrões de arritmia cardíaca e sopros pulmonares via processamento de sinais de áudio e redes neurais convolucionais.

**Título técnico proposto:** Identificação de padrões sugestivos de fibrilação atrial e sons respiratórios adventícios por processamento de áudio e redes neurais convolucionais.

**Equipe:** a preencher. **Instituição e orientador(a):** a preencher. **Prazo disponível:** aproximadamente um mês; planejamento em quatro semanas, com data exata de entrega a confirmar.

**Recorte de execução proposto para o prazo:** concentrar o desenvolvimento e a avaliação em um módulo pulmonar com CNN e modelo convencional de comparação. O tema cardíaco permanece na revisão e no plano de expansão, condicionado à obtenção de dados. A exigência de dois módulos funcionando ainda precisa ser esclarecida com a orientação; este planejamento não pressupõe sua conclusão em um mês.

**Título possível para a entrega delimitada:** Classificação de sons respiratórios adventícios por processamento de áudio e redes neurais convolucionais. Utilizar esse título se o recorte pulmonar for adotado na entrega acadêmica; manter no título os alvos efetivamente avaliados.

## 1. Resumo da proposta

Desenvolver um protótipo de pesquisa que receba gravações de ausculta, prepare os sinais e utilize modelos supervisionados para reconhecer padrões acústicos previamente anotados. A proposta temática prevê dois módulos independentes: um pulmonar, para sons respiratórios, e um cardíaco, inicialmente voltado à fibrilação atrial (FA), mediante acesso a uma base adequada. Para a entrega em um mês, a prioridade proposta é implementar e avaliar o módulo pulmonar, documentando separadamente a viabilidade do cardíaco.

O componente central de machine learning será uma rede neural convolucional (CNN) treinada com espectrogramas. Um modelo convencional servirá de comparação. O projeto entregará código, protocolo de avaliação e uma demonstração com arquivos de teste.

Classificar gravações já anotadas não demonstra, por si só, antecipação do diagnóstico. A expressão “detecção precoce” será tratada como motivação de aplicação; comprovar esse benefício exigiria um estudo clínico específico.

## 2. Delimitação e terminologia

- Para a parte pulmonar, o alvo proposto são **sons respiratórios adventícios**, especificamente sibilos e estertores. Essas categorias são usadas nas pesquisas com a base ICBHI [1, 2].
- Um sopro relacionado à valva pulmonar pertence à ausculta cardíaca. Caso esse seja o sentido pretendido no tema original, será necessário ajustar os alvos pulmonares do projeto.
- Para arritmia, o recorte inicial proposto é **FA presente versus FA ausente**, com rótulos estabelecidos por ECG. O modelo não será apresentado como reconhecedor de todas as arritmias [3].
- Os sinais de entrada do classificador serão áudios reais de ausculta. Um ECG convertido em arquivo de áudio não equivale a um fonocardiograma.
- As bases cardíaca e pulmonar serão avaliadas separadamente. Não se presumirá que contêm os mesmos pacientes ou gravações simultâneas.

## 3. Problema de pesquisa

**Pergunta principal:** uma CNN treinada com representações de áudio consegue reconhecer os padrões selecionados em pacientes não utilizados no treinamento e oferecer desempenho superior a um classificador convencional?

Pergunta complementar prioritária no prazo: quais classes apresentam mais erros? Estudos controlados de ruído e generalização entre equipamentos ficam como extensões, caso as entregas principais estejam concluídas.

## 4. Justificativa

O projeto integra sinais biológicos, processamento digital e inteligência artificial para estudar uma possibilidade de apoio à triagem. Pesquisas anteriores já investigam classificação de sons respiratórios por CNN e identificação de FA por ausculta [1, 3].

A contribuição pretendida é construir e avaliar um procedimento reproduzível, com comparação entre modelos e análise de erros. Não se pressupõe que aplicar uma CNN ao tema seja uma inovação inédita.

## 5. Hipótese

Uma CNN que utiliza espectrogramas pode superar um modelo baseado em características acústicas convencionais na métrica macro-F1, sob o mesmo protocolo de separação por paciente.

A hipótese poderá ser confirmada ou rejeitada. Um resultado desfavorável à CNN também será um resultado válido do projeto.

## 6. Objetivos

### Objetivo geral

Desenvolver e avaliar um protótipo de classificação de sons de ausculta com machine learning, visando investigar seu potencial de apoio à triagem cardiopulmonar.

### Objetivos específicos

1. Revisar trabalhos sobre processamento de áudio, sons cardiopulmonares e CNN.
2. Verificar a origem, os rótulos, o acesso e as condições de uso dos dados.
3. Organizar um catálogo rastreável de gravações, pacientes e anotações.
4. Desenvolver o processamento de áudio e a extração de características.
5. Treinar um modelo convencional e uma CNN com os mesmos dados elegíveis.
6. Avaliar os modelos em pacientes separados do treinamento.
7. Investigar erros e desequilíbrio entre classes; avaliar ruído como extensão, se houver tempo.
8. Demonstrar a análise de um novo arquivo do conjunto reservado para teste.

## 7. Dados, módulos e viabilidade

| Módulo | Alvo e unidade de análise | Dados propostos | Situação inicial |
| --- | --- | --- | --- |
| Pulmonar | Ciclo respiratório: sem sibilos/estertores, estertores, sibilos ou ambos | ICBHI 2017; 920 gravações de 126 participantes auditadas [1, 2] | Implementado com dados reais; treino 62 pacientes, validação 15 e teste 49; origem e exclusões em DADOS.md |
| Cardíaco — FA | Janela de ausculta com rótulo de FA presente/ausente estabelecido por ECG no período correspondente | Base clínica de áudio com identificação de paciente e confirmação de ritmo | Acesso a uma base apropriada ainda não confirmado; o estudo de Guo et al. disponibiliza dados mediante solicitação aos autores [3] |
| Cardíaco — extensão opcional | Detecção de sopro cardíaco | CirCor / PhysioNet Challenge 2022 [4] | Base identificada para esse alvo; não substitui dados de arritmia |

**Ponto de decisão — primeiros três dias:** verificar acesso, rótulos e exigências acadêmicas. Na ausência de dados adequados de FA, seu treinamento não será uma entrega prometida para o mês. O módulo pulmonar será o foco proposto para implementação. Se a orientação exigir os dois alvos funcionando, essa restrição deve ser resolvida já nessa etapa com um recorte compatível com os dados; trocar FA por sopros cardíacos exigiria reformular explicitamente o objetivo.

**Cuidados com os rótulos:** ausência de sibilos e estertores não equivale a ausência de doença pulmonar. A base CirCor traz anotações de sopros e desfechos cardíacos; esses rótulos não podem ser convertidos em “com arritmia” ou “sem arritmia” [4].

## 8. Metodologia

### 8.1 Tipo de estudo

Pesquisa aplicada e experimental com dados secundários. A primeira versão será executada com bases existentes e dados desidentificados. Uma eventual coleta própria será uma etapa separada, com protocolo e aprovações institucionais aplicáveis.

### 8.2 Inventário e auditoria

Registrar identificador da base, paciente desidentificado, gravação, caminho do arquivo, canal, posição de ausculta, frequência de amostragem, duração, anotação e origem do rótulo. Identificar duplicatas, arquivos corrompidos e classes com poucos pacientes.

Preservar arquivos originais. Documentar exclusões e contabilizar pacientes, gravações e segmentos separadamente.

### 8.3 Separação dos dados

- Separar por paciente **antes** de gerar segmentos e aumentar os dados.
- Preservar o teste oficial da base quando disponível e verificar ausência de sobreposição de pacientes.
- Realizar a escolha de configurações na parcela de treinamento, usando validação agrupada por paciente.
- Bases sem divisão oficial receberão uma partição documentada por paciente, compatível com o tamanho e a distribuição das classes.
- Se houver múltiplas visitas ou identificadores da mesma pessoa, agrupá-los na mesma partição.
- Manter o teste reservado até a definição do modelo. Ajustes aprendidos de normalização serão estimados no treinamento; balanceamento e aumento de dados serão aplicados somente ali.

### 8.4 Preparação do áudio

Ler arquivos WAV e suas anotações; tratar canais de forma documentada; padronizar a taxa de amostragem conforme o módulo; verificar silêncio e saturação; segmentar os sinais; gerar espectrogramas log-Mel.

No módulo pulmonar, usar inicialmente os limites dos ciclos fornecidos pelos anotadores. No módulo de FA, avaliar janelas que preservem vários batimentos. Filtros, duração e taxa serão definidos com base nas características de cada sinal e na validação, sem copiar automaticamente o mesmo processamento entre coração e pulmão.

### 8.5 Modelo de referência

Extrair MFCC e estatísticas acústicas e treinar um **Random Forest**. Comparar também com um preditor simples baseado nas frequências das classes.

Essa comparação permitirá verificar se a complexidade da CNN traz benefício sobre uma abordagem mais simples.

### 8.6 Modelo principal — CNN

**Modalidade:** aprendizado supervisionado; deep learning é parte de machine learning.

**Entrada:** espectrograma log-Mel de um segmento de áudio.

**Arquitetura inicialmente candidata:** três blocos convolucionais com 32, 64 e 128 filtros. **Arquitetura executada:** versão compacta com 16, 32 e 64 canais, ReLU, max-pooling, agregação adaptativa e camadas densas; 154.804 parâmetros. O detalhamento reproduzível está nos metadados e no relatório experimental.

**Saídas propostas:** quatro categorias no módulo pulmonar e duas no módulo de FA, se houver dados. Cada módulo terá seu próprio modelo e mapeamento de classes. Se a auditoria da primeira semana indicar inviabilidade da tarefa pulmonar com quatro classes no prazo, registrar antes do treinamento um recorte binário: presença versus ausência de sibilos e/ou estertores. Esse recorte não equivale a classificar pessoas como saudáveis ou doentes.

Usar ponderação das classes, interrupção antecipada do treinamento e registro de sementes e configurações. Para quatro semanas, começar com uma CNN pequena e limitar a poucas configurações escolhidas na validação. Arquiteturas híbridas, grande busca de hiperparâmetros e treinamento de vários modelos profundos ficam como trabalhos futuros.

### 8.7 Avaliação

**Métrica principal:** macro-F1, calculada de forma independente em cada tarefa.

**Métricas complementares:** sensibilidade/recall por classe, especificidade com definição explícita, matriz de confusão, acurácia balanceada e, no caso binário, curvas ROC e precisão-revocação. No módulo pulmonar, calcular também o escore ICBHI conforme a definição do benchmark, sem confundi-lo com macro-F1.

Definir a agregação de janelas antes de avaliar o teste. A métrica de ciclos pulmonares não será apresentada como diagnóstico de paciente. Se houver resultado por gravação ou por paciente, apresentar essas análises separadamente.

Comparar os modelos nas mesmas partições. Quando viável, estimar intervalos de confiança com reamostragem por paciente. Análises por equipamento e qualidade ficam como extensões após concluir o experimento principal. Não juntar bases de populações diferentes para gerar uma única “acurácia cardiopulmonar”.

### 8.8 Demonstração do protótipo

A demonstração inicial poderá ser feita em notebook, selecionando o módulo e um arquivo de teste. Exibir onda do áudio, espectrograma, categoria prevista, pontuações do modelo e eventuais problemas técnicos do arquivo.

No módulo pulmonar inicial, classificar um ciclo com limites conhecidos pela anotação da base ou selecionados pelo usuário para outra gravação. A detecção automática de ciclos em gravação contínua fica como extensão; as anotações de classe serão usadas apenas na comparação do resultado, sem entrar na inferência.

Pontuações não calibradas serão identificadas como pontuações, sem equivaler a probabilidade clínica. Arquivo ilegível ou insuficiente para o processamento deverá produzir uma mensagem de impossibilidade de análise. Um classificador treinado de qualidade só será incluído se houver rótulos adequados para esse objetivo.

## 9. Fluxo técnico

```text
Base de áudio + anotações + identificador de paciente
                        |
                 Auditoria dos dados
                        |
             Separação por paciente
                        |
         Preparação específica de cada módulo
                        |
          +-------------+-------------+
          |                           |
     MFCC e estatísticas       Espectrograma log-Mel
          |                           |
      Random Forest                  CNN
          +-------------+-------------+
                        |
          Comparação no mesmo teste reservado
                        |
          Relatório e demonstração de inferência
```

## 10. Tecnologias e recursos propostos

Python para a implementação; NumPy/SciPy e uma biblioteca de áudio para sinais; scikit-learn para o modelo de referência e métricas; PyTorch para a CNN; Jupyter e Matplotlib para exploração e demonstração.

As versões serão escolhidas e registradas no início da implementação. Um computador com CPU permite iniciar a organização e o modelo convencional. A necessidade de GPU e armazenamento será estimada após um primeiro experimento pequeno; não há necessidade de comprar equipamento nesta fase de planejamento.

## 11. Cronograma revisado — quatro semanas

Planejamento relativo ao início efetivo do trabalho, com aproximadamente 28 dias de execução e os dias restantes do mês como margem. Não foi informada uma data exata nem a disponibilidade diária da equipe.

| Período | Atividade | Entrega verificável |
| --- | --- | --- |
| Semana 1 — dias 1 a 7 | Resolver fontes e recorte nos primeiros três dias; obter e auditar a base; separar pacientes; preparar ciclos e espectrogramas | Base organizada, partições conferidas e metodologia inicial escrita |
| Semana 2 — dias 8 a 14 | Treinar MFCC + Random Forest e primeira CNN pequena; comparar na validação | Modelos candidatos e primeiro quadro comparativo de validação |
| Semana 3 — dias 15 a 21 | Fazer ajustes limitados; fixar a configuração; executar teste reservado e analisar erros | Modelo final, métricas e figuras para o relatório |
| Semana 4 — dias 22 a 28 | Finalizar notebook demonstrativo, texto, figuras e apresentação; reproduzir o fluxo completo | Protótipo demonstrável e entrega acadêmica |

**Documentação paralela:** introdução, objetivos e descrição dos dados na primeira semana; métodos, ambiente e registros dos treinamentos na segunda; resultados, gráficos e ficha do modelo na terceira; discussão, conclusão, manual de uso e revisão de reprodução na quarta. Os documentos iniciais estão no [índice da documentação](docs/README.md).

**Proteção do prazo proposta:** a partir do dia 22, priorizar correções e consolidação. A demonstração inicialmente planejada em notebook foi substituída, na execução, por uma interface web exclusivamente local, ligada ao mesmo processamento e aos pesos treinados. Aplicativo móvel, coleta própria e hardware permanecem fora do escopo.

**Dependência cardíaca:** se uma base de FA for obtida, estimar seu ciclo próprio de auditoria e treinamento sem retirar tempo das entregas essenciais. A disponibilidade de dados, por si só, não garante dois módulos concluídos no prazo.

## 12. Entregáveis e critérios de conclusão

- Documento do projeto com objetivos e recorte explícitos.
- Relatório acadêmico com introdução, fundamentação, metodologia executada, resultados, discussão, conclusão e referências.
- Documentação técnica com arquitetura, origem dos dados, parâmetros, ambiente, versões e instruções de reprodução conferidas.
- Manual de uso da demonstração e fichas de experimentos vinculadas aos resultados reais.
- Inventário rastreável e partições sem sobreposição de pacientes.
- Código do processamento, um modelo convencional e uma CNN para o módulo principal escolhido.
- Relatório com comparação nas mesmas partições, resultados por classe e análise de erros.
- Demonstração reproduzível na aplicação local, com quatro exemplos reais do teste escolhidos sem seleção por acerto.
- Registro das condições de uso dos dados e das limitações.
- Situação do módulo de FA explicitada: revisão e viabilidade documentadas, com experimento somente se dados e prazo permitirem.

Não será fixada uma “acurácia de 95%” como promessa de entrega. O critério científico é executar a comparação corretamente e relatar os resultados obtidos, inclusive se a CNN não superar o modelo convencional.

## 13. Estrutura sugerida do relatório acadêmico

1. Introdução e delimitação do problema.
2. Justificativa e objetivos.
3. Fundamentação: sons biológicos, processamento de sinais, machine learning e CNN.
4. Trabalhos relacionados.
5. Materiais e métodos.
6. Resultados experimentais.
7. Discussão, limitações e possibilidades de aplicação.
8. Conclusão e trabalhos futuros.
9. Referências e apêndices de reprodução.

## 14. Referências iniciais

[1] PETMEZAS, G. et al. Automated Lung Sound Classification Using a Hybrid CNN-LSTM Network and Focal Loss Function. *Sensors*, 22(3), 1232, 2022. https://doi.org/10.3390/s22031232 — texto: https://pmc.ncbi.nlm.nih.gov/articles/PMC8838187/

[2] ROCHA, B. M. et al. An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*, 40, 035001, 2019. https://doi.org/10.1088/1361-6579/ab03ea — portal da base: https://bhichallenge.med.auth.gr/node/50

[3] GUO, Y. et al. Detection of atrial fibrillation via artificial intelligence-assisted auscultation. *BMC Cardiovascular Disorders*, 25, 795, 2025. https://doi.org/10.1186/s12872-025-05236-z — a seção de disponibilidade informa acesso aos dados mediante solicitação aos autores.

[4] REYNA, M. A. et al. Heart murmur detection from phonocardiogram recordings: The George B. Moody PhysioNet Challenge 2022. *PLOS Digital Health*, 2(9), e0000324, 2023. https://doi.org/10.1371/journal.pdig.0000324 — dados e documentação: https://physionet.org/content/challenge-2022/1.0.0/

Fontes iniciais consultadas em 23/10/2026. O plano foi executado para o módulo pulmonar em 26/10/2026; consultar os relatórios para evidência, limitações e resultados reais.
