# Ficha de experimento — modelo para preenchimento

**Estado:** modelo documental, sem execução ou resultado registrado. Criar uma ficha por execução real, identificada pelo mesmo `experiment_id` do índice CSV.

## 1. Identificação

| Campo | Valor |
| --- | --- |
| Identificador do experimento | A preencher |
| Data, horário e responsável | A preencher |
| Objetivo e hipótese específica | A preencher |
| Tarefa, classes e unidade de avaliação | A preencher |
| Etapa: desenvolvimento, validação ou teste final | A preencher |
| Versão do código ou referência ao snapshot | A preencher |

## 2. Dados e partições

Registrar base e versão, caminho do manifesto e configuração das partições; contagens de pacientes, gravações e segmentos por classe; exclusões; verificação de duplicatas e agrupamento de visitas. Identificar onde está a evidência de que pacientes não atravessam partições.

## 3. Configuração e ambiente

Registrar caminho da configuração completa, preparação do áudio, características, modelo, hiperparâmetros, semente, dependências, equipamento e origem dos pesos. Registrar tempo e recursos medidos.

## 4. Seleção e regras de avaliação

Descrever a métrica usada para escolher a configuração na validação, quantas configurações foram consideradas e a regra de agregação de segmentos. Se houver limiar de decisão, registrar como foi escolhido antes do teste.

Para uma execução de teste final, identificar a configuração já fixada e a data em que foi congelada. Uma decisão posterior baseada nesse resultado não deve ser apresentada como parte de uma avaliação independente inalterada.

## 5. Resultados

Preencher após a execução, mantendo validação e teste separados. Usar “não aplicável” para métricas sem definição na tarefa escolhida e explicar o motivo; campo sem medição permanece “pendente”.

| Medida | Validação | Teste reservado | Definição/arquivo de evidência |
| --- | --- | --- | --- |
| Macro-F1 | Pendente | Pendente | A preencher |
| Acurácia balanceada | Pendente | Pendente | A preencher |
| Sensibilidade por classe | Pendente | Pendente | A preencher |
| Especificidade, com definição da classe negativa | Pendente | Pendente | A preencher |
| Escore ICBHI, se aplicável | Pendente | Pendente | A preencher |
| ROC-AUC/PR-AUC, se aplicáveis | Pendente | Pendente | A preencher |

Anexar matriz de confusão, contagens por classe, curvas quando aplicáveis e os intervalos de confiança que tiverem sido calculados. Informar o método de cálculo e sua unidade de reamostragem.

## 6. Análise e decisão

Registrar padrões de erro observados, limitações, comparação com o modelo de referência nas mesmas partições e interpretação dos resultados. Separar observação de hipótese explicativa. Registrar a próxima decisão e quais dados podem ser usados para tomá-la.

## 7. Evidências e reprodução

| Artefato | Localização |
| --- | --- |
| Configuração e dependências | A preencher |
| Manifesto e definição das partições | A preencher |
| Modelo salvo | A preencher |
| Log e métricas exportadas | A preencher |
| Predições e figuras | A preencher |
| Comando ou notebook realmente executado | A preencher |
| Execução de reprodução e diferenças observadas | A preencher |

Concluir a ficha e adicionar sua referência em `experimentos.template.csv` ou no índice de experimentos criado a partir dele.
