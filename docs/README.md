# Documentação do Ausculta Lab

**Registro preenchido:** [Experimento 001](EXPERIMENTO_001.md), com dados, configuração, resultados reais e decisões. A ficha genérica permanece como modelo para próximas execuções.

Documentação da aplicação local implementada, dos dados reais e do procedimento de avaliação. O planejamento original de um mês permanece identificado como histórico; a demonstração evoluiu de notebook para interface web local.

A referência para métricas finais é models/metrics.json e reports/RESULTADOS.md. A evidência dos testes e retestes fica em reports/qa e docs/QA_INDEPENDENTE.md. Não inferir conclusão clínica ou aprovação de testes apenas da presença de código ou documentos.

## Começar

1. Extrair o pacote completo.
2. Executar INSTALAR.bat para criar o ambiente e instalar as dependências.
3. Executar INICIAR.bat e acessar http://127.0.0.1:8765.
4. Carregar um exemplo real, conferir o ciclo e analisar.

O modelo, quatro exemplos reais e a documentação acompanham o pacote. A base completa, o cache de treinamento e .venv ficam fora dele. Consulte o [manual](MANUAL_USO.md) para os limites e a solução de problemas.

## Documentos

| Documento | Finalidade |
| --- | --- |
| [Como funciona](COMO_FUNCIONA.md) | Dados, representação, aprendizado e utilização |
| [Manual de uso](MANUAL_USO.md) | Instalação, inicialização, exemplos, WAV próprio e interpretação |
| [Documentação técnica](DOCUMENTACAO_TECNICA.md) | Arquitetura executada, API, parâmetros, modelos e reprodução |
| [Dados e proveniência](DADOS.md) | Fonte, termos, checksums, correção de sobreposição e contagens |
| [Proposta científica](../PROJETO.md) | Problema, objetivos e delimitação acadêmica |
| [Relatório acadêmico](RELATORIO_ACADEMICO.md) | Fundamentação, métodos, resultados e discussão do trabalho |
| [Plano e estado da entrega](PLANO_IMPLEMENTACAO.md) | Cronograma inicial, decisões implementadas e critérios de conclusão |
| [Ficha de experimento](REGISTRO_EXPERIMENTO.md) | Modelo para registrar futuras execuções, não uma execução fictícia |
| [Esquema inicial do manifesto](manifesto.template.csv) | Referência documental do planejamento; o esquema executado está no plano/técnica |
| [Cabeçalho de experimentos](experimentos.template.csv) | Modelo para indexar execuções futuras |

Documentos e artefatos adicionais da versão de entrega: reports/RESULTADOS.md, docs/QA_INDEPENDENTE.md, models/model.json, models/metrics.json, reports/test_predictions.csv e examples/ORIGEM.md.

## Verificar depois de instalar

Na pasta do projeto:

    .\.venv\Scripts\python.exe -m pytest --require-real-artifacts
    .\.venv\Scripts\python.exe scripts/retest.py

Para reprocessar os WAVs originais do teste e compará-los ao experimento salvo:

    .\.venv\Scripts\python.exe scripts/retest.py --full

A última modalidade exige os WAVs originais, manifesto, cache e resultados do mesmo experimento. Reextrai os ciclos e compara suas representações finitas ao cache com tolerância absoluta de quatro epsilons float32 (aproximadamente 4,7684e-7), antes de exigir previsões idênticas e métricas dentro de 1e-12. O relatório completo é separado do reteste leve. Não necessita ser executada para utilizar a aplicação com os exemplos distribuídos.

## Manutenção e limites

Atualizar configurações, resultados e evidências quando houver novo treinamento. Manter separados validação e teste e identificar a unidade de avaliação. Não reutilizar o teste para escolher a melhor configuração nem selecionar exemplos apenas porque foram classificados corretamente.

O componente implementado é respiratório. O módulo de arritmia continua indisponível por falta de dados adequados. A aplicação é um protótipo acadêmico, com seleção manual do ciclo e sem validação para uso assistencial.
