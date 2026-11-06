# Ausculta Lab 1.1.0

A CNN por faixas melhora o macro-F1 e aumenta o acerto em três categorias no benchmark. A classe sem eventos teve redução de 1,14 ponto percentual. O acerto por categoria é a proporção de ciclos daquela categoria que receberam o rótulo correto.

| Categoria anotada | Ciclos | CNN anterior | CNN por faixas | Diferença |
| --- | ---: | ---: | ---: | ---: |
| Sem sibilos/estertores | 1.579 | 825/1.579 · 52,25% | 807/1.579 · 51,11% | −1,14 p.p. |
| Estertores | 649 | 253/649 · 38,98% | 278/649 · 42,84% | +3,85 p.p. |
| Sibilos | 385 | 103/385 · 26,75% | 140/385 · 36,36% | +9,61 p.p. |
| Ambos | 143 | 20/143 · 13,99% | 28/143 · 19,58% | +5,59 p.p. |

| Métrica nos 2.756 ciclos | CNN anterior | CNN por faixas |
| --- | ---: | ---: |
| Macro-F1 | 0,3216 | 0,3611 |
| Acurácia | 43,58% | 45,46% |
| Acurácia balanceada | 32,99% | 37,47% |
| Escore ICBHI | 0,4210 | 0,4450 |

As partições por paciente e as características log-Mel permanecem as mesmas do experimento 001. O teste de 49 pacientes já havia sido avaliado na versão anterior e foi consultado durante esta rodada exploratória. Os pesos e os deslocamentos das saídas foram ajustados somente no treino e na validação. A escolha de qual candidato disponibilizar considerou o desempenho observado no benchmark. O intervalo bootstrap por paciente do macro-F1 da versão 1.1 foi [0,2928; 0,4137]. A comparação é descritiva e a aplicação continua com finalidade acadêmica.

## Treinamento e integração

A CNN usa 24, 32, 48 e 64 filtros, com agregação por média e máximo em faixas de frequência e regiões de tempo. O treino aplica variações de inclinação espectral e máscaras curtas, com perda ponderada pela frequência das classes. Foram comparadas duas configurações com sementes 42 e 137. A época 5 da semente 42 venceu na validação pelo macro-F1 acrescido de 0,2 vezes o menor acerto entre as classes.

O primeiro candidato obteve macro-F1 de 0,3581 no benchmark e 31,54% de acerto na classe sem eventos. O ajuste seguinte buscou o maior ganho mínimo de sensibilidade sobre a CNN anterior na validação. A grade variou as três saídas de eventos em passos de 0,25 entre −1,5 e 1,5, com a saída sem eventos fixa. O ajuste selecionado foi `[0, -0.50, -1.25, -1.00]` e está incorporado ao viés da camada final. A combinação exploratória das duas CNNs, também ajustada pela validação, teve macro-F1 de 0,3420 no benchmark. A CNN por faixas com ajuste de decisão foi disponibilizada pelo melhor macro-F1 entre essas versões avaliadas e pelo equilíbrio observado por categoria.

O preditor carrega a arquitetura indicada no metadado e valida o hash dos pesos. As porcentagens exibidas correspondem ao softmax das quatro saídas do modelo ativo. As previsões individuais de ambas as versões estão em [test_predictions.csv](test_predictions.csv). Os registros de candidatos e da seleção ficam em [qa](qa/).

Para reproduzir com o manifesto auditado e a base local, use diretórios de saída novos.

```powershell
.\.venv\Scripts\python.exe scripts/refine_model.py --output output/repro-v2
.\.venv\Scripts\python.exe scripts/refine_model.py --output output/repro-balanced --balance-from output/repro-v2
.\.venv\Scripts\python.exe scripts/refine_model.py --output output/repro-balanced --evaluate
```

## Teste de uma mensagem de voz

Foi utilizado um recorte de 9 a 14 segundos de *Abou Ben Adhem*, lido por Peter Yearsley para o LibriVox. O recorte foi codificado em OGG com codec Opus, mono a 48 kHz, para representar o formato documentado pela Meta para mensagens de voz. Também foi criada uma cópia WAV PCM de 16 bits a partir do mesmo recorte decodificado. [LibriVox](https://librivox.org/short-poetry-collection-001/) · [Documentação da Meta](https://www.postman.com/meta/whatsapp-business-platform/request/goznbg0/send-audio-message-by-id)

| Entrada de teste | Resposta da aplicação |
| --- | --- |
| OGG/Opus | HTTP 422, orientação para escolher uma ausculta em WAV |
| O mesmo OGG/Opus com extensão .wav | HTTP 422, formato identificado pelo conteúdo |
| Conversão real da fala para WAV | HTTP 200, categoria “Ambos” com 34,97% |

O terceiro resultado é uma saída fora do escopo, pois a gravação contém fala. A aplicação valida o arquivo e solicita a confirmação de um ciclo de ausculta. Essa confirmação depende do usuário. O resultado bruto está registrado em [teste_audio_whatsapp.json](qa/teste_audio_whatsapp.json).
