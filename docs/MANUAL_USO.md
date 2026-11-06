# Manual de uso — Ausculta Lab

Aplicação acadêmica local de classificação de ciclos respiratórios. A interface é web, servida pelo próprio computador; não é necessário executar notebook ou instalar Node.js para utilizar a aplicação.

## 1. Instalar no Windows

1. Extraia o ZIP inteiro para uma pasta curta, por exemplo C:\AuscultaLab ou uma pasta AuscultaLab diretamente no seu perfil de usuário. Não execute os arquivos de dentro do ZIP. O instalador informa o problema antes de baixar as dependências se o caminho do projeto tiver mais de 90 caracteres; isso evita a falha de instalação de caminhos longos do PyTorch observada no teste de entrega, sem modificar configurações do Windows.
2. Utilize Python de 64 bits. O ambiente de referência é Python 3.13; outras versões declaradas pelo pacote ainda precisam ser verificadas no ambiente do usuário.
3. Execute INSTALAR.bat. Ele chama Instalar.ps1, cria .venv, instala requirements.lock.txt e instala o pacote Python local.
4. Aguarde a mensagem de conclusão. A primeira instalação precisa de internet e pode demorar por causa das dependências de machine learning.

O ambiente .venv é criado no computador do usuário. O modelo e os 16 exemplos do pacote permitem usar a aplicação com uma instalação local.

## 2. Abrir e encerrar

Execute INICIAR.bat. Ele chama Iniciar.ps1 e abre a interface no navegador em:

    http://127.0.0.1:8765

Mantenha a janela do servidor aberta. Para encerrar, volte a ela e pressione Ctrl+C.

Alternativa pelo terminal, dentro da pasta do projeto e após a instalação:

    .\.venv\Scripts\python.exe run.py --port 8765

Para usar outra porta:

    .\.venv\Scripts\python.exe run.py --port 8766

Nesse caso, abra http://127.0.0.1:8766. O servidor utiliza o endereço de loopback; não é uma publicação para acesso externo.

## 3. Experimentar um exemplo real

1. Aguarde a indicação de modelo disponível.
2. Escolha um áudio na lista “Ou teste com um exemplo” e clique em “Carregar”.
3. Ouça o áudio e confira os limites de início e fim. A referência anotada aparece separadamente da previsão.
4. Clique em “Analisar ciclo respiratório”.
5. Confira a categoria prevista e as quatro pontuações.
6. Abra “Espectrograma e detalhes” para ver o gráfico, consultar avisos ou salvar o JSON.

Os 16 exemplos vêm da partição de teste, quatro por categoria. A seleção usa uma ordenação fixa e pacientes distintos dentro de cada categoria. A origem de cada ciclo está registrada em examples/index.json e examples/ORIGEM.md.

## 4. Analisar outro WAV

Selecione um arquivo ou arraste-o para a área de envio. Use início e fim para delimitar uma respiração completa e confirme a seleção antes de analisar. Ao alterar os limites, será necessário confirmar novamente.

| Requisito | Limite implementado |
| --- | --- |
| Formato | WAV real, com cabeçalho e conteúdo decodificáveis |
| Tamanho do envio | Até 20 MiB; a interface usa a indicação 20 MB |
| Duração da gravação | Positiva, até 120 segundos |
| Canais | Mono ou estéreo |
| Taxa de amostragem | Entre 2.000 e 192.000 Hz |
| Trecho selecionado | Entre 0,1 e 20 segundos, dentro da gravação |
| Conteúdo | Amostras finitas e sinal de amplitude suficiente |

A aplicação rejeita silêncio ou entradas tecnicamente inválidas. Ela não verifica automaticamente se um arquivo válido é uma ausculta pulmonar nem se o trecho escolhido contém uma respiração completa. A seleção do ciclo é manual nesta versão.

Mensagens de voz em OGG ou Opus são recusadas pelo formato. A conversão de fala para WAV permite a leitura técnica do arquivo e pode gerar uma categoria inadequada. Use gravações de ausculta feitas com estetoscópio.

## 5. Interpretar a saída

As categorias são “Sem sibilos/estertores”, “Estertores”, “Sibilos” e “Ambos”. A saída descreve padrões acústicos aprendidos; não constitui diagnóstico de uma doença.

As pontuações softmax expressam a preferência do modelo entre as categorias. Não equivalem a probabilidades clínicas calibradas. A anotação de um exemplo não é enviada como entrada para a rede.

O módulo cardíaco não está implementado. Nenhum resultado desta tela corresponde à detecção de arritmias.

O macro-F1 da CNN por faixas no teste foi 0,3611. A avaliação por classe está no [relatório da versão 1.1](../reports/REFINAMENTO_CNN.md).

## 6. Resolver problemas

| Situação | Ação |
| --- | --- |
| Falha na instalação | Conferir Python de 64 bits, conexão e mensagem da janela; repetir a instalação após resolver a causa |
| Porta ocupada | Encerrar a instância anterior ou iniciar com outra porta |
| Navegador não abriu | Acessar manualmente o endereço informado no terminal |
| Modelo indisponível | Conferir presença de models/cnn.pt e models/model.json do pacote; não editar os hashes manualmente |
| Arquivo recusado | Conferir formato real, tamanho, canais, duração e integridade |
| Botão de análise desabilitado | Carregar áudio, corrigir limites e confirmar o ciclo; conferir disponibilidade do modelo |
| Mensagem de silêncio | Selecionar outro trecho com sinal mensurável |
| Resultado diferente da anotação | Registrar como erro do modelo; não alterar o rótulo apenas para concordar com a previsão |

Os uploads são enviados ao servidor local para análise e não são gravados permanentemente pela aplicação. O download do resultado é uma ação separada do usuário.

## 7. Conferência para apresentação

Abra uma sessão nova do servidor, carregue um exemplo, repita a análise e exporte o JSON. Confira também uma entrada inválida. As verificações formais e suas evidências estão descritas em docs/QA_INDEPENDENTE.md e reports/qa.

A classificação de exemplos não demonstra eficácia clínica. Consulte os resultados agregados do teste em reports/RESULTADOS.md.
