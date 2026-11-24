# Registro da coleta: outros modelos (OpenRouter)

Pedido do orientador: testar o desempenho de outros modelos no desenho do
Experimento A. Piloto de N=30 por modelo, `--seed 123` (mesmos perfis dos 30
primeiros agentes do Experimento A), temperatura 0,7, BFI-44 + IPA,
raciocínio interno (*reasoning*) desligado, 8 s de pausa entre chamadas.

Comando (a partir da raiz do repositório): `python src/geracao/rodar_modelos_openrouter.py --n 30 --intervalo 8`
(retoma de onde parou; os logs detalhados de cada modelo ficam em `<modelo>.log`).

| Apelido | Modelo no OpenRouter |
|---|---|
| gemma4_31b | `google/gemma-4-31b-it:free` |
| nemotron35_super | `nvidia/nemotron-3.5-super-vl:free` |
| nemotron3_ultra | `nvidia/nemotron-3-ultra-550b-a55b:free` |

Conta: plano gratuito, sem créditos. Limite observado: **50 chamadas por dia**
nos modelos `:free` (mensagem da API: "Rate limit exceeded: free-models-per-day.
Add 10 credits to unlock 1000 free model requests per day").

## Histórico

### 2026-10-27

- **Teste inicial:** 1 chamada do IPA por modelo. Os três responderam no
  formato pedido e as 7 respostas foram interpretadas pelo código.
- **gemma4_31b:** provedor congestionado o dia todo (HTTP 429 "temporarily
  rate-limited upstream", vindo do provedor do Google, não do limite da conta).
  Após ~25 min de esperas progressivas, o script desistiu sem gerar agentes
  (`gemma4_31b.log`). Será tentado de novo nas próximas passadas.
- **nemotron35_super:** 11 agentes gerados, mas **o BFI-44 falhou em todos**.
  Causa: o modelo repete o texto de cada afirmação antes da resposta, e o
  teto de 1.024 tokens cortava a resposta por volta do item 42 de 44; o
  código não aceita BFI-44 incompleto (3 tentativas por agente, todas
  cortadas). O IPA foi lido nos 11.
  - **Correção:** teto de resposta para os modelos do OpenRouter elevado
    para 4.096 tokens (só limita o tamanho máximo, não muda o conteúdo) e
    aviso no log quando uma resposta vier cortada.
  - Os 11 agentes foram **descartados** e serão gerados de novo (mesmos
    perfis, pela mesma seed). Arquivos guardados como registro:
    `nemotron35_super_descartado_truncado.csv` / `.log`.
- **Limite diário** atingido às 12:10, durante o nemotron35_super
  (grande parte das 50 chamadas foi consumida pelas tentativas repetidas do
  BFI-44 cortado). Coleta retomada no dia seguinte.
- nemotron3_ultra: ainda não iniciado.

### 2026-10-28

- **gemma4_31b:** às 09:59, provedor congestionado de novo (mesmo HTTP 429
  "temporarily rate-limited upstream"). A rodada foi interrompida
  manualmente às ~10:01, durante as esperas (por isso o log desse trecho não
  tem linha FIM). A ordem do `rodar_modelos_openrouter.py` foi mudada para
  rodar o Gemma por último, para as esperas dele não atrasarem os outros.
- **nemotron35_super:** 25 agentes gerados (10:02 a 10:16), com **BFI-44 e
  IPA válidos em 25 de 25**, todos na 1ª tentativa e sem respostas cortadas.
  A correção do teto de tokens funcionou. Resumo parcial (N=25): perfis
  Conservador 14, Arrojado 7, Moderado 4; manipulação ρ de +0,92 a +0,99.
- **Limite diário** atingido às 10:16, logo após 50 chamadas do Nemotron (25
  agentes × 2). Os 429 do Gemma não parecem ter contado para a cota.
- Faltam: nemotron35_super 5/30, nemotron3_ultra 30/30, gemma4_31b 30/30.

### 2026-11-03

- Coleta retomada às ~09:20 (nada rodou entre 29/10 e 02/11).
- **nemotron35_super: concluído (30/30).**
- **nemotron3_ultra:** 21 agentes gerados (09:22 a 09:38), BFI-44 e IPA
  válidos em 21 de 21. Resumo parcial (N=21): perfis Arrojado 9,
  Conservador 8, Moderado 4; manipulação ρ de +0,92 a +0,95.
- Limite diário atingido às 09:38. Faltam: nemotron3_ultra 9/30, gemma4_31b 30/30.
- **Automação:** criada a tarefa "TCC - pilotos OpenRouter" no Agendador de
  Tarefas do Windows. Ela roda `rodar_diario_openrouter.ps1` (script local da
  automação, removido do repositório após o encerramento da coleta) todo dia às
  10:00 a partir de 04/11 (ou assim que o computador for ligado, se estiver
  desligado nesse horário). Cada execução usa a cota gratuita do dia e é
  registrada em `agendamento.log`. A tarefa se remove sozinha quando os 3
  modelos chegam a 30 agentes.
- Ajuste: a tarefa passou a rodar **4 vezes por dia** (10:00, 14:00, 18:00 e
  22:00), para tentar o Gemma em horários diferentes quando o provedor
  estiver congestionado. Se uma execução ainda estiver rodando, a seguinte é
  ignorada (sem duas rodadas ao mesmo tempo). Limite de 3 h por execução.

### 2026-11-04

- Primeira execução automática às 10:00.
- **nemotron3_ultra: concluído (30/30)** às 10:06. Resumo final (N=30): BFI-44
  e IPA válidos em 30 de 30; perfis Arrojado 12, Conservador 12, Moderado 6;
  IPA média 19,43; manipulação ρ: Extr +0,95, Afab +0,92, Cons +0,97,
  Neur +0,96, Aber +0,94.
- **gemma4_31b:** às 10:06 o provedor estava congestionado de novo (HTTP 429
  "temporarily rate-limited upstream"); o script entrou nas esperas
  progressivas e tentou de novo às 10:32 e às 10:57, sem resposta.
- O computador entrou em suspensão às 11:07 e só voltou em 05/11 às 09:57:
  as execuções de 14:00, 18:00 e 22:00 não ocorreram.

### 2026-11-05

- A coleta do Gemma iniciada em 04/11 às 10:57 ficou parada durante a
  suspensão e, ao voltar, gerou o **agente 0 do gemma4_31b** (válido; tempo
  registrado de 83.310 s por incluir a suspensão).
- Às 10:00 a execução agendada iniciou outra coleta do Gemma em paralelo (o
  limite de 3 h da tarefa tinha encerrado o script de agendamento, mas não o
  Python). As duas gravariam no mesmo CSV, com risco de agentes duplicados.
  Às ~10:10 as duas foram encerradas manualmente, antes de gravar qualquer
  outro agente (CSV conferido: 1 agente, sem duplicação).
- **Correções na automação:**
  - `rodar_diario_openrouter.ps1` agora encerra coletas antigas ainda abertas
    antes de começar (registrado no `agendamento.log`);
  - a tarefa passou a acordar o computador nos horários agendados
    (WakeToRun) e os despertadores do Windows foram ativados na tomada (na
    bateria seguem desligados).
- Nova coleta iniciada às 10:13, continuando do agente 1 (gemma4_31b 1/30).
- Às 10:39 o provedor liberou uma resposta: **agente 1 gerado** (válido, 1ª
  tentativa). Nas demais passadas (até 11:30) o provedor seguiu congestionado.
- Execução extra às 12:00 (gatilho único, só neste dia): sem resposta nas
  passadas de 12:00 e 12:25; a de 12:51 estava em andamento às 12:55.
  Situação às 12:55: gemma4_31b 2/30 (BFI-44 e IPA válidos em 2 de 2).

### 2026-11-06: encerramento do gemma4_31b

- O provedor seguiu congestionado nas execuções seguintes (as últimas
  recusas registradas são de 06/11, ~11:00).
- **Decisão da autora:** encerrar a coleta do Gemma 4 31B, tendo em vista o
  prazo de apresentação do TCC neste período. Às ~11:00 a coleta em andamento
  foi parada, a tarefa agendada "TCC - pilotos OpenRouter" foi removida e o
  despertador do Windows na tomada voltou à configuração anterior
  ("somente despertadores importantes").
- Balanço do Gemma 4 31B: tentativas em 5 dias (27/10, 28/10, 04/11, 05/11
  e 06/11), com 17 rodadas iniciadas, 128 recusas HTTP 429 do provedor e **2 agentes
  gerados** (ids 0 e 1, BFI-44 e IPA válidos na 1ª tentativa). Os 2 agentes
  ficam no CSV como registro, mas o modelo não entra na comparação.
- **Situação final:** nemotron35_super 30/30, nemotron3_ultra 30/30,
  gemma4_31b 2/30 (encerrado).
