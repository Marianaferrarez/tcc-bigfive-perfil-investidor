# Registro da coleta: modelos abertos no Colab (Ollama, GPU T4)

Complemento ao pedido do orientador de testar outros modelos: mesmo desenho do
Experimento A (5 traços simultâneos, BFI-44 + IPA, N=300, `--seed 123`,
temperatura 0,7), portanto os mesmos perfis do Experimento A, agente a agente.
Teto de resposta de 1.536 tokens (o llama3.1:8b usou 512), porque alguns
modelos repetem o texto dos itens do BFI-44. Notebook: `colab_testes_extras.ipynb`.

| Modelo (Ollama) | Arquivo |
|---|---|
| qwen2.5:7b | `exp_a_qwen25_7b.csv` / `.log` |
| mistral:7b | `exp_a_mistral_7b.csv` / `.log` |
| gemma3:12b | `exp_a_gemma3_12b.csv` / `.log` |

## Histórico

### 2026-10-28

- Primeira tentativa (18:45 qwen2.5:7b, 19:35 mistral:7b, 19:40 gemma3:12b):
  **nenhum agente gerado** nos três. Cada log tem só a linha INÍCIO e cada
  CSV só o cabeçalho: o `gerar_agentes.py` parou logo no começo sem registrar
  o motivo. Até então, erros de conexão ou de timeout com o Ollama não eram
  tratados e iam só para a tela do Colab, não para o log.
- **Correção:**
  - o `gerar_agentes.py` passou a tentar de novo as chamadas ao Ollama que
    falham (5 tentativas, 30 s entre elas, timeout de 600 s) e a registrar
    cada erro no log;
  - qualquer erro inesperado agora é registrado no log com o traceback;
  - o notebook passou a reiniciar o Ollama e a retomar a rodada
    automaticamente (até 3 vezes).
- As três rodadas serão refeitas. Os arquivos vazios desta tentativa ficam
  como estão: a nova rodada usa o cabeçalho e continua do agente 0.

### 2026-10-31 a 2026-11-05: rodadas completas

As sessões do Colab caíram algumas vezes; cada rodada foi retomada do ponto
em que parou (linhas INÍCIO com "já existentes" > 0 no log):

| Modelo | Retomadas (agentes já existentes) | Fim |
|---|---|---|
| qwen2.5:7b | 31/10 (0), 31/10 (62), 03/11 (248) | 03/11 13:13 |
| mistral:7b | 03/11 (0), 03/11 (64) | 03/11 15:48 |
| gemma3:12b | 03/11 (0), 05/11 (63), 05/11 (189) | 05/11 17:51 |

Nenhum registro de "erro na chamada", "PARADA" ou "ERRO INESPERADO" nos
logs. Uma resposta cortada pelo teto de 1.536 tokens no mistral:7b.

**Conferência (06/11):** nos três CSVs, `agente_id` de 0 a 299 sem repetição;
níveis dos 5 traços e Q1 atribuída idênticos aos do Experimento A, agente a
agente (as retomadas não alteraram o sorteio).

| Modelo | BFI-44 válido | IPA válido | Perfis (log, com Q1) |
|---|---|---|---|
| qwen2.5:7b | 286/300 | 257/300 | Arrojado 115, Conservador 11, Moderado 131 |
| mistral:7b | 294/300 | 300/300 | Arrojado 275, Moderado 25 |
| gemma3:12b | 300/300 | 300/300 | Arrojado 50, Conservador 72, Moderado 178 |

Análise: `analise_temperatura_modelos.py` →
`resultados/comparacao_temperatura_modelos.md` (seção "Modelos abertos locais").
O zip `drive-download-20261106T135600Z-1-001.zip` tem os mesmos 6 arquivos
(tamanhos idênticos), e o de 28/10, os arquivos vazios da primeira tentativa.
