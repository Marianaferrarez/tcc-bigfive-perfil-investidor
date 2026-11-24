# Traços de Personalidade Big Five e Perfil de Investidor: uma Simulação com Agentes de Inteligência Artificial

Trabalho de Conclusão de Curso (Bacharelado em Ciência da Computação, DCC/UFRRJ, 2026)
de **Mariana Ferreira Ferrarez Ribeiro**, orientado pelo Prof. Leandro Guimarães Marques Alvim.

O trabalho investiga se traços de personalidade do modelo Big Five, induzidos por
instrução (*prompt*) em agentes baseados em modelos de linguagem de grande escala (LLMs),
predizem o perfil de investidor desses agentes, medido por um questionário de
*suitability* (IPA). Cada agente recebe um nível (Baixo/Médio/Alto) sorteado para cada
um dos cinco traços, responde ao BFI-44 (checagem de manipulação) e ao IPA, e a relação
entre personalidade induzida e escore do IPA é testada por correlação de Spearman e
regressão OLS.

## Principais resultados

- A indução de personalidade foi fiel nos cinco traços (ρ entre 0,813 e 0,929 com o
  autorrelato no BFI-44, modelo `llama3.1:8b`, N = 300).
- **Abertura à Experiência** e **Extroversão** elevam o escore do IPA em todas as
  condições testadas: quatro temperaturas de geração, dois sorteios de perfis e quatro
  modelos abertos (`llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, `gemma3:12b`).
- Os efeitos de Consciência e Neuroticismo são instáveis entre condições, e Amabilidade
  não tem efeito consistente.
- O perfil de investidor absoluto atribuído a uma mesma personalidade varia fortemente
  entre modelos.

Detalhes, tabelas e discussão estão na monografia (`monografia/`).

## Estrutura do repositório

```
├── monografia/            Texto do TCC em LaTeX (template DCC/UFRRJ)
├── questionarios/         Instrumentos: BFI-10, BFI-44, IPA e descrições usadas na indução
├── src/
│   ├── geracao/           Geração dos agentes (Ollama local ou API do OpenRouter)
│   └── analise/           Análises estatísticas, tabelas e figuras usadas na monografia
├── notebooks/
│   ├── colab/             Notebooks que geraram os dados no Google Colab (GPU T4)
│   └── analise/           Notebooks de análise exploratória (piloto e Experimentos A e B)
├── resultados/
│   ├── principal/         Experimento A (análise principal) e Experimento B
│   ├── piloto/            Estudo piloto com BFI-10 e comparação inicial de modelos
│   ├── temperatura/       Experimento A com temperaturas 0,0, 0,3 e 1,0
│   ├── robustez/          Experimento A com outro sorteio de perfis (semente 456)
│   ├── modelos_locais/    Experimento A com qwen2.5:7b, mistral:7b e gemma3:12b
│   ├── modelos/           Piloto com modelos Nemotron via OpenRouter (N = 30)
│   ├── figuras/, tabelas/ Saídas usadas na monografia
│   └── *.json, *.md       Números consolidados das análises
├── exploratorio/          Scripts de teste usados no início do projeto (não usados nos resultados)
├── apresentacao/          Rascunho da apresentação
└── referencias/artigos/   Os dois artigos-base (acesso aberto)
```

As pastas `temperatura/`, `robustez/`, `modelos_locais/` e `modelos/` trazem, junto
de cada CSV, o log da geração; `modelos/` e `modelos_locais/` têm também um
`REGISTRO.md` com o histórico da coleta.

## Como reproduzir

Requisitos: Python 3.13 e as bibliotecas de `requirements.txt`.

```bash
pip install -r requirements.txt
```

Todos os comandos abaixo partem da raiz do repositório.

**Análises** (recriam os números, tabelas e figuras em `resultados/` a partir dos CSVs):

```bash
python src/analise/analise_aleatorizacao.py        # verificação do sorteio dos perfis
python src/analise/analise_intercorrelacoes.py     # correlação entre traços autorrelatados
python src/analise/analise_temperatura_modelos.py  # robustez e generalização
python src/analise/verificar_sorteio_seed456.py    # sorteio da rodada com semente 456
python src/analise/regen_figs.py                   # figuras do Experimento A
python src/analise/gerar_figs_exp_ab.py            # figuras Experimento A × B
```

**Geração de agentes** (exige um modelo servido pelo [Ollama](https://ollama.com) em
`localhost:11434`; para modelos do OpenRouter, defina `OPENROUTER_API_KEY` no ambiente
ou em um arquivo `.env` na raiz, que não é versionado):

```bash
python src/geracao/gerar_agentes.py --n 300 --output resultados/exemplo.csv \
    --seed 123 --modo multi --modelo llama3.1:8b --temperatura 0.7
```

`--seed` controla apenas o sorteio dos perfis de personalidade; com a mesma semente, os
perfis são idênticos agente a agente. Os dados deste trabalho com `llama3.1:8b` e os
modelos abertos foram gerados nos notebooks de `notebooks/colab/`.

**Monografia** (a partir de `monografia/`):

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Licença

Código sob licença MIT; dados, instrumentos e texto sob CC BY 4.0. Veja `LICENSE`.
