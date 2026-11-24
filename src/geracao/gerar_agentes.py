"""
Pipeline principal: gera N "agentes" com personalidade Big Five induzida
(níveis sorteados aleatoriamente: Baixo/Médio/Alto em cada um dos 5 traços),
faz cada agente responder o BFI-10 (autorrelato de personalidade, usado
como checagem de manipulação) e o questionário IPA (perfil de investidor),
e salva tudo em um CSV para análise estatística posterior.

Uso:
    python src/geracao/gerar_agentes.py --n 20 --output resultados/piloto.csv
    python src/geracao/gerar_agentes.py --n 300 --output resultados/x.csv --seed 123 --temperatura 0.0

    # modelos do OpenRouter (precisa de OPENROUTER_API_KEY no ambiente ou em .env)
    python src/geracao/gerar_agentes.py --n 30 --output resultados/modelos/gemma4.csv --seed 123 \\
        --backend openrouter --modelo google/gemma-4-31b-it:free --intervalo 8

Retomar: rodar o mesmo comando de novo (mesmo --output, --seed e --modo). O
script detecta os agentes já gerados, avança o sorteio até o ponto certo e
continua; --n é o número de agentes NOVOS a gerar.

Log: cada agente e um resumo parcial (a cada --resumo-cada agentes) são
anotados em <output>.log (ou --log), para registro do andamento.
"""

import argparse
import csv
import difflib
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime

import requests

sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/generate"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TENTATIVAS = 3
MAX_TOKENS = 512              # Ollama (piloto e Experimentos A e B)
MAX_TOKENS_OPENROUTER = 4096  # modelos do OpenRouter

# Preenchido em main() a partir dos argumentos; os valores abaixo são os
# padrões históricos (Ollama local, temperatura 0,7).
CONFIG = {
    "backend": "ollama",
    "modelo": os.environ.get("AGENTE_MODEL", "llama3.2:3b"),
    "temperatura": 0.7,
    "intervalo": 0.0,
    "api_key": None,
    "max_tokens": None,  # None = MAX_TOKENS (Ollama) / MAX_TOKENS_OPENROUTER
}

NIVEIS = ["Baixo", "Médio", "Alto"]
NIVEL_NUM = {"Baixo": 1, "Médio": 2, "Alto": 3}

# Mapeia o nome da dimensão no BFI-10 para o nome usado em perfis_personalidade.json
DIM_BFI_PARA_PERFIL = {
    "Extroversão": "Extroversão",
    "Afabilidade": "Afabilidade",
    "Consciência": "Consciência",
    "Neuroticismo": "Neuroticismo",
    "Abertura à Experiência": "Abertura à Experiência",
}

# Raiz do repositório (este script fica em src/<pasta>/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(BASE_DIR, "questionarios", "bfi44.json"), encoding="utf-8") as f:
    BFI44 = json.load(f)

with open(os.path.join(BASE_DIR, "questionarios", "ipa.json"), encoding="utf-8") as f:
    IPA = json.load(f)

with open(os.path.join(BASE_DIR, "questionarios", "perfis_personalidade.json"), encoding="utf-8") as f:
    PERFIS = json.load(f)["dimensoes"]

DIMENSOES = list(PERFIS.keys())

# Lookups para parsing de respostas textuais
ESCALA_BFI_TEXTO = {v.lower(): int(k) for k, v in BFI44["escala"].items()}

IPA_TEXTO_POR_QUESTAO = {
    q["id"]: {op["texto"].lower(): letra for letra, op in q["opcoes"].items()}
    for q in IPA["perguntas"]
}


# ---------------------------------------------------------------------------
# Sorteio e construção de prompts
# ---------------------------------------------------------------------------

def sortear_personalidade(modo="multi"):
    """modo='multi' (padrão): sorteia Baixo/Médio/Alto para os 5 traços,
    de forma independente (desenho igual ao dataset piloto BFI-10).

    modo='dominante': sorteia 1 dos 5 traços para ser o "traço dominante"
    do agente (varia Baixo/Médio/Alto); os outros 4 ficam fixos em "Médio".
    Usado no Experimento B (traço único) para testar se induzir os 5 traços
    simultaneamente está diluindo/confundindo o sinal de cada traço
    individual, em comparação com manipular só 1 por vez.
    """
    if modo == "multi":
        return {dim: random.choice(NIVEIS) for dim in DIMENSOES}, None
    elif modo == "dominante":
        dominante = random.choice(DIMENSOES)
        personalidade = {dim: ("Médio" if dim != dominante else random.choice(NIVEIS)) for dim in DIMENSOES}
        return personalidade, dominante
    else:
        raise ValueError(f"modo inválido: {modo!r} (use 'multi' ou 'dominante')")


def montar_system_prompt(personalidade):
    linhas = ["Você deve responder como uma pessoa com o seguinte perfil de personalidade:"]
    for dim, nivel in personalidade.items():
        linhas.append(f"- {dim} ({nivel}): {PERFIS[dim][nivel]}")
    linhas.append(
        "\nResponda TODAS as perguntas a seguir como essa pessoa responderia, "
        "de forma consistente com esse perfil. Não explique, não justifique, "
        "apenas dê as respostas no formato pedido."
    )
    return "\n".join(linhas)


def montar_prompt_bfi44():
    opcoes_str = " | ".join(f'"{v}"' for v in BFI44["escala"].values())
    linhas = [
        "Para cada afirmação abaixo, escreva o texto exato da opção escolhida logo após o símbolo →.",
        f"Opções: {opcoes_str}",
        "",
    ]
    for item in BFI44["itens"]:
        linhas.append(f"{item['id']}. Eu me vejo como alguém que {item['texto']} →")
    return "\n".join(linhas)


def montar_prompt_ipa():
    # Q1 (faixa etária) é atribuída aleatoriamente em gerar_agente() — ver nota abaixo.
    # O prompt apresenta apenas Q2-Q8 para o modelo responder.
    linhas = [
        "Para cada pergunta abaixo, escolha a opção que melhor representa você.",
        "Escreva o texto exato da opção escolhida, sem alterar nada.",
        "",
    ]
    for q in IPA["perguntas"]:
        if q["id"] == 1:
            continue  # faixa etária não se aplica a agentes de IA; atribuída aleatoriamente
        linhas.append(f"{q['id']}. {q['texto']}")
        for letra, op in q["opcoes"].items():
            linhas.append(f"   - {op['texto']}")
        linhas.append("")
    linhas.append(
        "Responda com exatamente 7 linhas (uma por pergunta, na ordem acima, perguntas 2 a 8). "
        "Em cada linha, escreva APENAS o texto exato da opção escolhida, "
        "sem letras, sem números, sem qualquer outro texto.\n"
        "Exemplo (primeiras 2 linhas):\n"
        "Entre 26% e 50%\n"
        "No máximo 5 anos"
    )
    linhas.append("\nResposta:")
    return "\n".join(linhas)


PROMPT_BFI44 = montar_prompt_bfi44()
PROMPT_IPA = montar_prompt_ipa()


# ---------------------------------------------------------------------------
# Chamada ao modelo (Ollama local ou OpenRouter)
# ---------------------------------------------------------------------------

class LimiteDiarioAtingido(Exception):
    """O OpenRouter recusou por limite diário de chamadas gratuitas. O agente
    em andamento é descartado (não é gravado) e o script para; basta rodar o
    mesmo comando no dia seguinte para retomar."""


class ProvedorIndisponivel(Exception):
    """O provedor do modelo seguiu recusando (congestionamento, erro 5xx)
    depois de todas as esperas. O agente em andamento é descartado e o script
    para sem erro; rodar o mesmo comando mais tarde retoma."""


def chamar_ollama(system_prompt, prompt, max_esperas=4):
    # Erros de conexão/timeout (servidor caiu, modelo lento demais) são
    # registrados no log e tentados de novo; esgotadas as tentativas, a rodada
    # para sem perder o que já foi gerado.
    for espera in range(max_esperas + 1):
        inicio = time.time()
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": CONFIG["modelo"],
                    "system": system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": CONFIG["temperatura"], "num_predict": CONFIG["max_tokens"] or MAX_TOKENS},
                },
                timeout=600,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            if espera == max_esperas:
                raise ProvedorIndisponivel(f"Ollama ({CONFIG['modelo']}) falhou {max_esperas + 1} vezes: {str(e)[:200]}")
            registrar(f"erro na chamada ao Ollama ({str(e)[:200]}); nova tentativa em 30s")
            time.sleep(30)
            continue
        dur = time.time() - inicio
        dados = resp.json()
        if dados.get("done_reason") == "length":
            registrar(f"aviso: resposta cortada pelo limite de {CONFIG['max_tokens'] or MAX_TOKENS} tokens")
        return dados["response"].strip(), dur


def chamar_openrouter(system_prompt, prompt, max_esperas=8):
    """Chamada ao OpenRouter com espera progressiva em erro 429/5xx.
    O raciocínio interno (reasoning) é desligado para que a resposta seja
    comparável à dos modelos do Ollama, que respondem direto."""
    corpo = {
        "model": CONFIG["modelo"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": CONFIG["temperatura"],
        # Teto folgado: alguns modelos repetem o texto de cada item antes da
        # resposta, e 44 itens assim passam de 1.000 tokens (respostas eram
        # cortadas no item ~42 com o teto antigo). O teto não muda o conteúdo.
        "max_tokens": CONFIG["max_tokens"] or MAX_TOKENS_OPENROUTER,
        "reasoning": {"enabled": False},
    }
    cabecalho = {"Authorization": f"Bearer {CONFIG['api_key']}", "Content-Type": "application/json"}
    for espera_n in range(max_esperas):
        inicio = time.time()
        try:
            resp = requests.post(OPENROUTER_URL, json=corpo, headers=cabecalho, timeout=300)
        except requests.RequestException as e:
            espera = min(300, 15 * 2 ** espera_n)
            registrar(f"erro de rede ({e.__class__.__name__}); nova tentativa em {espera}s")
            time.sleep(espera)
            continue
        dur = time.time() - inicio
        if resp.status_code == 200:
            dados = resp.json()
            if "error" in dados:
                registrar(f"erro do provedor: {str(dados['error'])[:200]}")
                time.sleep(min(300, 15 * 2 ** espera_n))
                continue
            escolha = dados["choices"][0]
            texto = (escolha["message"].get("content") or "").strip()
            if escolha.get("finish_reason") == "length":
                registrar(f"aviso: resposta cortada pelo limite de {CONFIG['max_tokens'] or MAX_TOKENS_OPENROUTER} tokens")
            return texto, dur
        msg = resp.text[:300]
        if resp.status_code == 429 and ("per-day" in msg or "per day" in msg.lower()):
            raise LimiteDiarioAtingido(msg)
        if resp.status_code in (429, 500, 502, 503, 504, 524, 529):
            espera = resp.headers.get("Retry-After")
            espera = float(espera) if espera and espera.replace(".", "", 1).isdigit() else min(300, 20 * 2 ** espera_n)
            registrar(f"HTTP {resp.status_code}; esperando {espera:.0f}s ({msg[:120]})")
            time.sleep(espera)
            continue
        resp.raise_for_status()
    raise ProvedorIndisponivel(f"{CONFIG['modelo']} não respondeu após {max_esperas} esperas")


def chamar_modelo(system_prompt, prompt):
    if CONFIG["backend"] == "openrouter":
        texto, dur = chamar_openrouter(system_prompt, prompt)
    else:
        texto, dur = chamar_ollama(system_prompt, prompt)
    if CONFIG["intervalo"] > 0:
        time.sleep(CONFIG["intervalo"])
    return texto, dur


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------

LOG = {"arquivo": None}


def registrar(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha)
    if LOG["arquivo"]:
        with open(LOG["arquivo"], "a", encoding="utf-8") as f:
            f.write(linha + "\n")


def resumo_parcial(caminho_csv):
    """Resumo do que já foi gerado: validade das respostas, perfis do IPA e
    checagem de manipulação (Spearman nível induzido x autorrelato)."""
    with open(caminho_csv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    n = len(linhas)
    if n == 0:
        return "nenhum agente ainda"
    bfi_ok = [l for l in linhas if l[f"bfi_{DIMENSOES[0]}"] != ""]
    ipa_ok = [l for l in linhas if l["ipa_perfil"] not in ("", None)]
    perfis = {}
    for l in ipa_ok:
        perfis[l["ipa_perfil"]] = perfis.get(l["ipa_perfil"], 0) + 1
    partes = [f"N={n}", f"BFI válido={len(bfi_ok)}/{n}", f"IPA válido={len(ipa_ok)}/{n}",
              "perfis=" + ", ".join(f"{k} {v}" for k, v in sorted(perfis.items()))]
    if ipa_ok:
        escores = [int(l["ipa_score"]) for l in ipa_ok]
        partes.append(f"IPA média={sum(escores) / len(escores):.2f}")
    if len(bfi_ok) >= 5:
        try:
            from scipy import stats
            rhos = []
            for d in DIMENSOES:
                x = [int(l[f"nivel_{d}_num"]) for l in bfi_ok]
                y = [float(l[f"bfi_{d}"]) for l in bfi_ok]
                r = stats.spearmanr(x, y)[0] if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")
                rhos.append(f"{d[:4]} {r:+.2f}" if not math.isnan(r) else f"{d[:4]} n/d")
            partes.append("manipulação ρ: " + ", ".join(rhos))
        except ImportError:
            pass
    return " | ".join(partes)


# ---------------------------------------------------------------------------
# Parsing das respostas
# ---------------------------------------------------------------------------

def _limpar_linhas(texto):
    """Remove linhas vazias e prefixos numéricos como '1. ' ou '1) '."""
    linhas = []
    for linha in texto.strip().splitlines():
        linha = linha.strip()
        if not linha:
            continue
        linha = re.sub(r"^\d+[.)]\s*", "", linha)
        linha = re.sub(r"^[Pp]ergunta\s+\d+[.:)]\s*", "", linha)
        if linha:
            linhas.append(linha)
    return linhas


def _match_escala_bfi(resposta):
    """Mapeia texto de resposta para nota BFI (1-5). Retorna None se não encontrar."""
    resp = resposta.lower().strip()
    if resp in ESCALA_BFI_TEXTO:
        return ESCALA_BFI_TEXTO[resp]
    for texto, nota in ESCALA_BFI_TEXTO.items():
        if texto in resp or resp in texto:
            return nota
    # fuzzy match para typos do modelo (ex: "totalmete" → "totalmente")
    matches = difflib.get_close_matches(resp, ESCALA_BFI_TEXTO.keys(), n=1, cutoff=0.75)
    if matches:
        return ESCALA_BFI_TEXTO[matches[0]]
    return None


def _match_opcao_ipa(resposta, q_id):
    """Mapeia texto de resposta para letra de opção IPA. Retorna None se não encontrar."""
    resp = resposta.lower().strip()
    lookup = IPA_TEXTO_POR_QUESTAO[q_id]
    if resp in lookup:
        return lookup[resp]
    for texto, letra in lookup.items():
        if resp in texto or texto in resp:
            return letra
    # word overlap como último recurso
    resp_palavras = set(resp.split())
    melhor_letra, melhor_score = None, 0
    for texto, letra in lookup.items():
        score = len(resp_palavras & set(texto.split()))
        if score > melhor_score:
            melhor_score, melhor_letra = score, letra
    return melhor_letra


def parse_bfi44(texto):
    """Extrai 44 notas a partir do formato '1. resposta' gerado pelo modelo."""
    linhas = _limpar_linhas(texto)  # remove prefixos numéricos como "8. "
    notas = []
    for linha in linhas:
        nota = _match_escala_bfi(linha)
        if nota is not None:
            notas.append(nota)
    notas = notas[:44]
    if len(notas) != 44:
        return None
    return notas


def parse_ipa_sem_q1(texto):
    """Extrai 7 letras (a-d) para Q2-Q8 a partir de respostas em texto."""
    linhas = _limpar_linhas(texto)
    if len(linhas) < 7:
        return None
    q_ids = [q["id"] for q in IPA["perguntas"] if q["id"] != 1]
    letras = []
    for i, q_id in enumerate(q_ids):
        if i >= len(linhas):
            break
        letra = _match_opcao_ipa(linhas[i], q_id)
        if letra is not None:
            letras.append(letra)
    if len(letras) != 7:
        return None
    return letras


def parse_ipa(texto):
    """Extrai 8 letras (a-d). Mantido para compatibilidade com scripts de teste."""
    letras = re.findall(r"[a-dA-D]", texto)
    letras = [c.lower() for c in letras][:8]
    if len(letras) != 8:
        return None
    return letras


# ---------------------------------------------------------------------------
# Cálculo de escores
# ---------------------------------------------------------------------------

def calcular_bfi44(notas):
    """Recebe lista de 44 notas (na ordem dos itens do BFI44) e retorna
    a média por dimensão, já aplicando a reversão quando necessário."""
    soma = {dim: 0.0 for dim in DIMENSOES}
    conta = {dim: 0 for dim in DIMENSOES}
    for item, nota in zip(BFI44["itens"], notas):
        valor = (6 - nota) if item["reverso"] else nota
        soma[item["dimensao"]] += valor
        conta[item["dimensao"]] += 1
    return {dim: soma[dim] / conta[dim] for dim in DIMENSOES}


def calcular_ipa(letras):
    total = sum(q["opcoes"][letra]["pontos"] for q, letra in zip(IPA["perguntas"], letras))
    for faixa in IPA["pontuacao_perfil"]["classificacao"]:
        ini, fim = faixa["faixa_soma"].split(" a ")
        if int(ini) <= total <= int(fim):
            return total, faixa["perfil"]
    return total, None


# ---------------------------------------------------------------------------
# Geração de um agente
# ---------------------------------------------------------------------------

def gerar_agente(agente_id, modo="multi"):
    personalidade, traco_dominante = sortear_personalidade(modo)
    system_prompt = montar_system_prompt(personalidade)

    resp_bfi44, notas, t_bfi44, tent_bfi = None, None, None, 0
    for tent_bfi in range(1, MAX_TENTATIVAS + 1):
        resp_bfi44, t_bfi44 = chamar_modelo(system_prompt, PROMPT_BFI44)
        notas = parse_bfi44(resp_bfi44)
        if notas is not None:
            break

    # Q1 (faixa etária) atribuída aleatoriamente — LLMs não têm idade real e
    # apresentam viés de resposta sistemático nessa pergunta (100% "a" observado
    # no dataset piloto de 299 agentes com llama3.2:3b).
    q1_aleatorio = random.choice(["a", "b", "c", "d"])

    resp_ipa, letras_q2q8, t_ipa, tent_ipa = None, None, None, 0
    for tent_ipa in range(1, MAX_TENTATIVAS + 1):
        resp_ipa, t_ipa = chamar_modelo(system_prompt, PROMPT_IPA)
        letras_q2q8 = parse_ipa_sem_q1(resp_ipa)
        if letras_q2q8 is not None:
            break
    letras = [q1_aleatorio] + letras_q2q8 if letras_q2q8 is not None else None

    linha = {"agente_id": agente_id, "traco_dominante": traco_dominante or ""}

    for dim in DIMENSOES:
        linha[f"nivel_{dim}"] = personalidade[dim]
        linha[f"nivel_{dim}_num"] = NIVEL_NUM[personalidade[dim]]

    if notas is not None:
        bfi_scores = calcular_bfi44(notas)
        for dim in DIMENSOES:
            linha[f"bfi_{dim}"] = round(bfi_scores[dim], 2)
        linha["bfi_raw"] = ",".join(str(n) for n in notas)
    else:
        for dim in DIMENSOES:
            linha[f"bfi_{dim}"] = ""
        linha["bfi_raw"] = f"ERRO: {resp_bfi44!r}"

    linha["ipa_q1_atribuida"] = q1_aleatorio
    if letras is not None:
        score, perfil = calcular_ipa(letras)
        linha["ipa_respostas"] = ",".join(letras)
        linha["ipa_score"] = score
        linha["ipa_perfil"] = perfil
    else:
        linha["ipa_respostas"] = f"ERRO: {resp_ipa!r}"
        linha["ipa_score"] = ""
        linha["ipa_perfil"] = ""

    linha["tempo_bfi44_s"] = round(t_bfi44, 2) if t_bfi44 else ""
    linha["tempo_ipa_s"] = round(t_ipa, 2) if t_ipa else ""
    linha["modelo"] = CONFIG["modelo"]
    linha["temperatura"] = CONFIG["temperatura"]
    linha["tentativas_bfi44"] = tent_bfi
    linha["tentativas_ipa"] = tent_ipa

    return linha


def avancar_sorteio(n_existentes, modo):
    """Consome do gerador aleatório exatamente os sorteios que os agentes já
    gravados consumiram (personalidade + pergunta 1 do IPA), na mesma ordem
    de gerar_agente(). Assim, retomar com a mesma --seed continua a sequência
    em vez de repetir os primeiros perfis (bug que duplicou sorteios no
    Experimento A)."""
    for _ in range(n_existentes):
        sortear_personalidade(modo)
        random.choice(["a", "b", "c", "d"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="Número de agentes a gerar")
    parser.add_argument("--output", required=True, help="Caminho do CSV de saída")
    parser.add_argument("--seed", type=int, default=None, help="Seed para reprodutibilidade do sorteio")
    parser.add_argument(
        "--modo", choices=["multi", "dominante"], default="multi",
        help=(
            "'multi' (padrão): os 5 traços variam Baixo/Médio/Alto de forma "
            "independente (Experimento A). 'dominante': só 1 traço (sorteado "
            "por agente) varia, os outros 4 ficam fixos em Médio (Experimento B)."
        ),
    )
    parser.add_argument("--backend", choices=["ollama", "openrouter"], default="ollama",
                        help="'ollama' (padrão, servidor local/Colab) ou 'openrouter' (API, precisa de OPENROUTER_API_KEY)")
    parser.add_argument("--modelo", default=None,
                        help="Nome do modelo (padrão: variável AGENTE_MODEL ou llama3.2:3b)")
    parser.add_argument("--temperatura", type=float, default=0.7,
                        help="Temperatura de geração (padrão 0.7, a usada no piloto e nos Experimentos A e B)")
    parser.add_argument("--intervalo", type=float, default=0.0,
                        help="Segundos de pausa depois de cada chamada ao modelo (evita limite por minuto da API)")
    parser.add_argument("--log", default=None, help="Arquivo de log (padrão: <output>.log)")
    parser.add_argument("--resumo-cada", type=int, default=10,
                        help="A cada quantos agentes anotar um resumo parcial no log")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help=f"Teto de tokens por resposta (padrão {MAX_TOKENS} no Ollama, {MAX_TOKENS_OPENROUTER} no OpenRouter)")
    args = parser.parse_args()
    CONFIG["max_tokens"] = args.max_tokens

    CONFIG["backend"] = args.backend
    CONFIG["temperatura"] = args.temperatura
    CONFIG["intervalo"] = args.intervalo
    if args.modelo:
        CONFIG["modelo"] = args.modelo
    if args.backend == "openrouter":
        CONFIG["api_key"] = os.environ.get("OPENROUTER_API_KEY") or ler_chave_env()
        if not CONFIG["api_key"]:
            sys.exit("ERRO: defina OPENROUTER_API_KEY no ambiente ou no arquivo .env do projeto.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    LOG["arquivo"] = args.log or os.path.splitext(args.output)[0] + ".log"

    campos = ["agente_id", "traco_dominante"]
    for dim in DIMENSOES:
        campos += [f"nivel_{dim}", f"nivel_{dim}_num"]
    for dim in DIMENSOES:
        campos.append(f"bfi_{dim}")
    campos += ["bfi_raw", "ipa_q1_atribuida", "ipa_respostas", "ipa_score", "ipa_perfil", "tempo_bfi44_s", "tempo_ipa_s",
               "modelo", "temperatura", "tentativas_bfi44", "tentativas_ipa"]

    # Arquivo vazio (processo caiu antes de gravar o cabeçalho) conta como novo
    novo_arquivo = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    agente_id_inicial = 0
    if not novo_arquivo:
        with open(args.output, encoding="utf-8") as f:
            cabecalho = next(csv.reader(f), [])
        if cabecalho and cabecalho != campos:
            sys.exit(
                f"ERRO: {args.output} foi gerado por uma versão anterior do script "
                f"(colunas diferentes). Use outro --output para não misturar formatos."
            )
        with open(args.output, encoding="utf-8") as f:
            linhas_existentes = list(csv.DictReader(f))
        agente_id_inicial = len(linhas_existentes)
        # Guarda contra retomar um arquivo com --modo diferente do usado
        # originalmente (ex.: retomar um dataset do Experimento B sem passar
        # --modo dominante geraria agentes multi-traço misturados no mesmo
        # arquivo, corrompendo o experimento).
        if linhas_existentes:
            modo_anterior = "dominante" if linhas_existentes[0].get("traco_dominante") else "multi"
            if modo_anterior != args.modo:
                sys.exit(
                    f"ERRO: {args.output} já contém agentes gerados com "
                    f"--modo {modo_anterior}, mas este comando usa "
                    f"--modo {args.modo}. Use o mesmo --modo do arquivo "
                    f"original para retomar, ou escolha outro --output."
                )

    if args.seed is not None:
        random.seed(args.seed)
        avancar_sorteio(agente_id_inicial, args.modo)

    registrar(
        f"INÍCIO {args.output} | backend={CONFIG['backend']} modelo={CONFIG['modelo']} "
        f"temperatura={CONFIG['temperatura']} modo={args.modo} seed={args.seed} "
        f"intervalo={CONFIG['intervalo']}s | já existentes={agente_id_inicial}, novos pedidos={args.n}"
    )

    gerados = 0
    inicio_total = time.time()
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if novo_arquivo:
            writer.writeheader()
            f.flush()

        for i in range(args.n):
            t0 = time.time()
            try:
                linha = gerar_agente(agente_id_inicial + i, modo=args.modo)
            except LimiteDiarioAtingido as e:
                registrar(f"PARADA: limite diário do OpenRouter atingido ({str(e)[:150]}). "
                          f"Agente {agente_id_inicial + i} descartado; rode o mesmo comando depois para retomar.")
                break
            except ProvedorIndisponivel as e:
                registrar(f"PARADA: provedor indisponível ({e}). "
                          f"Agente {agente_id_inicial + i} descartado; rode o mesmo comando depois para retomar.")
                break
            writer.writerow(linha)
            f.flush()
            gerados += 1
            dt = time.time() - t0
            dominante_str = f" dominante={linha['traco_dominante']}" if args.modo == "dominante" else ""
            erro = "" if linha["ipa_perfil"] and linha["bfi_raw"] and not linha["bfi_raw"].startswith("ERRO") else " [ERRO DE PARSING]"
            registrar(
                f"[{i + 1}/{args.n}] agente_id={linha['agente_id']}{dominante_str} "
                f"perfil_ipa={linha['ipa_perfil']!r} score={linha['ipa_score']!r} "
                f"tentativas bfi/ipa={linha['tentativas_bfi44']}/{linha['tentativas_ipa']} ({dt:.1f}s){erro}"
            )
            if args.resumo_cada and gerados % args.resumo_cada == 0:
                registrar("PARCIAL " + resumo_parcial(args.output))

    total = time.time() - inicio_total
    registrar(f"FIM: {gerados} agentes novos em {total:.1f}s"
              + (f" ({total / gerados:.1f}s/agente)" if gerados else ""))
    registrar("RESUMO " + resumo_parcial(args.output))
    print(f"Resultados salvos em: {args.output}")


def ler_chave_env():
    """Lê OPENROUTER_API_KEY de um arquivo .env na pasta do projeto (não versionado)."""
    caminho = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha.startswith("OPENROUTER_API_KEY="):
                return linha.split("=", 1)[1].strip().strip('"').strip("'")
    return None


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        registrar("ERRO INESPERADO, rodada interrompida: " + traceback.format_exc())
        raise
