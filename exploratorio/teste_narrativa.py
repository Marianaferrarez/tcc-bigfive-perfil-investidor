"""
Diagnóstico: indução de personalidade via narrativa em primeira pessoa.

Etapa 1: o agente escreve um parágrafo curto sobre sua vida/rotina,
consistente com a personalidade induzida (descrição dos traços).

Etapa 2: usando a descrição dos traços + a narrativa gerada como contexto,
o agente responde o IPA.

Compara 'tudo Alto' vs 'tudo Baixo' para ver se a narrativa melhora a
diferenciação das respostas.
"""

import json
import re
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"

with open("questionarios/ipa.json", encoding="utf-8") as f:
    ipa = json.load(f)

with open("questionarios/perfis_personalidade.json", encoding="utf-8") as f:
    perfis = json.load(f)["dimensoes"]


def descricao_personalidade(nivel):
    linhas = []
    for dim in perfis:
        linhas.append(f"- {dim} ({nivel}): {perfis[dim][nivel]}")
    return "\n".join(linhas)


def montar_prompt_narrativa():
    return (
        "Escreva um parágrafo curto (3 a 5 frases), em primeira pessoa, "
        "contando um pouco sobre sua vida, sua rotina e como você costuma "
        "lidar com dinheiro e decisões no dia a dia. Seja consistente com o "
        "seu perfil de personalidade. Não repita a descrição do perfil, "
        "apenas escreva a narrativa."
    )


def montar_prompt_ipa():
    linhas = ["Para cada pergunta abaixo, escolha a alternativa (a, b, c ou d) que melhor representa você.", ""]
    for q in ipa["perguntas"]:
        linhas.append(f"{q['id']}. {q['texto']}")
        for letra, op in q["opcoes"].items():
            linhas.append(f"   {letra}) {op['texto']}")
        linhas.append("")
    linhas.append(
        "Dê uma resposta (a, b, c ou d) para cada pergunta, na ordem acima. "
        "Responda APENAS com as 8 letras separadas por vírgula, "
        "sem mais nenhum texto. Exemplo de formato: a,c,b,d,a,c,b,d"
    )
    linhas.append("\nResposta:")
    return "\n".join(linhas)


def chamar_ollama(system_prompt, prompt, temperature=0.7):
    inicio = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    dur = time.time() - inicio
    return resp.json()["response"].strip(), dur


def extrair_letras(texto):
    letras = re.findall(r"[a-dA-D]", texto)
    return [c.lower() for c in letras][:8]


def escore(letras):
    return sum(q["opcoes"][letra]["pontos"] for q, letra in zip(ipa["perguntas"], letras))


def rodar_agente(nivel):
    descricao = descricao_personalidade(nivel)
    system_base = "Você é uma pessoa com o seguinte perfil de personalidade:\n" + descricao

    narrativa, t1 = chamar_ollama(system_base, montar_prompt_narrativa())

    system_com_narrativa = (
        system_base
        + "\n\nAqui está um pouco sobre sua vida, na sua própria voz:\n"
        + narrativa
        + "\n\nResponda as próximas perguntas como essa pessoa responderia, "
        "de forma consistente com o perfil e a narrativa acima. Não explique, "
        "apenas dê as respostas no formato pedido."
    )

    resp_ipa, t2 = chamar_ollama(system_com_narrativa, montar_prompt_ipa(), temperature=0.7)
    letras = extrair_letras(resp_ipa)
    score = escore(letras) if len(letras) == 8 else None

    return narrativa, t1, resp_ipa, letras, score, t2


if __name__ == "__main__":
    for nivel in ["Alto", "Baixo"]:
        print(f"=== Agente 'tudo {nivel.upper()}' ===")
        narrativa, t1, resp_ipa, letras, score, t2 = rodar_agente(nivel)
        print(f"Narrativa ({t1:.1f}s): {narrativa}")
        print(f"IPA ({t2:.1f}s): {resp_ipa}")
        print(f"Letras: {letras} | Score: {score}")
        print(f"Tempo total: {t1 + t2:.1f}s")
        print()
