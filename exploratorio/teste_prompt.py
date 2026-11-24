"""
Script de teste (piloto) para validar o formato de prompt/resposta
do llama3.2:1b via Ollama, antes de rodar o pipeline completo.

Gera UM agente com personalidade aleatória, pede para ele responder
o BFI-10 e o IPA, e imprime as respostas brutas + parsing.
"""

import json
import random
import re
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"

with open("questionarios/bfi10.json", encoding="utf-8") as f:
    bfi10 = json.load(f)

with open("questionarios/ipa.json", encoding="utf-8") as f:
    ipa = json.load(f)

with open("questionarios/perfis_personalidade.json", encoding="utf-8") as f:
    perfis = json.load(f)["dimensoes"]

NIVEIS = ["Baixo", "Médio", "Alto"]


def sortear_personalidade():
    return {dim: random.choice(NIVEIS) for dim in perfis}


def montar_system_prompt(personalidade):
    linhas = ["Você deve responder como uma pessoa com o seguinte perfil de personalidade:"]
    for dim, nivel in personalidade.items():
        linhas.append(f"- {dim} ({nivel}): {perfis[dim][nivel]}")
    linhas.append(
        "\nResponda TODAS as perguntas a seguir como essa pessoa responderia, "
        "de forma consistente com esse perfil. Não explique, não justifique, "
        "apenas dê as respostas no formato pedido."
    )
    return "\n".join(linhas)


def montar_prompt_bfi10():
    linhas = [
        bfi10["instrucoes"],
        "",
        "Escala: " + " | ".join(f"{k} = {v}" for k, v in bfi10["escala"].items()),
        "",
        "Afirmações:",
    ]
    for item in bfi10["itens"]:
        linhas.append(f"{item['id']}. {item['texto']}")
    linhas.append(
        "\nDê uma nota de 1 a 5 para cada afirmação, na ordem acima. "
        "Responda APENAS com os 10 números separados por vírgula, "
        "sem mais nenhum texto. Exemplo de formato: 3,4,2,5,1,4,3,2,5,4"
    )
    linhas.append("\nResposta:")
    return "\n".join(linhas)


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


def chamar_ollama(system_prompt, prompt):
    inicio = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
        },
    )
    dur = time.time() - inicio
    data = resp.json()
    return data["response"], dur


if __name__ == "__main__":
    personalidade = sortear_personalidade()
    print("Personalidade sorteada:")
    for dim, nivel in personalidade.items():
        print(f"  {dim}: {nivel}")
    print()

    system_prompt = montar_system_prompt(personalidade)
    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print()

    print("=== Enviando BFI-10 ===")
    resposta_bfi10, dur = chamar_ollama(system_prompt, montar_prompt_bfi10())
    print(f"(demorou {dur:.2f}s)")
    print(resposta_bfi10)
    print()

    print("=== Enviando IPA ===")
    resposta_ipa, dur = chamar_ollama(system_prompt, montar_prompt_ipa())
    print(f"(demorou {dur:.2f}s)")
    print(resposta_ipa)
