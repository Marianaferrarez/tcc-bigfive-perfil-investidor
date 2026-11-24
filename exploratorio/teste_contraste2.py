"""
Segunda tentativa de prompt para o BFI-10: formato "item -> nota",
repetindo o texto do item na linha de resposta (em vez de uma lista
de números desconectada no final), para tentar melhorar a ligação
entre a personalidade induzida e a nota dada a cada item.
"""

import json
import re
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"

with open("questionarios/bfi10.json", encoding="utf-8") as f:
    bfi10 = json.load(f)

with open("questionarios/perfis_personalidade.json", encoding="utf-8") as f:
    perfis = json.load(f)["dimensoes"]


def montar_system_prompt(nivel):
    linhas = ["Você deve responder como uma pessoa com o seguinte perfil de personalidade:"]
    for dim in perfis:
        linhas.append(f"- {dim} ({nivel}): {perfis[dim][nivel]}")
    linhas.append(
        "\nResponda TODAS as perguntas a seguir como essa pessoa responderia, "
        "de forma consistente com esse perfil."
    )
    return "\n".join(linhas)


def montar_prompt_bfi10():
    linhas = [
        bfi10["instrucoes"],
        "",
        "Escala: " + " | ".join(f"{k} = {v}" for k, v in bfi10["escala"].items()),
        "",
        "Para cada afirmação abaixo, complete com a nota de 1 a 5 que essa pessoa daria.",
        "Mantenha exatamente o formato 'N. texto -> nota'. Não escreva mais nada.",
        "",
    ]
    for item in bfi10["itens"]:
        linhas.append(f"{item['id']}. {item['texto']} ->")
    return "\n".join(linhas)


def chamar_ollama(system_prompt, prompt, temperature=0.1):
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
    )
    dur = time.time() - inicio
    data = resp.json()
    return data["response"].strip(), dur


def extrair_notas(texto):
    # procura padrões "N. texto -> nota" ou "N -> nota" etc.
    notas = {}
    for linha in texto.splitlines():
        m = re.search(r"^\s*(\d+)\..*?(?:->|→|:)\s*(\d)", linha)
        if m:
            notas[int(m.group(1))] = int(m.group(2))
    return notas


if __name__ == "__main__":
    prompt_bfi10 = montar_prompt_bfi10()
    print("=== PROMPT ===")
    print(prompt_bfi10)
    print()

    print("=== Agente 'tudo ALTO' ===")
    sp_alto = montar_system_prompt("Alto")
    resp_alto, dur = chamar_ollama(sp_alto, prompt_bfi10)
    print(f"(demorou {dur:.2f}s)")
    print(resp_alto)
    print()

    print("=== Agente 'tudo BAIXO' ===")
    sp_baixo = montar_system_prompt("Baixo")
    resp_baixo, dur = chamar_ollama(sp_baixo, prompt_bfi10)
    print(f"(demorou {dur:.2f}s)")
    print(resp_baixo)
    print()

    notas_alto = extrair_notas(resp_alto)
    notas_baixo = extrair_notas(resp_baixo)
    print("=== Comparação item a item ===")
    print(f"{'Item':<5}{'Dimensão':<25}{'Reverso':<8}{'Alto':<6}{'Baixo':<6}")
    for item in bfi10["itens"]:
        i = item["id"]
        print(f"{i:<5}{item['dimensao']:<25}{str(item['reverso']):<8}"
              f"{notas_alto.get(i, '?'):<6}{notas_baixo.get(i, '?'):<6}")
