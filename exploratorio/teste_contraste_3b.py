"""
Teste de validação: será que o llama3.2:3b realmente muda as respostas
do BFI-10 de acordo com a personalidade induzida no system prompt?

Compara um agente "tudo Alto" com um agente "tudo Baixo" nos 5 traços.
Se a indução estiver funcionando, esperamos respostas em direções
diferentes nos itens do BFI-10 (ex.: item 6 "é sociável, extrovertido"
deve ter nota alta no agente Alto-Extroversão e baixa no agente
Baixo-Extroversão).
"""

import json
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

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


def chamar_ollama(system_prompt, prompt):
    inicio = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7},
        },
    )
    dur = time.time() - inicio
    data = resp.json()
    return data["response"].strip(), dur


if __name__ == "__main__":
    prompt_bfi10 = montar_prompt_bfi10()

    print("=== Agente 'tudo ALTO' ===")
    sp_alto = montar_system_prompt("Alto")
    resp_alto, dur = chamar_ollama(sp_alto, prompt_bfi10)
    print(f"(demorou {dur:.2f}s)")
    print("Resposta:", resp_alto)
    print()

    print("=== Agente 'tudo BAIXO' ===")
    sp_baixo = montar_system_prompt("Baixo")
    resp_baixo, dur = chamar_ollama(sp_baixo, prompt_bfi10)
    print(f"(demorou {dur:.2f}s)")
    print("Resposta:", resp_baixo)
    print()

    # Comparação item a item
    try:
        notas_alto = [int(x) for x in resp_alto.replace(" ", "").split(",")]
        notas_baixo = [int(x) for x in resp_baixo.replace(" ", "").split(",")]
        print("=== Comparação item a item ===")
        print(f"{'Item':<5}{'Dimensão':<25}{'Reverso':<8}{'Alto':<6}{'Baixo':<6}")
        for item, a, b in zip(bfi10["itens"], notas_alto, notas_baixo):
            print(f"{item['id']:<5}{item['dimensao']:<25}{str(item['reverso']):<8}{a:<6}{b:<6}")
    except Exception as e:
        print("Não foi possível comparar (formato inesperado):", e)
