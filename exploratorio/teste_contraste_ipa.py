"""
Testa se as respostas ao questionário IPA (variável dependente real do
estudo) variam de forma sensata entre um agente "tudo Alto" e um agente
"tudo Baixo" nos 5 traços do Big Five.
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


def extrair_letras(texto):
    letras = re.findall(r"[a-dA-D]", texto)
    return [c.lower() for c in letras][:8]


def escore(letras):
    total = 0
    for q, letra in zip(ipa["perguntas"], letras):
        total += q["opcoes"][letra]["pontos"]
    return total


if __name__ == "__main__":
    prompt_ipa = montar_prompt_ipa()

    print("=== Agente 'tudo ALTO' (Aberto, Extrovertido, pouco Neurótico... espera-se mais arrojado) ===")
    sp_alto = montar_system_prompt("Alto")
    resp_alto, dur = chamar_ollama(sp_alto, prompt_ipa)
    print(f"(demorou {dur:.2f}s) -> {resp_alto}")
    letras_alto = extrair_letras(resp_alto)
    print("Letras:", letras_alto, "Escore:", escore(letras_alto) if len(letras_alto) == 8 else "incompleto")
    print()

    print("=== Agente 'tudo BAIXO' (Fechado, Reservado, Neurótico... espera-se mais conservador) ===")
    sp_baixo = montar_system_prompt("Baixo")
    resp_baixo, dur = chamar_ollama(sp_baixo, prompt_ipa)
    print(f"(demorou {dur:.2f}s) -> {resp_baixo}")
    letras_baixo = extrair_letras(resp_baixo)
    print("Letras:", letras_baixo, "Escore:", escore(letras_baixo) if len(letras_baixo) == 8 else "incompleto")
