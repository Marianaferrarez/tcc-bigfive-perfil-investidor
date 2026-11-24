"""
Diagnóstico: testa se pedir um breve "raciocínio" (1-2 frases sobre como
a personalidade afeta a relação da pessoa com dinheiro/investimentos)
antes da resposta estruturada do IPA melhora a diferenciação entre
personalidades opostas (tudo Alto vs tudo Baixo).
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
    linhas = ["Você é uma pessoa com o seguinte perfil de personalidade:"]
    for dim in perfis:
        linhas.append(f"- {dim} ({nivel}): {perfis[dim][nivel]}")
    return "\n".join(linhas)


def montar_prompt_ipa_cot():
    linhas = [
        "Primeiro, em 1 ou 2 frases, descreva como o seu perfil de personalidade "
        "(descrito acima) influencia a sua relação com dinheiro, risco e investimentos.",
        "",
        "Depois, para cada pergunta abaixo, escolha a alternativa (a, b, c ou d) "
        "que melhor representa você, de forma CONSISTENTE com o que você escreveu.",
        "",
    ]
    for q in ipa["perguntas"]:
        linhas.append(f"{q['id']}. {q['texto']}")
        for letra, op in q["opcoes"].items():
            linhas.append(f"   {letra}) {op['texto']}")
        linhas.append("")
    linhas.append(
        "Termine sua resposta com uma linha no formato exato abaixo "
        "(substituindo pelas letras escolhidas, na ordem das 8 perguntas):"
    )
    linhas.append("RESPOSTAS: a,c,b,d,a,c,b,d")
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
    m = re.search(r"RESPOSTAS?\s*:?\s*([a-dA-D](?:\s*,\s*[a-dA-D]){7})", texto, re.IGNORECASE)
    if not m:
        return None
    return [c.lower() for c in re.findall(r"[a-dA-D]", m.group(1))]


def escore(letras):
    return sum(q["opcoes"][letra]["pontos"] for q, letra in zip(ipa["perguntas"], letras))


if __name__ == "__main__":
    prompt_ipa = montar_prompt_ipa_cot()

    print("=== Agente 'tudo ALTO' ===")
    resp_alto, dur = chamar_ollama(montar_system_prompt("Alto"), prompt_ipa)
    print(f"(demorou {dur:.2f}s)")
    print(resp_alto)
    letras_alto = extrair_letras(resp_alto)
    print("Letras:", letras_alto, "Escore:", escore(letras_alto) if letras_alto and len(letras_alto) == 8 else "incompleto")
    print()

    print("=== Agente 'tudo BAIXO' ===")
    resp_baixo, dur = chamar_ollama(montar_system_prompt("Baixo"), prompt_ipa)
    print(f"(demorou {dur:.2f}s)")
    print(resp_baixo)
    letras_baixo = extrair_letras(resp_baixo)
    print("Letras:", letras_baixo, "Escore:", escore(letras_baixo) if letras_baixo and len(letras_baixo) == 8 else "incompleto")
