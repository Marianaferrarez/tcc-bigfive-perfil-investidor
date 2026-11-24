"""
Comparação de modelos (pedido do orientador): roda o desenho do Experimento A
(5 traços simultâneos, BFI-44, IPA, --seed 123) em modelos gratuitos do
OpenRouter, um piloto de N agentes por modelo.

Cada modelo gera resultados/modelos/<apelido>.csv e <apelido>.log (andamento
e resumos parciais). Pode ser rodado de novo a qualquer momento: continua de
onde parou em cada modelo, sem repetir perfis. Se o limite diário de
chamadas gratuitas for atingido, para e avisa; basta rodar de novo depois.

Uso:
    python rodar_modelos_openrouter.py            # N=30 por modelo, 8s entre chamadas
    python rodar_modelos_openrouter.py --n 30 --intervalo 10
"""

import argparse
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Raiz do repositório (este script fica em src/<pasta>/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gemma por último: o provedor dele vive congestionado e as esperas gastam a cota diária.
MODELOS = {
    "nemotron35_super": "nvidia/nemotron-3.5-super-vl:free",
    "nemotron3_ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "gemma4_31b": "google/gemma-4-31b-it:free",
}


def ja_gerados(caminho):
    if not os.path.exists(caminho):
        return 0
    with open(caminho, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="Agentes por modelo (padrão 30, piloto)")
    p.add_argument("--intervalo", type=float, default=8.0,
                   help="Segundos entre chamadas (padrão 8: fica abaixo de 8 chamadas/min)")
    p.add_argument("--temperatura", type=float, default=0.7, help="Padrão 0.7, igual ao Experimento A")
    p.add_argument("--so", choices=list(MODELOS), help="Rodar só um dos modelos")
    args = p.parse_args()

    os.makedirs(os.path.join(BASE_DIR, "resultados", "modelos"), exist_ok=True)
    alvo = {a: m for a, m in MODELOS.items() if not args.so or a == args.so}

    # Até 3 passadas: um modelo cujo provedor estiver congestionado é pulado
    # e tentado de novo na passada seguinte, depois dos outros.
    for passada in range(1, 4):
        pendentes = {a: m for a, m in alvo.items()
                     if ja_gerados(os.path.join(BASE_DIR, "resultados", "modelos", f"{a}.csv")) < args.n}
        if not pendentes:
            break
        print(f"\n######## Passada {passada}: {', '.join(pendentes)}")
        for apelido, modelo in pendentes.items():
            saida = os.path.join("resultados", "modelos", f"{apelido}.csv")
            faltam = args.n - ja_gerados(os.path.join(BASE_DIR, saida))
            print(f"\n===== {modelo}: faltam {faltam}")
            cmd = [sys.executable, os.path.join(BASE_DIR, "src", "geracao", "gerar_agentes.py"), "--n", str(faltam), "--output", saida,
                   "--modo", "multi", "--seed", "123", "--backend", "openrouter", "--modelo", modelo,
                   "--temperatura", str(args.temperatura), "--intervalo", str(args.intervalo), "--resumo-cada", "5"]
            r = subprocess.run(cmd, cwd=BASE_DIR)
            if r.returncode != 0:
                print(f"AVISO: gerar_agentes.py terminou com erro em {modelo}; veja {saida[:-4]}.log")
                continue
            log = os.path.join(BASE_DIR, saida[:-4] + ".log")
            with open(log, encoding="utf-8") as f:
                ultimas = f.read()[-2000:]
            if "limite diário" in ultimas and ja_gerados(os.path.join(BASE_DIR, saida)) < args.n:
                sys.exit("Limite diário de chamadas gratuitas atingido. Rode este script de novo amanhã; ele continua de onde parou.")

    for apelido in alvo:
        print(f"{apelido}: {ja_gerados(os.path.join(BASE_DIR, 'resultados', 'modelos', f'{apelido}.csv'))}/{args.n}")


if __name__ == "__main__":
    main()
