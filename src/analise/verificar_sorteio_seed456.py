"""Verifica o sorteio dos perfis da rodada com outra semente (seed 456).

Mesmos testes da verificação principal (analise_aleatorizacao.py), em versão
resumida: equilíbrio dos níveis por traço (qui-quadrado contra a uniforme) e
independência entre os traços (Spearman entre os níveis sorteados).
Saída: resultados/robustez/verificacao_sorteio_seed456.json
"""
import itertools
import json
import os

import pandas as pd
from scipy import stats

# Raiz do repositório (este script fica em src/<pasta>/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(BASE_DIR, "resultados", "robustez", "exp_a_seed456_bfi44_8b.csv")
SAIDA = os.path.join(BASE_DIR, "resultados", "robustez", "verificacao_sorteio_seed456.json")
TRACOS = ["Extroversão", "Afabilidade", "Consciência", "Neuroticismo", "Abertura à Experiência"]


def main():
    df = pd.read_csv(CSV)
    niveis = {t: df[f"nivel_{t}_num"] for t in TRACOS}

    equilibrio = {}
    for t, s in niveis.items():
        contagens = [int((s == n).sum()) for n in (1, 2, 3)]
        chi2, p = stats.chisquare(contagens)
        equilibrio[t] = {"contagens": contagens, "chi2": round(float(chi2), 3), "p": round(float(p), 4)}

    independencia = []
    for a, b in itertools.combinations(TRACOS, 2):
        rho, p = stats.spearmanr(niveis[a], niveis[b])
        independencia.append({"a": a, "b": b, "rho": round(float(rho), 3), "p": round(float(p), 4)})

    todas = [c for e in equilibrio.values() for c in e["contagens"]]
    resumo = {
        "n": len(df),
        "contagem_min": min(todas),
        "contagem_max": max(todas),
        "p_qui2_min": min(e["p"] for e in equilibrio.values()),
        "rho_abs_max": max(abs(x["rho"]) for x in independencia),
    }
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"resumo": resumo, "equilibrio": equilibrio, "independencia": independencia},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False))


if __name__ == "__main__":
    main()
