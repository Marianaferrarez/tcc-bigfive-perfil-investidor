"""
Correlação entre os traços autorrelatados (motivado por Petrov et al., 2024,
que relatam correlações entre traços acima das humanas em LLMs).

Como os níveis induzidos são independentes por construção (sorteio), toda
correlação a mais no autorrelato vem do modelo. Para o Experimento A (BFI-44)
e o piloto (BFI-10), calcula:
  1. Spearman entre os escores autorrelatados (10 pares) vs. entre os níveis induzidos;
  2. "vazamento": Spearman nível induzido de um traço x autorrelato de outro;
  3. correlação entre resíduos (autorrelato controlado pelo próprio nível induzido);
  4. ganho de R² do autorrelato ao incluir os outros 4 níveis induzidos.

Saída: resultados/intercorrelacoes.json

Uso:
    python analise_intercorrelacoes.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

# Raiz do repositório (este script fica em src/<pasta>/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE_DIR, "resultados")
D = ["Extroversão", "Afabilidade", "Consciência", "Neuroticismo", "Abertura à Experiência"]
DATASETS = {"exp_a": "principal/exp_a_bfi44_multitraco_8b.csv", "piloto": "piloto/dataset_llama31_8b.csv"}


def pares():
    return [(a, b) for i, a in enumerate(D) for b in D[i + 1:]]


def analisar(df):
    bfi = df[[f"bfi_{d}" for d in D]].apply(pd.to_numeric, errors="coerce")
    bfi.columns = D
    niv = df[[f"nivel_{d}_num" for d in D]].astype(int)
    niv.columns = D
    ok = bfi.notna().all(axis=1)
    bfi, niv = bfi[ok], niv[ok]
    out = {"n": int(len(bfi)), "pares": [], "vazamento": {}, "residuos": [], "r2": {}}
    absr = []
    for a, b in pares():
        r, p = stats.spearmanr(bfi[a], bfi[b])
        r_niv = stats.spearmanr(niv[a], niv[b])[0]
        absr.append(abs(float(r)))
        out["pares"].append({"a": a, "b": b, "rho_autorrelato": round(float(r), 3), "p": round(float(p), 4),
                             "rho_induzido": round(float(r_niv), 3)})
    out["rho_medio_abs"] = round(float(np.mean(absr)), 3)
    out["rho_max_abs"] = round(max(absr), 3)
    out["pares_significativos"] = int(sum(x["p"] < 0.05 for x in out["pares"]))
    for x in D:
        out["vazamento"][x] = {}
        for y in D:
            r, p = stats.spearmanr(niv[x], bfi[y])
            out["vazamento"][x][y] = {"rho": round(float(r), 3), "p": round(float(p), 4)}
    res = pd.DataFrame({d: sm.OLS(bfi[d], sm.add_constant(niv[d])).fit().resid for d in D})
    for a, b in pares():
        r, p = stats.pearsonr(res[a], res[b])
        out["residuos"].append({"a": a, "b": b, "r": round(float(r), 3), "p": round(float(p), 4)})
    for d in D:
        proprio = sm.OLS(bfi[d], sm.add_constant(niv[d])).fit().rsquared
        todos = sm.OLS(bfi[d], sm.add_constant(niv)).fit().rsquared
        out["r2"][d] = {"proprio": round(float(proprio), 3), "todos": round(float(todos), 3),
                        "ganho": round(float(todos - proprio), 3)}
    return out


def main():
    R = {k: analisar(pd.read_csv(os.path.join(RES, arq))) for k, arq in DATASETS.items()}
    caminho = os.path.join(RES, "intercorrelacoes.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    for k, r in R.items():
        print(f"{k}: N={r['n']} |rho| médio={r['rho_medio_abs']} máx={r['rho_max_abs']} "
              f"significativos={r['pares_significativos']}/10")
    print("Salvo em:", caminho)


if __name__ == "__main__":
    main()
