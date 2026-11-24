"""
Análise dos dois estudos pedidos pelo orientador, sobre o desenho do
Experimento A (5 traços simultâneos, BFI-44, IPA, --seed 123):

  1. Temperatura (llama3.1:8b, N=300): 0,0 e 0,3 contra 0,7 (Experimento A).
     resultados/temperatura/exp_a_t00_bfi44_8b.csv, exp_a_t03_bfi44_8b.csv
     Temperatura 1,0 (testes extras): exp_a_t10_bfi44_8b.csv
  2. Outros modelos (OpenRouter, piloto N=30): resultados/modelos/*.csv,
     comparados com os mesmos 30 primeiros perfis do Experimento A.
  3. Testes extras no Colab (colab_testes_extras.ipynb):
     - outra semente (--seed 456, 0,7): resultados/robustez/exp_a_seed456_bfi44_8b.csv
       (perfis diferentes: compara só os resumos, sem pareamento);
     - modelos abertos locais (N=300, --seed 123): resultados/modelos_locais/*.csv,
       pareados com o Experimento A.

Como todas as rodadas usam a mesma seed, os perfis são idênticos agente a
agente, o que permite comparações pareadas (mesmo perfil, condição diferente).

Analisa o que já existir (rodadas incompletas entram com o N que tiverem) e
grava resultados/comparacao_temperatura_modelos.json e .md.

Uso:
    python analise_temperatura_modelos.py
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
DIMENSOES = ["Extroversão", "Afabilidade", "Consciência", "Neuroticismo", "Abertura à Experiência"]
NOME = {"Afabilidade": "Amabilidade"}
PERFIS = ["Conservador", "Moderado", "Arrojado"]

EXP_A = os.path.join(RES, "principal", "exp_a_bfi44_multitraco_8b.csv")
TEMPERATURAS = {
    "0,0": os.path.join(RES, "temperatura", "exp_a_t00_bfi44_8b.csv"),
    "0,3": os.path.join(RES, "temperatura", "exp_a_t03_bfi44_8b.csv"),
    "0,7": EXP_A,
    "1,0": os.path.join(RES, "temperatura", "exp_a_t10_bfi44_8b.csv"),
}
SEMENTES = {
    "seed 123 (Exp. A)": EXP_A,
    "seed 456": os.path.join(RES, "robustez", "exp_a_seed456_bfi44_8b.csv"),
}
MODELOS_LOCAIS = {
    "llama3.1:8b (Exp. A)": EXP_A,
    "qwen2.5:7b": os.path.join(RES, "modelos_locais", "exp_a_qwen25_7b.csv"),
    "mistral:7b": os.path.join(RES, "modelos_locais", "exp_a_mistral_7b.csv"),
    "gemma3:12b": os.path.join(RES, "modelos_locais", "exp_a_gemma3_12b.csv"),
}
MODELOS = {
    "llama3.1:8b (Exp. A, 30 primeiros)": None,  # recorte do Experimento A
    "gemma-4-31b": os.path.join(RES, "modelos", "gemma4_31b.csv"),
    "nemotron-3.5-super": os.path.join(RES, "modelos", "nemotron35_super.csv"),
    "nemotron-3-ultra": os.path.join(RES, "modelos", "nemotron3_ultra.csv"),
}


def existe_com_dados(caminho):
    if not caminho or not os.path.exists(caminho) or os.path.getsize(caminho) == 0:
        return False
    with open(caminho, encoding="utf-8") as f:
        return sum(1 for _ in f) > 1


def carregar(caminho):
    df = pd.read_csv(caminho)
    # Experimento A guarda agente_id a partir de 252 (correção da duplicação);
    # a posição na sequência da seed é a ordem das linhas.
    df = df.reset_index(drop=True)
    df["pos"] = df.index
    df["bfi_ok"] = df[f"bfi_{DIMENSOES[0]}"].notna()
    df["ipa_ok"] = df["ipa_perfil"].isin(PERFIS)
    q1 = df["ipa_q1_atribuida"].map({"a": 4, "b": 3, "c": 2, "d": 1})
    df["ipa_score_sem_q1"] = pd.to_numeric(df["ipa_score"], errors="coerce") - q1
    return df


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def spearman(x, y):
    if len(x) < 5 or x.nunique() < 2 or y.nunique() < 2:
        return None, None
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def resumo(df, com_ols=True):
    n = len(df)
    ok = df[df["bfi_ok"] & df["ipa_ok"]]
    out = {
        "n": n,
        "n_validos": int(len(ok)),
        "pct_bfi_valido": round(100 * df["bfi_ok"].mean(), 1) if n else None,
        "pct_ipa_valido": round(100 * df["ipa_ok"].mean(), 1) if n else None,
    }
    if "tentativas_bfi44" in df:
        out["tentativas_media_bfi44"] = round(float(pd.to_numeric(df["tentativas_bfi44"]).mean()), 2)
        out["tentativas_media_ipa"] = round(float(pd.to_numeric(df["tentativas_ipa"]).mean()), 2)
    if len(ok) == 0:
        return out
    cont = ok["ipa_perfil"].value_counts().reindex(PERFIS, fill_value=0)
    out["perfis"] = {k: int(v) for k, v in cont.items()}
    out["perfis_pct"] = {k: round(100 * v / len(ok), 1) for k, v in cont.items()}
    out["ipa_sem_q1_media"] = round(float(ok["ipa_score_sem_q1"].mean()), 2)
    out["ipa_sem_q1_dp"] = round(float(ok["ipa_score_sem_q1"].std()), 2)
    out["manipulacao"], out["spearman_ipa"] = {}, {}
    for d in DIMENSOES:
        r, p = spearman(ok[f"nivel_{d}_num"], ok[f"bfi_{d}"])
        out["manipulacao"][d] = {"rho": None if r is None else round(r, 3), "p": None if p is None else round(p, 4)}
        r, p = spearman(ok[f"nivel_{d}_num"], ok["ipa_score_sem_q1"])
        out["spearman_ipa"][d] = {"rho": None if r is None else round(r, 3), "p": None if p is None else round(p, 4)}
    if com_ols and len(ok) >= 20 and ok["ipa_score_sem_q1"].nunique() > 1:
        X = sm.add_constant(ok[[f"nivel_{d}_num" for d in DIMENSOES]].astype(float))
        m = sm.OLS(ok["ipa_score_sem_q1"].astype(float), X).fit()
        out["ols"] = {
            "r2": round(float(m.rsquared), 3), "f": round(float(m.fvalue), 2), "p": round(float(m.f_pvalue), 4),
            "coef": {d: {"b": round(float(m.params[f"nivel_{d}_num"]), 3), "p": round(float(m.pvalues[f"nivel_{d}_num"]), 4)}
                     for d in DIMENSOES},
        }
    return out


def pareado(df, base):
    """Compara a mesma posição da seed (mesmo perfil) entre duas condições."""
    m = df.merge(base, on="pos", suffixes=("", "_base"))
    m = m[m["ipa_ok"] & m["ipa_ok_base"]]
    if len(m) < 5:
        return None
    perfis_iguais = [all(a[f"nivel_{d}"] == a[f"nivel_{d}_base"] for d in DIMENSOES) for _, a in m.iterrows()]
    resp = m["ipa_respostas"].str.split(",")
    resp_b = m["ipa_respostas_base"].str.split(",")
    concord_q = np.mean([np.mean([x == y for x, y in zip(a[1:], b[1:])]) for a, b in zip(resp, resp_b)])  # Q2-Q8
    r, p = stats.spearmanr(m["ipa_score_sem_q1"], m["ipa_score_sem_q1_base"])
    return {
        "n_pares": int(len(m)),
        "perfis_identicos": bool(all(perfis_iguais)),
        "spearman_escore_ipa": round(float(r), 3),
        "p": round(float(p), 4),
        "concordancia_respostas_q2_q8_pct": round(100 * float(concord_q), 1),
        "mesmo_perfil_ipa_pct": round(100 * float((m["ipa_perfil"] == m["ipa_perfil_base"]).mean()), 1),
    }


def variabilidade_mesmo_perfil(df):
    """Agentes com a mesma combinação de níveis: o IPA varia entre eles?
    (Com temperatura 0, espera-se variação quase nula.)"""
    ok = df[df["ipa_ok"]].copy()
    ok["combo"] = ok[[f"nivel_{d}_num" for d in DIMENSOES]].astype(str).agg("".join, axis=1)
    grupos = ok.groupby("combo")["ipa_score_sem_q1"]
    rep = grupos.filter(lambda s: len(s) >= 2)
    if rep.empty:
        return None
    dp = ok[ok["combo"].isin(grupos.size()[grupos.size() >= 2].index)].groupby("combo")["ipa_score_sem_q1"].std()
    return {"combinacoes_repetidas": int(len(dp)), "dp_medio_intra_combinacao": round(float(dp.mean()), 3),
            "pct_combinacoes_com_ipa_identico": round(100 * float((dp == 0).mean()), 1)}


def main():
    R = {"temperatura": {}, "modelos": {}, "sementes": {}, "modelos_locais": {}}
    base = carregar(EXP_A)

    for rot, cam in TEMPERATURAS.items():
        if not existe_com_dados(cam):
            R["temperatura"][rot] = {"status": "ainda não gerado"}
            continue
        df = carregar(cam)
        r = resumo(df)
        r["variabilidade_mesmo_perfil"] = variabilidade_mesmo_perfil(df)
        if rot != "0,7":
            r["pareado_com_0,7"] = pareado(df, base)
        R["temperatura"][rot] = r

    for rot, cam in MODELOS.items():
        if cam is None:
            n_ref = max([len(carregar(c)) for c in MODELOS.values() if existe_com_dados(c)] or [30])
            df = base.iloc[:n_ref].copy()
        elif not existe_com_dados(cam):
            R["modelos"][rot] = {"status": "ainda não gerado"}
            continue
        else:
            df = carregar(cam)
        r = resumo(df, com_ols=True)
        if cam is not None:
            r["pareado_com_llama"] = pareado(df, base)
            if "tempo_ipa_s" in df:
                t = pd.to_numeric(df["tempo_bfi44_s"], errors="coerce") + pd.to_numeric(df["tempo_ipa_s"], errors="coerce")
                r["tempo_medio_resposta_s"] = round(float(t.mean()), 1)
        R["modelos"][rot] = r

    for chave, grupo, parear in (("sementes", SEMENTES, False), ("modelos_locais", MODELOS_LOCAIS, True)):
        for rot, cam in grupo.items():
            if not existe_com_dados(cam):
                R[chave][rot] = {"status": "ainda não gerado"}
                continue
            df = carregar(cam)
            r = resumo(df)
            if parear and cam != EXP_A:
                r["pareado_com_llama"] = pareado(df, base)
            R[chave][rot] = r

    with open(os.path.join(RES, "comparacao_temperatura_modelos.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    md = gerar_md(R)
    with open(os.path.join(RES, "comparacao_temperatura_modelos.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def fmt_rho(x):
    if not x or x.get("rho") is None:
        return "n/d"
    return f"{x['rho']:+.3f} {sig(x['p'])}"


def tabela_estudo(bloco, titulo):
    linhas = [f"## {titulo}\n"]
    nomes = list(bloco)
    linhas.append("| Indicador | " + " | ".join(nomes) + " |")
    linhas.append("|---|" + "---|" * len(nomes))

    def lin(rot, f):
        vals = []
        for k in nomes:
            b = bloco[k]
            try:
                vals.append(f(b) if "status" not in b else b["status"])
            except Exception:
                vals.append("n/d")
        linhas.append(f"| {rot} | " + " | ".join(vals) + " |")

    lin("N (válidos)", lambda b: f"{b['n']} ({b['n_validos']})")
    lin("Respostas válidas BFI / IPA (%)", lambda b: f"{b['pct_bfi_valido']} / {b['pct_ipa_valido']}")
    lin("Tentativas médias BFI / IPA", lambda b: f"{b.get('tentativas_media_bfi44', 'n/d')} / {b.get('tentativas_media_ipa', 'n/d')}")
    lin("Perfis C / M / A (%)", lambda b: " / ".join(str(b["perfis_pct"][p]) for p in PERFIS))
    lin("IPA Q2-Q8 média (dp)", lambda b: f"{b['ipa_sem_q1_media']} ({b['ipa_sem_q1_dp']})")
    for d in DIMENSOES:
        lin(f"Manipulação ρ — {NOME.get(d, d)}", lambda b, d=d: fmt_rho(b["manipulacao"][d]))
    for d in DIMENSOES:
        lin(f"ρ nível × IPA — {NOME.get(d, d)}", lambda b, d=d: fmt_rho(b["spearman_ipa"][d]))
    lin("OLS R²", lambda b: f"{b['ols']['r2']}")
    for d in DIMENSOES:
        lin(f"OLS coef — {NOME.get(d, d)}", lambda b, d=d: f"{b['ols']['coef'][d]['b']:+.3f} {sig(b['ols']['coef'][d]['p'])}")
    return "\n".join(linhas) + "\n"


def gerar_md(R):
    partes = ["# Temperatura e outros modelos — resultados\n",
              "Gerado por `analise_temperatura_modelos.py`. Significância: * p<0,05; ** p<0,01; *** p<0,001; n.s. p≥0,05.\n"]
    partes.append(tabela_estudo(R["temperatura"], "Temperatura (llama3.1:8b, N=300)"))
    tabela_pareada(partes, R["temperatura"], "pareado_com_0,7", "Temperatura", "a temperatura 0,7")
    partes.append(tabela_estudo(R["modelos"], "Outros modelos (piloto, mesmos perfis dos primeiros agentes do Experimento A)"))
    tabela_pareada(partes, R["modelos"], "pareado_com_llama", "Modelo", "o llama3.1:8b")
    partes.append(tabela_estudo(R["sementes"], "Outra semente (llama3.1:8b, 0,7, N=300, perfis diferentes)"))
    partes.append(tabela_estudo(R["modelos_locais"], "Modelos abertos locais (N=300, mesmos perfis do Experimento A)"))
    tabela_pareada(partes, R["modelos_locais"], "pareado_com_llama", "Modelo", "o llama3.1:8b")
    return "\n".join(partes)


def tabela_pareada(partes, bloco, chave, coluna, contra):
    pares = {k: v.get(chave) for k, v in bloco.items() if isinstance(v, dict) and v.get(chave)}
    if not pares:
        return
    partes.append(f"### Comparação pareada com {contra} (mesmos perfis)\n")
    partes.append(f"| {coluna} | Pares | ρ escore IPA | Mesmo perfil IPA (%) | Respostas Q2-Q8 iguais (%) |\n|---|---|---|---|---|")
    for k, v in pares.items():
        partes.append(f"| {k} | {v['n_pares']} | {v['spearman_escore_ipa']:+.3f} {sig(v['p'])} | {v['mesmo_perfil_ipa_pct']} | {v['concordancia_respostas_q2_q8_pct']} |")
    partes.append("")


if __name__ == "__main__":
    main()
