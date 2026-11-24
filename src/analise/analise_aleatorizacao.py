"""
Verificação da qualidade do sorteio aleatório das personalidades induzidas.

Pergunta (orientador): o sorteio dos níveis Baixo/Médio/Alto ficou bem
espalhado entre os agentes, ou concentrou em alguns níveis/combinações a
ponto de distorcer os resultados?

Testes, por dataset:
  1. Frequência de cada nível por traço (qui-quadrado de aderência à uniforme).
  2. Independência entre traços (Spearman e qui-quadrado/Cramér's V por par).
  3. Multicolinearidade dos preditores da OLS (VIF).
  4. Cobertura das 3^5 = 243 combinações possíveis vs. o esperado por acaso
     (simulação Monte Carlo do mesmo sorteio).
  5. Estabilidade ao longo da geração (1ª metade vs. 2ª metade dos agentes).
  6. Pergunta 1 do IPA (faixa etária sorteada): uniformidade e independência
     em relação aos traços.
  7. Reprodutibilidade: o CSV bate exatamente com random.seed(123)?
  8. Impacto nos resultados: coeficiente de cada traço na regressão simples
     vs. na OLS com os 5 traços (se o sorteio estivesse desbalanceado ou
     correlacionado, os coeficientes mudariam entre os dois modelos).

Saídas (todas regeneradas a cada execução, a partir dos dados):
  resultados/aleatorizacao.json                      todos os números
  resultados/tabelas/aleatorizacao_*.tex             tabelas LaTeX (\\input na monografia)
  resultados/tabelas/aleatorizacao_numeros.tex       macros \\aleat... com números citáveis no texto
  resultados/figuras/fig10_aleatorizacao_niveis.png  contagem de agentes por nível e traço

Uso:
    python analise_aleatorizacao.py
"""

import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.stdout.reconfigure(encoding="utf-8")

# Raiz do repositório (este script fica em src/<pasta>/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE_DIR, "resultados")
TAB_DIR = os.path.join(RES, "tabelas")
FIG_DIR = os.path.join(RES, "figuras")

# Nome usado no texto da monografia (o CSV usa "Afabilidade")
NOME_TEXTO = {"Afabilidade": "Amabilidade"}

DIMENSOES = ["Extroversão", "Afabilidade", "Consciência", "Neuroticismo", "Abertura à Experiência"]
NIVEIS = ["Baixo", "Médio", "Alto"]
SEED = 123
N_SIM = 10_000

DATASETS = {
    "exp_a": ("Experimento A (principal)", "principal/exp_a_bfi44_multitraco_8b.csv", "multi"),
    "piloto": ("Piloto BFI-10", "piloto/dataset_llama31_8b.csv", "multi"),
    "exp_b": ("Experimento B (traço dominante)", "principal/exp_b_bfi44_dominante_8b.csv", "dominante"),
}


def cramers_v(tab):
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    k = min(tab.shape) - 1
    return np.sqrt(chi2 / (n * k))


def reproduzir_sorteio(n, modo, seed=SEED):
    """Reproduz a sequência de chamadas a `random` feita por gerar_agentes.py."""
    random.seed(seed)
    linhas = []
    for _ in range(n):
        if modo == "multi":
            p = {d: random.choice(NIVEIS) for d in DIMENSOES}
            dom = ""
        else:
            dom = random.choice(DIMENSOES)
            p = {d: ("Médio" if d != dom else random.choice(NIVEIS)) for d in DIMENSOES}
        q1 = random.choice(["a", "b", "c", "d"])
        linhas.append((p, dom, q1))
    return linhas


def analisar_multi(df, rng):
    out = {}
    n = len(df)
    num = df[[f"nivel_{d}_num" for d in DIMENSOES]].astype(int)
    num.columns = DIMENSOES

    # 1. frequências
    freq = {}
    for d in DIMENSOES:
        c = df[f"nivel_{d}"].value_counts().reindex(NIVEIS, fill_value=0)
        chi = stats.chisquare(c)
        freq[d] = {"contagens": c.tolist(), "pct": (100 * c / n).round(1).tolist(),
                   "chi2": round(chi.statistic, 3), "p": round(chi.pvalue, 3)}
    out["frequencias"] = freq

    # 2. independência entre pares
    pares = []
    for a, b in [(a, b) for i, a in enumerate(DIMENSOES) for b in DIMENSOES[i + 1:]]:
        rho, p_rho = stats.spearmanr(num[a], num[b])
        tab = pd.crosstab(df[f"nivel_{a}"], df[f"nivel_{b}"])
        chi2, p_chi, _, _ = stats.chi2_contingency(tab, correction=False)
        pares.append({"a": a, "b": b, "rho": round(rho, 3), "p_rho": round(p_rho, 3),
                      "chi2": round(chi2, 2), "p_chi2": round(p_chi, 3),
                      "cramer_v": round(cramers_v(tab), 3)})
    out["pares"] = pares

    # 3. VIF
    X = sm.add_constant(num)
    out["vif"] = {d: round(variance_inflation_factor(X.values, i + 1), 3) for i, d in enumerate(DIMENSOES)}

    # 4. combinações + Monte Carlo (mesmo sorteio, N_SIM repetições)
    combos = num.astype(str).agg("".join, axis=1)
    max_abs_rho_obs = max(abs(p["rho"]) for p in pares)
    max_dev_obs = max(max(abs(x - n / 3) for x in freq[d]["contagens"]) for d in DIMENSOES)
    sim_unicos, sim_maxrho, sim_maxdev, sim_maxrep = [], [], [], []
    for _ in range(N_SIM):
        s = rng.integers(1, 4, size=(n, 5))
        codes = s @ np.array([81, 27, 9, 3, 1])
        _, cnt = np.unique(codes, return_counts=True)
        sim_unicos.append(len(cnt))
        sim_maxrep.append(cnt.max())
        r = np.corrcoef(s, rowvar=False)  # Pearson em ranks 1-3 ~ Spearman
        sim_maxrho.append(np.abs(r[np.triu_indices(5, 1)]).max())
        sim_maxdev.append(max(np.abs(np.bincount(s[:, j], minlength=4)[1:] - n / 3).max() for j in range(5)))
    sim_unicos, sim_maxrho = np.array(sim_unicos), np.array(sim_maxrho)
    sim_maxdev, sim_maxrep = np.array(sim_maxdev), np.array(sim_maxrep)
    out["combinacoes"] = {
        "possiveis": 243, "unicas_obs": int(combos.nunique()),
        "unicas_sim_media": round(sim_unicos.mean(), 1),
        "unicas_sim_ic95": [int(np.percentile(sim_unicos, 2.5)), int(np.percentile(sim_unicos, 97.5))],
        "max_repeticoes_obs": int(combos.value_counts().max()),
        "max_repeticoes_sim_ic95": [int(np.percentile(sim_maxrep, 2.5)), int(np.percentile(sim_maxrep, 97.5))],
    }
    out["monte_carlo"] = {
        "n_sim": N_SIM,
        "max_abs_rho_obs": max_abs_rho_obs,
        "max_abs_rho_pct_sim_maior_igual": round(100 * (sim_maxrho >= max_abs_rho_obs - 1e-9).mean(), 1),
        "max_desvio_contagem_obs": round(max_dev_obs, 1),
        "max_desvio_pct_sim_maior_igual": round(100 * (sim_maxdev >= max_dev_obs - 1e-9).mean(), 1),
    }
    soma = num.sum(axis=1)
    out["soma_niveis"] = {"media": round(soma.mean(), 3), "dp": round(soma.std(ddof=1), 3),
                          "esperado_media": 10.0, "esperado_dp": round(np.sqrt(5 * 2 / 3), 3)}

    # 5. metades
    metade = np.where(np.arange(n) < n // 2, "1a", "2a")
    out["metades"] = {}
    for d in DIMENSOES:
        tab = pd.crosstab(metade, df[f"nivel_{d}"]).reindex(columns=NIVEIS, fill_value=0)
        out["metades"][d] = round(stats.chi2_contingency(tab, correction=False)[1], 3)

    # 8. impacto: coeficiente simples vs. multivariado
    # mesma variável dependente de analise_experimentos_bfi44.ipynb (escore sem a Q1 sorteada)
    q1_pts = df["ipa_q1_atribuida"].map({"a": 4, "b": 3, "c": 2, "d": 1})
    y = pd.to_numeric(df["ipa_score"], errors="coerce") - q1_pts
    ok = y.notna() & df["ipa_perfil"].isin(["Conservador", "Moderado", "Arrojado"])
    multi = sm.OLS(y[ok], sm.add_constant(num[ok])).fit()
    impacto = {}
    for d in DIMENSOES:
        simples = sm.OLS(y[ok], sm.add_constant(num.loc[ok, [d]])).fit()
        impacto[d] = {"coef_simples": round(simples.params[d], 3), "p_simples": round(simples.pvalues[d], 4),
                      "coef_multi": round(multi.params[d], 3), "p_multi": round(multi.pvalues[d], 4)}
    out["impacto"] = impacto
    out["n_ipa_valido"] = int(ok.sum())
    return out


def analisar_q1(df, num_cols):
    c = df["ipa_q1_atribuida"].value_counts().reindex(list("abcd"), fill_value=0)
    chi = stats.chisquare(c)
    indep = {}
    for d in DIMENSOES:
        tab = pd.crosstab(df["ipa_q1_atribuida"], df[f"nivel_{d}"])
        indep[d] = round(stats.chi2_contingency(tab, correction=False)[1], 3) if tab.shape[1] > 1 else None
    return {"contagens": c.tolist(), "chi2": round(chi.statistic, 3), "p": round(chi.pvalue, 3), "indep_p": indep}


def analisar_dominante(df):
    out = {}
    c = df["traco_dominante"].value_counts().reindex(DIMENSOES, fill_value=0)
    chi = stats.chisquare(c)
    out["dominante"] = {"contagens": c.tolist(), "chi2": round(chi.statistic, 3), "p": round(chi.pvalue, 3)}
    out["niveis_por_dominante"] = {}
    for d in DIMENSOES:
        s = df[df["traco_dominante"] == d][f"nivel_{d}"].value_counts().reindex(NIVEIS, fill_value=0)
        chi = stats.chisquare(s)
        out["niveis_por_dominante"][d] = {"contagens": s.tolist(), "n": int(s.sum()),
                                          "chi2": round(chi.statistic, 3), "p": round(chi.pvalue, 3)}
    # checa que os traços não-dominantes estão todos em Médio
    nao_dom_ok = all((df.loc[df["traco_dominante"] != d, f"nivel_{d}"] == "Médio").all() for d in DIMENSOES)
    out["nao_dominantes_todos_medio"] = bool(nao_dom_ok)
    return out


# ---------------------------------------------------------------------------
# Saídas para a monografia (LaTeX + figura)
# ---------------------------------------------------------------------------

def nome(d):
    return NOME_TEXTO.get(d, d)


def num(x, casas=3, sinal=False):
    """Número no formato da monografia: vírgula decimal, sinal opcional."""
    s = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
    return s.replace("-", "$-$").replace(".", ",")


def pval(p):
    return "$<$ 0,001" if p < 0.001 else num(p)


def tabela(label, caption, colunas, cabecalho, linhas, legenda, tamanho="\\small"):
    corpo = "\n".join("        " + " & ".join(l) + r" \\" for l in linhas)
    return (
        "% Gerado por analise_aleatorizacao.py -- não editar à mão; rode o script de novo.\n"
        "\\begin{table}[htb]\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "    \\centering\n"
        f"    {tamanho}\n"
        f"    \\begin{{tabular}}{{{colunas}}}\n"
        "        \\toprule\n"
        f"        {' & '.join(cabecalho)} \\\\\n"
        "        \\midrule\n"
        f"{corpo}\n"
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        f"    \\legend{{{legenda}}}\n"
        "\\end{table}\n"
    )


def gerar_latex(R):
    os.makedirs(TAB_DIR, exist_ok=True)
    a, b = R["exp_a"], R["exp_b"]
    n = a["n"]
    fonte = "Fonte: dados da pesquisa."
    arquivos = {}

    linhas = []
    for d in DIMENSOES:
        f = a["frequencias"][d]
        cel = [f"{c} ({num(p, 1)}\\%)" for c, p in zip(f["contagens"], f["pct"])]
        linhas.append([nome(d)] + cel + [num(f["chi2"], 2), num(f["p"])])
    arquivos["frequencias"] = tabela(
        "tab:aleat-frequencias",
        f"Distribuição dos níveis induzidos por traço (Experimento A e piloto, N={n})",
        "lccccc", ["Traço", "Baixo", "Médio", "Alto", "$\\chi^2$", "p-valor"], linhas,
        f"{fonte} Teste qui-quadrado de aderência à distribuição uniforme ({n // 3} agentes por nível), 2 graus de liberdade. "
        "O piloto usou o mesmo sorteio (\\textit{seed} 123).")

    linhas = []
    for p in sorted(a["pares"], key=lambda p: -abs(p["rho"])):
        linhas.append([f"{nome(p['a'])} $\\times$ {nome(p['b'])}", num(p["rho"], sinal=True),
                       num(p["p_rho"]), num(p["chi2"], 2), num(p["p_chi2"]), num(p["cramer_v"])])
    arquivos["independencia"] = tabela(
        "tab:aleat-independencia",
        "Associação entre os níveis induzidos de cada par de traços (Experimento A)",
        "lccccc", ["Par de traços", "$\\rho$", "p ($\\rho$)", "$\\chi^2$", "p ($\\chi^2$)", "V de Cramér"], linhas,
        f"{fonte} $\\rho$: correlação de Spearman; $\\chi^2$: teste de independência (4 graus de liberdade).",
        tamanho="\\footnotesize")

    c, mc = a["combinacoes"], a["monte_carlo"]
    linhas = [
        [f"Combinações distintas (de {c['possiveis']} possíveis)", str(c["unicas_obs"]),
         f"média {num(c['unicas_sim_media'], 1)}; IC 95\\% {c['unicas_sim_ic95'][0]} a {c['unicas_sim_ic95'][1]}"],
        ["Máximo de agentes com a mesma combinação", str(c["max_repeticoes_obs"]),
         f"IC 95\\% {c['max_repeticoes_sim_ic95'][0]} a {c['max_repeticoes_sim_ic95'][1]}"],
        ["Maior $|\\rho|$ entre pares de traços", num(mc["max_abs_rho_obs"]),
         f"{num(mc['max_abs_rho_pct_sim_maior_igual'], 1)}\\% das simulações $\\geq$ obtido"],
        [f"Maior desvio de uma contagem em relação a {n // 3}", str(int(mc["max_desvio_contagem_obs"])),
         f"{num(mc['max_desvio_pct_sim_maior_igual'], 1)}\\% das simulações $\\geq$ obtido"],
    ]
    arquivos["montecarlo"] = tabela(
        "tab:aleat-montecarlo",
        f"Sorteio obtido vs. {mc['n_sim']:,} sorteios simulados".replace(",", "."),
        "lcc", ["Indicador", "Obtido", "Simulação"], linhas,
        f"{fonte} Simulação: {n} agentes, 5 traços, 3 níveis equiprováveis e independentes.",
        tamanho="\\footnotesize")

    linhas = []
    for d in sorted(DIMENSOES, key=lambda d: -a["impacto"][d]["coef_multi"]):
        i = a["impacto"][d]
        linhas.append([nome(d), num(i["coef_simples"], sinal=True), pval(i["p_simples"]),
                       num(i["coef_multi"], sinal=True), pval(i["p_multi"])])
    arquivos["impacto"] = tabela(
        "tab:aleat-impacto",
        "Coeficiente de cada traço sobre o escore IPA: regressão simples vs. OLS com os 5 traços (Experimento A)",
        "lcccc", ["Traço", "Coef. simples", "p-valor", "Coef. OLS (5 traços)", "p-valor"], linhas,
        f"{fonte} Variável dependente: escore IPA sem a pergunta 1 (N={a['n_ipa_valido']}).")

    linhas = []
    for d in sorted(DIMENSOES, key=lambda d: -b["niveis_por_dominante"][d]["n"]):
        x = b["niveis_por_dominante"][d]
        linhas.append([nome(d), str(x["n"])] + [str(v) for v in x["contagens"]] + [num(x["p"])])
    arquivos["exp_b"] = tabela(
        "tab:aleat-exp-b",
        "Distribuição do traço dominante e de seus níveis (Experimento B)",
        "lccccc", ["Traço dominante", "N", "Baixo", "Médio", "Alto", "p-valor"], linhas,
        f"{fonte} p-valor: qui-quadrado de aderência à uniforme entre os 3 níveis. "
        f"Distribuição do traço dominante: $\\chi^2$ = {num(b['dominante']['chi2'], 2)}, p = {num(b['dominante']['p'])}.")

    for k, txt in arquivos.items():
        with open(os.path.join(TAB_DIR, f"aleatorizacao_{k}.tex"), "w", encoding="utf-8") as f:
            f.write(txt)

    # macros com números citáveis no texto (\aleat...), para não digitar valores à mão
    contagens = [x for d in DIMENSOES for x in a["frequencias"][d]["contagens"]]
    neuro_b = b["niveis_por_dominante"]["Neuroticismo"]
    macros = {
        "aleatN": str(n),
        "aleatContagemMin": str(min(contagens)),
        "aleatContagemMax": str(max(contagens)),
        "aleatPFreqMin": num(min(a["frequencias"][d]["p"] for d in DIMENSOES)),
        "aleatRhoMax": num(mc["max_abs_rho_obs"]),
        "aleatPRhoMin": num(min(p["p_rho"] for p in a["pares"])),
        "aleatCramerMax": num(max(p["cramer_v"] for p in a["pares"])),
        "aleatVIFMin": num(min(a["vif"].values())),
        "aleatVIFMax": num(max(a["vif"].values())),
        "aleatCombUnicas": str(c["unicas_obs"]),
        "aleatCombEsperado": num(c["unicas_sim_media"], 1),
        "aleatNSim": f"{mc['n_sim']:,}".replace(",", "."),
        "aleatSomaMedia": num(a["soma_niveis"]["media"], 2),
        "aleatSomaDP": num(a["soma_niveis"]["dp"], 2),
        "aleatPMetadesMin": num(min(a["metades"].values())),
        "aleatPMetadesMax": num(max(a["metades"].values())),
        "aleatAberturaCoefSimples": num(a["impacto"]["Abertura à Experiência"]["coef_simples"], sinal=True),
        "aleatAberturaCoefMulti": num(a["impacto"]["Abertura à Experiência"]["coef_multi"], sinal=True),
        "aleatQUmChi": num(a["q1"]["chi2"], 2),
        "aleatQUmP": num(a["q1"]["p"]),
        "aleatDominanteChi": num(b["dominante"]["chi2"], 2),
        "aleatDominanteP": num(b["dominante"]["p"]),
        "aleatNeuroBP": num(neuro_b["p"]),
    }
    with open(os.path.join(TAB_DIR, "aleatorizacao_numeros.tex"), "w", encoding="utf-8") as f:
        f.write("% Gerado por analise_aleatorizacao.py -- não editar à mão; rode o script de novo.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return list(arquivos) + ["numeros"]


def gerar_figura(R):
    """Barras agrupadas: agentes por nível em cada traço, com a linha do esperado (N/3).
    Níveis são ordinais -> uma rampa de azul (claro = Baixo, escuro = Alto)."""
    a = R["exp_a"]
    n = a["n"]
    cores = ["#9DC3E6", "#2E75B6", "#1F3864"]
    x = np.arange(len(DIMENSOES))
    larg = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for j, (niv, cor) in enumerate(zip(NIVEIS, cores)):
        vals = [a["frequencias"][d]["contagens"][j] for d in DIMENSOES]
        ax.bar(x + (j - 1) * (larg + 0.02), vals, width=larg, color=cor, label=niv, zorder=2)
    ax.axhline(n / 3, color="#555555", linestyle="--", linewidth=1.2, zorder=3)
    ax.text(len(DIMENSOES) - 0.45, n / 3 + 1.5, f"esperado ({n // 3})", ha="right", va="bottom",
            fontsize=9, color="#555555")
    rotulos = [f"{nome(d).replace(' à ', chr(10) + 'à ')}\n(p = {num(a['frequencias'][d]['p'])})" for d in DIMENSOES]
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.set_ylabel("Número de agentes")
    ax.set_ylim(0, max(max(a["frequencias"][d]["contagens"]) for d in DIMENSOES) * 1.25)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="Nível induzido", ncol=3, frameon=False, loc="upper left", fontsize=9, title_fontsize=9)
    fig.tight_layout()
    caminho = os.path.join(FIG_DIR, "fig10_aleatorizacao_niveis.png")
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
    return caminho


def main():
    rng = np.random.default_rng(2026)
    resultado = {}
    for chave, (nome, arq, modo) in DATASETS.items():
        df = pd.read_csv(os.path.join(RES, arq))
        r = {"nome": nome, "arquivo": arq, "n": len(df)}
        if modo == "multi":
            r.update(analisar_multi(df, rng))
        else:
            r.update(analisar_dominante(df))
        r["q1"] = analisar_q1(df, None)

        # 7. reprodutibilidade com a seed
        repro = reproduzir_sorteio(len(df), modo)
        bate = all(
            all(df.iloc[i][f"nivel_{d}"] == p[d] for d in DIMENSOES)
            and df.iloc[i]["ipa_q1_atribuida"] == q1
            and (modo == "multi" or df.iloc[i]["traco_dominante"] == dom)
            for i, (p, dom, q1) in enumerate(repro)
        )
        r["bate_com_seed_123"] = bool(bate)
        resultado[chave] = r

    a = pd.read_csv(os.path.join(RES, DATASETS["exp_a"][1]))
    p = pd.read_csv(os.path.join(RES, DATASETS["piloto"][1]))
    cols = [f"nivel_{d}" for d in DIMENSOES] + ["ipa_q1_atribuida"]
    resultado["piloto_igual_exp_a"] = bool((a[cols].values == p[cols].values).all())

    caminho = os.path.join(RES, "aleatorizacao.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(resultado, ensure_ascii=False, indent=2, default=float))
    print(f"\nSalvo em: {caminho}")
    print("Tabelas LaTeX:", ", ".join(gerar_latex(resultado)), f"(em {TAB_DIR})")
    print("Figura:", gerar_figura(resultado))


if __name__ == "__main__":
    main()
