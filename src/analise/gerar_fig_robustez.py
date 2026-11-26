"""Figura de síntese da análise de robustez e generalização.

Lê os coeficientes OLS das oito rodadas com N=300 (quatro temperaturas,
semente 456 e três modelos abertos) em
resultados/comparacao_temperatura_modelos.json, gerado por
analise_temperatura_modelos.py, e desenha um painel por traço.
Saída: resultados/figuras/fig11_robustez_coeficientes.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Raiz do repositório (este script fica em src/<pasta>/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENTRADA = os.path.join(BASE_DIR, "resultados", "comparacao_temperatura_modelos.json")
SAIDA = os.path.join(BASE_DIR, "resultados", "figuras", "fig11_robustez_coeficientes.png")

# Mesma ordem de traços e mesmas cores das demais figuras (regen_figs.py)
TRACOS = [("Abertura à Experiência", "Abertura\nà Experiência"), ("Extroversão", "Extroversão"),
          ("Consciência", "Consciência"), ("Neuroticismo", "Neuroticismo"),
          ("Afabilidade", "Amabilidade")]
COR_SIG, COR_NS = "#2E75B6", "#9A9A9A"


def rodadas(dados):
    """Lista (rótulo, coeficientes) das oito rodadas com N=300, na ordem da Tabela de síntese."""
    r = [(f"Temperatura {t}" + (" (Exp. A)" if t == "0,7" else ""), dados["temperatura"][t]["ols"]["coef"])
         for t in ("0,0", "0,3", "0,7", "1,0")]
    r.append(("Semente 456", dados["sementes"]["seed 456"]["ols"]["coef"]))
    for m in ("qwen2.5:7b", "mistral:7b", "gemma3:12b"):
        r.append((m, dados["modelos_locais"][m]["ols"]["coef"]))
    return r


def main():
    with open(ENTRADA, encoding="utf-8") as f:
        lista = rodadas(json.load(f))
    rotulos = [nome for nome, _ in lista]
    y = list(range(len(lista)))

    fig, eixos = plt.subplots(1, len(TRACOS), figsize=(11, 4.2), sharey=True, sharex=True)
    for ax, (chave, titulo) in zip(eixos, TRACOS):
        for yi, (_, coef) in zip(y, lista):
            b, p = coef[chave]["b"], coef[chave]["p"]
            sig = p < 0.05
            ax.plot(b, yi, "o", markersize=8, color=COR_SIG if sig else "white",
                    markeredgecolor=COR_SIG if sig else COR_NS, markeredgewidth=1.6, zorder=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axhline(4.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
        ax.set_title(titulo, fontsize=11, fontweight="bold", color="#1F3864")
        ax.grid(axis="x", color="#EEEEEE", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
    eixos[0].set_yticks(y)
    eixos[0].set_yticklabels(rotulos, fontsize=10)
    eixos[0].invert_yaxis()
    fig.supxlabel("Coeficiente OLS (efeito de um passo de nível induzido sobre o escore IPA)", fontsize=10)
    marcador_sig = plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=COR_SIG, label="p < 0,05")
    marcador_ns = plt.Line2D([], [], marker="o", linestyle="", markersize=8, color="white",
                             markeredgecolor=COR_NS, markeredgewidth=1.6, label="não significativo")
    fig.legend(handles=[marcador_sig, marcador_ns], loc="upper right", ncol=2, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(SAIDA, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print("ok", SAIDA)


if __name__ == "__main__":
    main()
