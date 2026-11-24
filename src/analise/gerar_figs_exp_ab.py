# -*- coding: utf-8 -*-
"""Gera fig8 e fig9: comparação Experimento A (5 traços juntos, N=300) vs.
Experimento B (traço isolado, N=49-69 por traço) -- uma para checagem de
manipulação, outra para correlação com o escore IPA. Usadas no Capítulo 4
(seção "Experimento B") e referenciadas no Capítulo 5.

Diferente de regen_figs.py (que regenera fig1-4 a partir de um único
dataset), este script sempre lê os dois datasets dos Experimentos A e B
juntos -- não precisa parametrizar."""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Caminhos relativos à raiz do repositório (este script fica em src/analise/)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTDIR = 'resultados/figuras'

DIMS = ['Extroversão', 'Afabilidade', 'Consciência', 'Neuroticismo', 'Abertura à Experiência']
nomes_curtos = ['Extroversão', 'Amabilidade', 'Consciência', 'Neuroticismo', 'Abertura\nà Experiência']

df_a = pd.read_csv('resultados/principal/exp_a_bfi44_multitraco_8b.csv')
df_a = df_a[df_a['ipa_perfil'].isin(['Conservador', 'Moderado', 'Arrojado'])].copy()
df_b = pd.read_csv('resultados/principal/exp_b_bfi44_dominante_8b.csv')
df_b = df_b[df_b['ipa_perfil'].isin(['Conservador', 'Moderado', 'Arrojado'])].copy()
for df in (df_a, df_b):
    q1 = df['ipa_q1_atribuida'].map({'a': 4, 'b': 3, 'c': 2, 'd': 1})
    df['ipa_score_sem_q1'] = df['ipa_score'] - q1


def sig(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))


def montar_grafico(coluna_alvo, titulo, arquivo, xlim):
    rho_a, p_a, rho_b, p_b, n_b = [], [], [], [], []
    for d in DIMS:
        col_a = df_a[f'bfi_{d}'] if coluna_alvo == 'bfi' else df_a[coluna_alvo]
        r, p = stats.spearmanr(df_a[f'nivel_{d}_num'], col_a)
        rho_a.append(r); p_a.append(p)
        sub = df_b[df_b['traco_dominante'] == d]
        n_b.append(len(sub))
        col_b = sub[f'bfi_{d}'] if coluna_alvo == 'bfi' else sub[coluna_alvo]
        r2, p2 = stats.spearmanr(sub[f'nivel_{d}_num'], col_b)
        rho_b.append(r2); p_b.append(p2)

    y = np.arange(len(DIMS))
    altura = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.2))
    barras_a = ax.barh(y + altura / 2, rho_a, height=altura, color='#2E75B6', alpha=0.9,
                        label='Experimento A (5 traços juntos, N=300)')
    barras_b = ax.barh(y - altura / 2, rho_b, height=altura, color='#F5B041', alpha=0.9,
                        label='Experimento B (traço isolado, N por traço abaixo)')

    for bar, r, p in zip(barras_a, rho_a, p_a):
        ax.text(r + 0.02 if r >= 0 else r - 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{r:+.3f} {sig(p)}', va='center', ha='left' if r >= 0 else 'right', fontsize=9)
    for bar, r, p, n in zip(barras_b, rho_b, p_b, n_b):
        ax.text(r + 0.02 if r >= 0 else r - 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{r:+.3f} {sig(p)}  (N={n})', va='center', ha='left' if r >= 0 else 'right', fontsize=9)

    ax.set_yticks(y); ax.set_yticklabels(nomes_curtos, fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_xlabel('Correlação de Spearman (ρ)', fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold', color='#1F3864')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9, frameon=True)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/{arquivo}', dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(arquivo, 'ok')


montar_grafico(
    'bfi',
    'Checagem de manipulação: 5 traços juntos (A) vs. traço isolado (B)\nllama3.1:8b, BFI-44',
    'fig8_exp_ab_manipulacao.png',
    (0, 1.15),
)
montar_grafico(
    'ipa_score_sem_q1',
    'Correlação com o escore IPA: 5 traços juntos (A) vs. traço isolado (B)\nllama3.1:8b, BFI-44',
    'fig9_exp_ab_ipa.png',
    (-0.3, 1.05),
)
