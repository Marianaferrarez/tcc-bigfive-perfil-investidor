import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
from scipy import stats
import statsmodels.api as sm
import os

# O diretório de trabalho é a raiz do repositório (este script fica em
# src/analise/), para funcionar independente de onde o repo estiver clonado.
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

NIVEL_ORDEM = ['Baixo', 'Médio', 'Alto']
PERFIL_ORDEM = ['Conservador', 'Moderado', 'Arrojado']
CORES = {'Conservador': '#E74C3C', 'Moderado': '#F5B041', 'Arrojado': '#27AE60'}

# Dataset principal: Experimento A (BFI-44, multi-traço, N=300).
# dataset_llama31_8b.csv (BFI-10) é o estudo piloto -- suas figuras foram
# preservadas como fig*_piloto.png antes de rodar este script pela primeira
# vez com o dataset novo (2026-08-28). Se precisar regenerar as figuras do
# piloto, troque a linha abaixo temporariamente.
df = pd.read_csv('resultados/principal/exp_a_bfi44_multitraco_8b.csv')
df = df[df['ipa_perfil'].isin(PERFIL_ORDEM)].copy()
df['ipa_score_sem_q1'] = df['ipa_score'] - df['ipa_q1_atribuida'].map({'a':4,'b':3,'c':2,'d':1})

DIMS = ['Extroversão', 'Afabilidade', 'Consciência', 'Neuroticismo', 'Abertura à Experiência']
# 'Afabilidade' em DIMS é o sufixo real das colunas do CSV
# (nivel_Afabilidade, bfi_Afabilidade -- gerar_agentes.py/perfis_personalidade.json
# usam esse nome internamente) e não deve ser alterado, senão os lookups
# df[f'nivel_{d}_num'] etc. quebram. O nome usado na monografia é
# "Amabilidade" (terminologia fixada no texto); ROTULO traduz DIMS só para
# exibição nos gráficos, sem tocar nos lookups de dado.
ROTULO = {d: ('Amabilidade' if d == 'Afabilidade' else d) for d in DIMS}
nomes_curtos = [ROTULO[d] if d != 'Abertura à Experiência' else 'Abertura\nà Experiência' for d in DIMS]
OUTDIR = 'resultados/figuras'


# ── Formatação em português ──────────────────────────────────────────────────
# Vírgula decimal e sinal de menos tipográfico, como no texto da monografia.
# As figuras não têm título próprio: o título fica na legenda (\caption) do
# LaTeX, conforme a ABNT.
def num(x, casas=3, sinal=False):
    s = f'{x:+.{casas}f}' if sinal else f'{x:.{casas}f}'
    return s.replace('-', '−').replace('.', ',')


def sig_de(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))


virgula = matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:g}'.replace('-', '−').replace('.', ','))


def preparar(caminho):
    d = pd.read_csv(caminho)
    d = d[d['ipa_perfil'].isin(PERFIL_ORDEM)].copy()
    d['ipa_score_sem_q1'] = d['ipa_score'] - d['ipa_q1_atribuida'].map({'a': 4, 'b': 3, 'c': 2, 'd': 1})
    return d


df_piloto = preparar('resultados/piloto/dataset_llama31_8b.csv')
# (dataset, sufixo do arquivo, instrumento de autorrelato)
VERSOES = [(df, '', 'BFI-44'), (df_piloto, '_piloto', 'BFI-10')]


# ── Fig 1: Spearman (nível induzido x escore IPA) ───────────────────────────
def gerar_fig1(dados, sufixo):
    rhos, pvals = [], []
    for d in DIMS:
        r, p = stats.spearmanr(dados[f'nivel_{d}_num'], dados['ipa_score_sem_q1'])
        rhos.append(r); pvals.append(p)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(nomes_curtos, rhos, color=['#2E75B6' if r > 0 else '#c0392b' for r in rhos], alpha=0.85, height=0.55)
    for bar, r, p in zip(bars, rhos, pvals):
        ax.text(r + 0.008 if r >= 0 else r - 0.008, bar.get_y() + bar.get_height() / 2,
                f'ρ = {num(r, sinal=True)} {sig_de(p)}', va='center', ha='left' if r >= 0 else 'right', fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(-0.25, 0.62)
    ax.xaxis.set_major_formatter(virgula)
    ax.set_xlabel('Correlação de Spearman (ρ) com o escore IPA (Q2–Q8)', fontsize=11)
    ax.invert_yaxis(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/fig1_spearman{sufixo}.png', dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close()


# ── Fig 2: Boxplots (escore IPA por nível induzido, um painel por traço) ────
# Grade 2x3 (o sexto espaço fica vazio), para que os painéis fiquem legíveis
# na largura da página da monografia.
def gerar_boxplots(dados_df, sufixo):
    fig_b, axes_b = plt.subplots(2, 3, figsize=(11, 8.0), sharey=True)
    for ax, trait in zip(axes_b.flat, DIMS):
        dados = [dados_df[dados_df[f'nivel_{trait}'] == n]['ipa_score_sem_q1'].values for n in NIVEL_ORDEM]
        bp = ax.boxplot(dados, tick_labels=NIVEL_ORDEM, patch_artist=True, widths=0.55,
                        medianprops=dict(color='#1F3864', linewidth=2),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))
        for patch in bp['boxes']:
            patch.set_facecolor('#AED6F1')
            patch.set_alpha(0.85)
        r, p = stats.spearmanr(dados_df[f'nivel_{trait}_num'], dados_df['ipa_score_sem_q1'])
        ax.set_title(f'{ROTULO[trait]}\nρ = {num(r, sinal=True)} {sig_de(p)}', fontsize=13, fontweight='bold',
                     color='#1F3864' if p < 0.05 else '#888888')
        ax.tick_params(axis='both', labelsize=12)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for ax in axes_b[:, 0]:
        ax.set_ylabel('Escore IPA (Q2–Q8)', fontsize=12)
    axes_b[1, 2].axis('off')
    plt.tight_layout(pad=0.8, h_pad=2.0)
    plt.savefig(f'{OUTDIR}/fig2_boxplots{sufixo}.png', dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close()


# ── Fig 3: OLS ───────────────────────────────────────────────────────────────
def gerar_fig3(dados, sufixo):
    X = dados[[f'nivel_{d}_num' for d in DIMS]]
    m = sm.OLS(dados['ipa_score_sem_q1'], sm.add_constant(X)).fit()
    coefs = [m.params[f'nivel_{d}_num'] for d in DIMS]
    errs = [m.bse[f'nivel_{d}_num'] * 1.96 for d in DIMS]
    pvals_ols = [m.pvalues[f'nivel_{d}_num'] for d in DIMS]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (coef, err, p) in enumerate(zip(coefs, errs, pvals_ols)):
        ax.barh(i, coef, color='#2E75B6' if p < 0.05 else '#D5D5D5', alpha=0.85, height=0.5)
        ax.errorbar(coef, i, xerr=err, color='#2c3e50', linewidth=2, capsize=5, capthick=2)
        ax.text(coef + err + 0.03 if coef >= 0 else coef - err - 0.03, i,
                f'{num(coef, sinal=True)} {sig_de(p)}', va='center', ha='left' if coef >= 0 else 'right', fontsize=10)
    ax.set_yticks(range(len(DIMS))); ax.set_yticklabels(nomes_curtos, fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(virgula)
    ax.set_xlabel('Coeficiente OLS (barras de erro: IC 95%)', fontsize=11)
    ax.invert_yaxis(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/fig3_ols{sufixo}.png', dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close()


# ── Fig 4: Manipulação ───────────────────────────────────────────────────────
def gerar_fig4(dados, sufixo, instrumento):
    rhos_m = [stats.spearmanr(dados[f'nivel_{d}_num'], dados[f'bfi_{d}'])[0] for d in DIMS]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    bars = ax.barh(nomes_curtos, rhos_m, color=['#27ae60' if r >= 0.5 else '#f39c12' for r in rhos_m], alpha=0.85, height=0.55)
    for bar, r in zip(bars, rhos_m):
        ax.text(r + 0.01, bar.get_y() + bar.get_height() / 2,
                f'ρ = {num(r)}', va='center', ha='left', fontsize=10, fontweight='bold')
    ax.axvline(0.5, color='#27ae60', linewidth=1.5, linestyle='--', alpha=0.6, label='Forte (ρ ≥ 0,50)')
    ax.axvline(0.3, color='#f39c12', linewidth=1.5, linestyle='--', alpha=0.6, label='Moderada (ρ ≥ 0,30)')
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(virgula)
    ax.set_xlabel(f'Correlação de Spearman (ρ) entre nível induzido e autorrelato ({instrumento})', fontsize=11)
    # Legenda abaixo do eixo, para não sobrepor as barras mais longas.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, frameon=False)
    ax.invert_yaxis(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig(f'{OUTDIR}/fig4_manipulacao{sufixo}.png', dpi=150, bbox_inches='tight', pad_inches=0.15)
    plt.close()


for dados, sufixo, instrumento in VERSOES:
    gerar_fig1(dados, sufixo)
    gerar_boxplots(dados, sufixo)
    gerar_fig3(dados, sufixo)
    gerar_fig4(dados, sufixo, instrumento)
print('fig1-fig4 ok (Experimento A e estudo piloto)')


# ── Fig 5, 6, 7: ATENÇÃO -- têm ρ HARDCODED como texto (não recalculado a
# partir de df) e fig5/fig7 assumem especificamente que só Abertura e
# Extroversão são significativos, o que era verdade no piloto BFI-10 mas
# não é mais com o Experimento A (4 dos 5 traços significativos agora).
# NÃO rode esta parte do script no dataset atual sem reescrever -- os
# fig5/6/7 atuais em resultados/figuras/ continuam sendo os do piloto
# (preservados como fig*_piloto.png antes desta migração, 2026-08-28) e
# não foram substituídos. Regenerar para o Experimento A fica como
# trabalho pendente, não crítico para o texto do TCC.
# ── Fig 5: barras empilhadas (2 traços) ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, trait, rho in zip(axes, ['Abertura à Experiência', 'Extroversão'], ['+0,326', '+0,221']):
    tab = pd.crosstab(df[f'nivel_{trait}'], df['ipa_perfil'], normalize='index') * 100
    tab = tab.reindex(index=NIVEL_ORDEM, columns=[p for p in PERFIL_ORDEM if p in tab.columns])
    tab.plot(kind='bar', stacked=True, ax=ax,
             color=[CORES[c] for c in tab.columns], width=0.55, legend=False)
    for cont in ax.containers:
        labels = [f'{v:.0f}%' if v >= 4 else '' for v in cont.datavalues]
        ax.bar_label(cont, labels=labels, label_type='center', color='white', fontsize=13, fontweight='bold')
    ax.set_title(f'{trait}\nρ = {rho}  ***', fontsize=13, fontweight='bold', color='#1F3864')
    ax.set_xlabel('Nível Induzido', fontsize=11)
    ax.set_ylabel('% de Agentes', fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

handles = [plt.Rectangle((0,0),1,1, color=CORES[p]) for p in PERFIL_ORDEM]
fig.legend(handles, PERFIL_ORDEM, loc='upper center', ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(pad=0.8, rect=[0, 0, 1, 0.92])
plt.savefig(f'{OUTDIR}/fig5_barras_perfil.png', dpi=150, bbox_inches='tight', pad_inches=0.15)
plt.close()
print('fig5 ok')


# ── Fig 6: visão geral 5 traços ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
rhos_vals = ['+0,221***', '+0,083 n.s.', '+0,013 n.s.', '+0,044 n.s.', '+0,326***']
sig_flags = [True, False, False, False, True]
for ax, trait, rho_t, is_sig in zip(axes2, DIMS, rhos_vals, sig_flags):
    tab = pd.crosstab(df[f'nivel_{trait}'], df['ipa_perfil'], normalize='index') * 100
    tab = tab.reindex(index=NIVEL_ORDEM, columns=[p for p in PERFIL_ORDEM if p in tab.columns])
    tab.plot(kind='bar', stacked=True, ax=ax, color=[CORES[c] for c in tab.columns], width=0.6, legend=False)
    for cont in ax.containers:
        labels = [f'{v:.0f}%' if v >= 6 else '' for v in cont.datavalues]
        ax.bar_label(cont, labels=labels, label_type='center', color='white', fontsize=8, fontweight='bold')
    short = trait.replace('à Experiência', '').strip()
    ax.set_title(f'{short}\nρ={rho_t}', fontsize=10, fontweight='bold', color='#1F3864' if is_sig else '#AAAAAA')
    ax.set_xlabel(''); ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if ax != axes2[0]: ax.set_ylabel('')
    else: ax.set_ylabel('% de Agentes', fontsize=10)

handles = [plt.Rectangle((0,0),1,1, color=CORES[p]) for p in PERFIL_ORDEM]
fig2.legend(handles, PERFIL_ORDEM, loc='upper center', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(pad=0.6, rect=[0, 0, 1, 0.92])
plt.savefig(f'{OUTDIR}/fig6_visao_geral.png', dpi=150, bbox_inches='tight', pad_inches=0.15)
plt.close()
print('fig6 ok')


# ── Fig 7: gráfico de linhas ─────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 4))
cores_linha = {'Abertura à Experiência': '#27AE60', 'Extroversão': '#2E75B6',
               'Afabilidade': '#D5D5D5', 'Consciência': '#D5D5D5', 'Neuroticismo': '#D5D5D5'}
for trait in DIMS:
    medias = [df[df[f'nivel_{trait}'] == n]['ipa_score_sem_q1'].mean() for n in NIVEL_ORDEM]
    label = ROTULO[trait].replace('à Experiência', '')
    is_sig = trait in ['Abertura à Experiência', 'Extroversão']
    ax3.plot([0, 1, 2], medias,
             marker='o' if is_sig else '.',
             markersize=10 if is_sig else 6,
             color=cores_linha[trait],
             linewidth=3 if is_sig else 1.5,
             label=label,
             zorder=5 if is_sig else 2,
             alpha=0.9 if is_sig else 0.4)
ax3.set_xticks([0, 1, 2]); ax3.set_xticklabels(NIVEL_ORDEM, fontsize=12)
ax3.set_xlabel('Nível Induzido de Personalidade', fontsize=11)
ax3.set_ylabel('Escore Médio IPA (Q2–Q8)', fontsize=11)
ax3.set_title('Tendência: Escore IPA por Nível de Personalidade\nTraços significativos em destaque', fontsize=12, fontweight='bold', color='#1F3864')
ax3.legend(fontsize=10, loc='upper left', frameon=True, framealpha=0.9)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.grid(axis='y', alpha=0.2)
plt.tight_layout(pad=0.3)
plt.savefig(f'{OUTDIR}/fig7_tendencia_linhas.png', dpi=150, bbox_inches='tight', pad_inches=0.15)
plt.close()
print('fig7 ok')

print('TODAS AS FIGURAS REGENERADAS')
