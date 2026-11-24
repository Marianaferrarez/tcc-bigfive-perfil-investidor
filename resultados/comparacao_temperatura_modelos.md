# Temperatura e outros modelos — resultados

Gerado por `analise_temperatura_modelos.py`. Significância: * p<0,05; ** p<0,01; *** p<0,001; n.s. p≥0,05.

## Temperatura (llama3.1:8b, N=300)

| Indicador | 0,0 | 0,3 | 0,7 | 1,0 |
|---|---|---|---|---|
| N (válidos) | 300 (299) | 300 (299) | 300 (300) | 300 (300) |
| Respostas válidas BFI / IPA (%) | 100.0 / 99.7 | 100.0 / 99.7 | 100.0 / 100.0 | 100.0 / 100.0 |
| Tentativas médias BFI / IPA | 1.0 / 1.01 | 1.0 / 1.02 | n/d / n/d | 1.13 / 1.07 |
| Perfis C / M / A (%) | 2.0 / 86.6 / 11.4 | 3.7 / 84.9 / 11.4 | 3.7 / 82.3 / 14.0 | 4.0 / 81.0 / 15.0 |
| IPA Q2-Q8 média (dp) | 17.05 (2.41) | 17.09 (2.47) | 17.58 (2.72) | 17.67 (2.77) |
| Manipulação ρ — Extroversão | +0.946 *** | +0.938 *** | +0.929 *** | +0.916 *** |
| Manipulação ρ — Amabilidade | +0.880 *** | +0.859 *** | +0.813 *** | +0.809 *** |
| Manipulação ρ — Consciência | +0.929 *** | +0.891 *** | +0.879 *** | +0.871 *** |
| Manipulação ρ — Neuroticismo | +0.873 *** | +0.870 *** | +0.835 *** | +0.816 *** |
| Manipulação ρ — Abertura à Experiência | +0.929 *** | +0.928 *** | +0.913 *** | +0.920 *** |
| ρ nível × IPA — Extroversão | +0.536 *** | +0.413 *** | +0.340 *** | +0.364 *** |
| ρ nível × IPA — Amabilidade | +0.109 n.s. | +0.107 n.s. | +0.051 n.s. | +0.098 n.s. |
| ρ nível × IPA — Consciência | +0.029 n.s. | +0.130 * | +0.206 *** | +0.088 n.s. |
| ρ nível × IPA — Neuroticismo | +0.168 ** | +0.106 n.s. | +0.136 * | +0.214 *** |
| ρ nível × IPA — Abertura à Experiência | +0.624 *** | +0.654 *** | +0.530 *** | +0.507 *** |
| OLS R² | 0.707 | 0.643 | 0.459 | 0.449 |
| OLS coef — Extroversão | +1.595 *** | +1.273 *** | +1.189 *** | +1.253 *** |
| OLS coef — Amabilidade | +0.359 *** | +0.344 ** | +0.094 n.s. | +0.246 n.s. |
| OLS coef — Consciência | +0.429 *** | +0.486 *** | +0.750 *** | +0.301 * |
| OLS coef — Neuroticismo | +0.419 *** | +0.278 * | +0.407 ** | +0.744 *** |
| OLS coef — Abertura à Experiência | +1.767 *** | +1.932 *** | +1.716 *** | +1.686 *** |

### Comparação pareada com a temperatura 0,7 (mesmos perfis)

| Temperatura | Pares | ρ escore IPA | Mesmo perfil IPA (%) | Respostas Q2-Q8 iguais (%) |
|---|---|---|---|---|
| 0,0 | 299 | +0.620 *** | 82.9 | 74.3 |
| 0,3 | 299 | +0.583 *** | 82.9 | 71.7 |
| 1,0 | 300 | +0.526 *** | 77.3 | 62.0 |

## Outros modelos (piloto, mesmos perfis dos primeiros agentes do Experimento A)

| Indicador | llama3.1:8b (Exp. A, 30 primeiros) | gemma-4-31b | nemotron-3.5-super | nemotron-3-ultra |
|---|---|---|---|---|
| N (válidos) | 30 (30) | 2 (2) | 30 (30) | 30 (30) |
| Respostas válidas BFI / IPA (%) | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| Tentativas médias BFI / IPA | n/d / n/d | 1.0 / 1.0 | 1.0 / 1.03 | 1.0 / 1.03 |
| Perfis C / M / A (%) | 3.3 / 83.3 / 13.3 | 0.0 / 100.0 / 0.0 | 56.7 / 13.3 / 30.0 | 40.0 / 20.0 / 40.0 |
| IPA Q2-Q8 média (dp) | 18.0 (2.67) | 14.0 (2.83) | 15.07 (7.71) | 16.83 (7.27) |
| Manipulação ρ — Extroversão | +0.937 *** | n/d | +0.949 *** | +0.946 *** |
| Manipulação ρ — Amabilidade | +0.776 *** | n/d | +0.918 *** | +0.925 *** |
| Manipulação ρ — Consciência | +0.898 *** | n/d | +0.950 *** | +0.966 *** |
| Manipulação ρ — Neuroticismo | +0.902 *** | n/d | +0.990 *** | +0.958 *** |
| Manipulação ρ — Abertura à Experiência | +0.901 *** | n/d | +0.950 *** | +0.942 *** |
| ρ nível × IPA — Extroversão | +0.409 * | n/d | +0.501 ** | +0.419 * |
| ρ nível × IPA — Amabilidade | +0.079 n.s. | n/d | -0.059 n.s. | -0.405 * |
| ρ nível × IPA — Consciência | +0.054 n.s. | n/d | +0.186 n.s. | +0.190 n.s. |
| ρ nível × IPA — Neuroticismo | -0.238 n.s. | n/d | -0.295 n.s. | -0.287 n.s. |
| ρ nível × IPA — Abertura à Experiência | +0.458 * | n/d | +0.590 *** | +0.684 *** |
| OLS R² | 0.455 | n/d | 0.659 | 0.799 |
| OLS coef — Extroversão | +1.322 * | n/d | +4.035 ** | +2.060 * |
| OLS coef — Amabilidade | +0.497 n.s. | n/d | -0.622 n.s. | -3.356 ** |
| OLS coef — Consciência | +0.106 n.s. | n/d | +1.443 n.s. | +1.274 n.s. |
| OLS coef — Neuroticismo | -0.270 n.s. | n/d | -0.541 n.s. | -1.543 n.s. |
| OLS coef — Abertura à Experiência | +1.637 ** | n/d | +6.096 *** | +6.751 *** |

### Comparação pareada com o llama3.1:8b (mesmos perfis)

| Modelo | Pares | ρ escore IPA | Mesmo perfil IPA (%) | Respostas Q2-Q8 iguais (%) |
|---|---|---|---|---|
| nemotron-3.5-super | 30 | +0.536 ** | 30.0 | 25.7 |
| nemotron-3-ultra | 30 | +0.500 ** | 36.7 | 30.5 |

## Outra semente (llama3.1:8b, 0,7, N=300, perfis diferentes)

| Indicador | seed 123 (Exp. A) | seed 456 |
|---|---|---|
| N (válidos) | 300 (300) | 300 (298) |
| Respostas válidas BFI / IPA (%) | 100.0 / 100.0 | 100.0 / 99.3 |
| Tentativas médias BFI / IPA | n/d / n/d | 1.06 / 1.04 |
| Perfis C / M / A (%) | 3.7 / 82.3 / 14.0 | 6.7 / 80.2 / 13.1 |
| IPA Q2-Q8 média (dp) | 17.58 (2.72) | 17.5 (2.77) |
| Manipulação ρ — Extroversão | +0.929 *** | +0.928 *** |
| Manipulação ρ — Amabilidade | +0.813 *** | +0.823 *** |
| Manipulação ρ — Consciência | +0.879 *** | +0.860 *** |
| Manipulação ρ — Neuroticismo | +0.835 *** | +0.836 *** |
| Manipulação ρ — Abertura à Experiência | +0.913 *** | +0.924 *** |
| ρ nível × IPA — Extroversão | +0.340 *** | +0.382 *** |
| ρ nível × IPA — Amabilidade | +0.051 n.s. | -0.003 n.s. |
| ρ nível × IPA — Consciência | +0.206 *** | -0.010 n.s. |
| ρ nível × IPA — Neuroticismo | +0.136 * | +0.028 n.s. |
| ρ nível × IPA — Abertura à Experiência | +0.530 *** | +0.612 *** |
| OLS R² | 0.459 | 0.526 |
| OLS coef — Extroversão | +1.189 *** | +1.293 *** |
| OLS coef — Amabilidade | +0.094 n.s. | +0.169 n.s. |
| OLS coef — Consciência | +0.750 *** | +0.048 n.s. |
| OLS coef — Neuroticismo | +0.407 ** | -0.092 n.s. |
| OLS coef — Abertura à Experiência | +1.716 *** | +2.106 *** |

## Modelos abertos locais (N=300, mesmos perfis do Experimento A)

| Indicador | llama3.1:8b (Exp. A) | qwen2.5:7b | mistral:7b | gemma3:12b |
|---|---|---|---|---|
| N (válidos) | 300 (300) | 300 (245) | 300 (294) | 300 (300) |
| Respostas válidas BFI / IPA (%) | 100.0 / 100.0 | 95.3 / 85.7 | 98.0 / 100.0 | 100.0 / 100.0 |
| Tentativas médias BFI / IPA | n/d / n/d | 1.32 / 1.55 | 1.18 / 1.02 | 1.01 / 1.0 |
| Perfis C / M / A (%) | 3.7 / 82.3 / 14.0 | 4.5 / 51.0 / 44.5 | 0.0 / 8.5 / 91.5 | 24.0 / 59.3 / 16.7 |
| IPA Q2-Q8 média (dp) | 17.58 (2.72) | 19.48 (4.34) | 22.86 (1.11) | 15.77 (3.89) |
| Manipulação ρ — Extroversão | +0.929 *** | +0.914 *** | +0.897 *** | +0.955 *** |
| Manipulação ρ — Amabilidade | +0.813 *** | +0.873 *** | +0.872 *** | +0.951 *** |
| Manipulação ρ — Consciência | +0.879 *** | +0.888 *** | +0.925 *** | +0.943 *** |
| Manipulação ρ — Neuroticismo | +0.835 *** | +0.846 *** | +0.881 *** | +0.957 *** |
| Manipulação ρ — Abertura à Experiência | +0.913 *** | +0.846 *** | +0.939 *** | +0.957 *** |
| ρ nível × IPA — Extroversão | +0.340 *** | +0.267 *** | +0.184 ** | +0.503 *** |
| ρ nível × IPA — Amabilidade | +0.051 n.s. | -0.115 n.s. | -0.175 ** | +0.011 n.s. |
| ρ nível × IPA — Consciência | +0.206 *** | -0.045 n.s. | -0.011 n.s. | +0.209 *** |
| ρ nível × IPA — Neuroticismo | +0.136 * | +0.104 n.s. | +0.175 ** | -0.067 n.s. |
| ρ nível × IPA — Abertura à Experiência | +0.530 *** | +0.817 *** | +0.416 *** | +0.619 *** |
| OLS R² | 0.459 | 0.796 | 0.383 | 0.648 |
| OLS coef — Extroversão | +1.189 *** | +1.551 *** | +0.391 *** | +2.697 *** |
| OLS coef — Amabilidade | +0.094 n.s. | -0.317 * | -0.287 *** | -0.193 n.s. |
| OLS coef — Consciência | +0.750 *** | -0.426 ** | +0.034 n.s. | +0.534 ** |
| OLS coef — Neuroticismo | +0.407 ** | +0.322 * | +0.285 *** | -0.125 n.s. |
| OLS coef — Abertura à Experiência | +1.716 *** | +4.643 *** | +0.632 *** | +2.622 *** |

### Comparação pareada com o llama3.1:8b (mesmos perfis)

| Modelo | Pares | ρ escore IPA | Mesmo perfil IPA (%) | Respostas Q2-Q8 iguais (%) |
|---|---|---|---|---|
| qwen2.5:7b | 257 | +0.559 *** | 54.5 | 54.9 |
| mistral:7b | 300 | +0.348 *** | 20.3 | 36.5 |
| gemma3:12b | 300 | +0.560 *** | 62.0 | 43.5 |
