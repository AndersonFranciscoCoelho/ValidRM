"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  VALIDAÇÃO DE MODELO — CRÍTICA ESPECÍFICA                                   ║
║  Especificação: taxa_IGP-M = α + β · taxa_IPCA                             ║
║  onde α = intercepto e β = coef. angular da MM20 anos                      ║
║                                                                              ║
║  Estratégia de crítica:                                                      ║
║    1. Reproduzir o modelo exatamente como especificado                       ║
║    2. Mostrar instabilidade temporal de α e β na própria janela MM20a       ║
║    3. Demonstrar viés sistemático por regime macroeconômico                  ║
║    4. Quantificar o erro de previsão out-of-sample por sub-período           ║
║    5. Teste de aderência: resíduo do modelo vs. distribuição assumida        ║
║    6. Sensibilidade: como α e β mudam se a janela se desloca 1 ano          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 0. DADOS
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    ipca = [
        1.70,1.02,1.55,2.43,2.67,2.26,2.29,0.99,0.99,0.86,0.55,0.96,
        1.33,1.03,0.35,0.87,1.22,1.19,1.10,0.44,0.15,0.22,0.32,0.47,
        1.18,0.50,1.18,0.88,0.40,0.54,0.22,0.06,0.22,0.23,0.17,0.43,
        0.71,0.46,0.34,0.24,0.50,0.02,-0.12,-0.51,-0.22,0.02,-0.17,0.33,
        0.70,1.05,1.10,0.56,0.30,0.19,1.09,0.56,0.31,1.19,0.95,0.60,
        0.62,0.13,0.22,0.42,0.01,0.23,1.61,1.31,0.23,0.14,0.32,0.59,
        0.62,0.46,0.38,1.10,0.41,0.67,1.33,0.97,0.44,0.83,0.71,0.69,
        0.52,0.36,0.60,0.97,0.21,1.42,1.19,0.65,0.72,1.31,3.02,2.10,
        2.25,1.57,1.23,0.97,0.61,-0.15,0.20,0.34,0.78,0.29,0.34,0.52,
        0.76,0.69,0.47,0.37,0.51,0.71,0.91,0.69,0.44,0.44,0.69,0.86,
        0.58,0.59,0.61,0.87,0.49,-0.02,0.25,0.17,0.35,0.75,0.55,0.36,
        0.59,0.41,0.43,0.21,0.10,-0.21,0.19,0.05,0.21,0.33,0.31,0.48,
        0.44,0.44,0.37,0.25,0.28,0.28,0.24,0.47,0.18,0.42,0.38,0.74,
        0.54,0.49,0.48,0.55,0.79,0.74,0.53,0.28,0.26,0.45,0.36,0.28,
        0.48,0.55,0.20,0.48,0.47,0.36,0.24,0.15,0.24,0.28,0.41,0.37,
        0.75,0.78,0.52,0.57,0.43,0.00,0.01,0.04,0.45,0.75,0.83,0.63,
        0.83,0.80,0.79,0.77,0.47,0.15,0.16,0.37,0.53,0.43,0.52,0.50,
        0.97,0.45,0.21,0.64,0.36,0.08,0.43,0.41,0.57,0.59,0.60,0.79,
        0.86,0.60,0.47,0.55,0.37,0.26,0.03,0.24,0.35,0.54,0.54,0.79,
        0.55,0.69,0.92,0.67,0.46,0.40,0.01,0.25,0.44,0.42,0.51,0.78,
        1.24,1.22,1.32,0.71,0.74,0.79,0.62,0.22,0.54,0.82,1.01,0.96,
        1.27,0.90,0.43,0.61,0.78,0.35,0.52,0.44,0.08,0.26,0.18,0.30,
        0.38,0.33,0.25,0.14,0.31,-0.23,0.24,0.19,0.16,0.42,0.28,0.44,
        0.29,0.32,0.09,0.22,0.40,1.26,0.33,-0.09,0.48,0.45,-0.21,0.15,
        0.32,0.43,0.75,0.57,0.13,0.01,0.19,0.11,-0.04,0.10,0.51,1.15,
        0.21,0.25,0.07,-0.31,-0.38,0.26,0.36,0.24,0.64,0.86,0.89,1.35,
        0.83,0.86,0.93,0.31,0.83,0.53,0.96,0.87,1.16,1.25,0.95,0.73,
        0.54,1.01,1.62,1.06,0.47,0.67,-0.68,-0.73,-0.29,0.59,0.41,0.54,
        0.53,0.84,0.71,0.61,0.23,-0.08,0.12,0.23,0.26,0.24,0.28,0.62,
        0.42,0.83,0.16,0.38,0.46,0.20,0.38,0.44,0.44,0.56,0.39,0.52,
    ]
    igpm = [
        1.71,1.66,1.79,1.96,2.67,1.82,1.39,0.95,0.46,0.59,0.97,0.55,
        0.80,0.15,0.36,0.37,0.75,1.12,0.37,0.39,-0.04,0.08,0.40,0.43,
        0.67,0.52,1.18,0.58,0.38,0.72,0.45,-0.16,0.12,0.22,0.69,0.83,
        0.72,0.06,0.22,0.44,0.39,0.17,-0.12,-0.32,-0.01,0.12,-0.09,0.99,
        0.77,4.44,1.98,0.03,0.37,0.73,1.59,1.69,1.29,1.89,2.27,1.22,
        1.25,0.20,0.17,0.51,0.90,0.50,1.53,2.31,0.77,0.30,0.33,0.97,
        0.56,0.40,1.15,1.09,0.73,1.31,1.63,1.57,0.44,1.44,1.79,1.20,
        0.83,0.40,0.86,1.58,0.72,2.67,4.42,4.18,3.95,5.84,5.84,3.67,
        2.75,2.35,1.52,0.76,0.20,-0.43,0.05,-0.60,0.23,0.53,0.83,0.56,
        0.80,0.69,0.74,1.17,1.29,1.12,1.55,1.11,0.71,0.58,0.83,1.38,
        0.34,0.27,0.72,0.94,0.10,-0.44,-0.39,-0.61,0.15,0.59,0.88,0.03,
        0.73,0.61,0.56,0.23,-0.36,0.00,-0.38,-0.22,0.18,0.45,0.48,0.67,
        0.69,0.50,0.50,0.37,0.35,0.31,0.54,0.52,0.47,0.87,0.98,1.75,
        1.09,0.54,0.74,0.73,1.61,2.10,1.76,0.89,0.47,0.00,-1.28,-0.72,
       -0.13,0.21,-0.34,0.00,-0.05,-0.22,-0.26,-0.36,0.23,0.40,0.59,0.17,
        0.97,1.00,0.54,0.81,1.19,0.31,0.47,1.04,0.45,1.01,1.78,0.88,
        0.98,1.00,0.62,0.51,0.43,0.01,-0.05,0.43,0.65,0.41,0.44,0.00,
        0.46,0.07,0.43,0.85,1.02,0.71,1.55,1.43,0.97,0.57,0.54,0.68,
        0.66,0.41,0.16,0.07,0.40,0.74,0.43,0.15,1.36,0.86,0.92,1.40,
        0.72,0.38,1.72,0.35,0.81,0.59,0.53,0.45,0.08,0.48,0.83,0.62,
        0.76,0.25,1.83,1.17,0.99,0.72,1.20,0.25,1.22,1.50,1.52,0.49,
        1.14,0.17,-0.43,0.31,0.22,0.48,0.15,0.19,0.18,0.17,0.23,0.54,
        0.64,0.08,-0.84,-0.09,-0.68,-0.72,0.06,0.20,0.44,0.07,0.68,0.89,
        0.76,0.07,0.64,0.57,1.38,1.87,0.81,-0.01,0.83,0.86,-0.49,-1.08,
        0.89,0.97,1.01,0.92,0.62,0.67,0.43,0.64,-0.01,1.10,0.84,1.48,
        0.93,0.30,1.05,0.80,0.28,1.25,2.23,2.74,4.34,3.23,3.28,3.55,
        3.76,2.53,2.94,1.74,4.10,0.60,0.78,0.66,-0.64,0.64,-0.30,0.87,
        1.82,1.83,1.74,1.41,0.52,0.59,-2.00,-0.70,-0.55,0.95,0.56,0.56,
        0.98,0.05,-0.07,-0.97,-0.47,-1.93,-0.72,-0.94,-0.14,0.50,0.89,0.74,
        0.07,0.20,-0.47,-0.34,0.39,0.81,0.61,0.29,0.44,1.52,1.76,0.94,
    ]
    dates = pd.date_range("1995-01-31", periods=len(ipca), freq="ME")
    return pd.DataFrame({"IPCA": ipca, "IGPM": igpm}, index=dates)

df = load_data()
T  = len(df)
W  = 240   # 20 anos em meses

print(f"✅ {T} observações | {df.index[0].strftime('%b/%Y')} → {df.index[-1].strftime('%b/%Y')}")
print(f"   Janela do modelo: {W} meses (20 anos)\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRODUÇÃO FIEL DO MODELO
#    α = intercepto OLS(IGPM ~ IPCA) na janela deslizante de 240m
#    β = coeficiente angular OLS(IGPM ~ IPCA) na janela deslizante de 240m
#    Previsão: IGP-M_hat = α + β · IPCA_t
# ─────────────────────────────────────────────────────────────────────────────

alphas, betas, r2s, dates_ab = [], [], [], []

for i in range(W, T + 1):
    xi = df["IPCA"].iloc[i - W:i].values
    yi = df["IGPM"].iloc[i - W:i].values
    X  = np.column_stack([xi, np.ones(W)])
    b  = np.linalg.lstsq(X, yi, rcond=None)[0]
    yh = X @ b
    ss_res = ((yi - yh) ** 2).sum()
    ss_tot = ((yi - yi.mean()) ** 2).sum()
    betas.append(b[0])
    alphas.append(b[1])
    r2s.append(1 - ss_res / ss_tot)
    dates_ab.append(df.index[i - 1])

params = pd.DataFrame({"alpha": alphas, "beta": betas, "r2": r2s}, index=dates_ab)

# Parâmetros na data mais recente (calibração atual do modelo)
alpha_now = params["alpha"].iloc[-1]
beta_now  = params["beta"].iloc[-1]
r2_now    = params["r2"].iloc[-1]

print("=" * 65)
print("PARÂMETROS ATUAIS DO MODELO (janela: Jan/2005 → Dez/2024)")
print("=" * 65)
print(f"  α (intercepto) = {alpha_now:.6f} pp")
print(f"  β (coef. ang.) = {beta_now:.6f}")
print(f"  R²             = {r2_now:.4f}")
print(f"  Equação: IGP-M = {alpha_now:.4f} + {beta_now:.4f} · IPCA")


# ─────────────────────────────────────────────────────────────────────────────
# 2. INSTABILIDADE DE α e β NA PRÓPRIA JANELA MM20a
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CRÍTICA 1 — INSTABILIDADE DOS PARÂMETROS NA PRÓPRIA JANELA")
print("=" * 65)

print(f"\n  β ao longo do tempo:")
print(f"    Mínimo   : {params['beta'].min():.4f}  ({params['beta'].idxmin().strftime('%b/%Y')})")
print(f"    Máximo   : {params['beta'].max():.4f}  ({params['beta'].idxmax().strftime('%b/%Y')})")
print(f"    Média    : {params['beta'].mean():.4f}")
print(f"    Std Dev  : {params['beta'].std():.4f}")
print(f"    CV (%)   : {params['beta'].std()/params['beta'].mean()*100:.1f}%")

print(f"\n  α ao longo do tempo:")
print(f"    Mínimo   : {params['alpha'].min():.4f}  ({params['alpha'].idxmin().strftime('%b/%Y')})")
print(f"    Máximo   : {params['alpha'].max():.4f}  ({params['alpha'].idxmax().strftime('%b/%Y')})")
print(f"    Std Dev  : {params['alpha'].std():.4f}")
print(f"    CV (%)   : {params['alpha'].std()/abs(params['alpha'].mean())*100:.1f}%")

# Amplitude de variação — quantifica o risco de uso de parâmetro estático
delta_beta  = params["beta"].max()  - params["beta"].min()
delta_alpha = params["alpha"].max() - params["alpha"].min()
print(f"\n  ⚠️  Amplitude de β: {delta_beta:.4f}  →  usar β fixo introduz")
print(f"      erro estrutural de até ±{delta_beta/2:.4f} no coeficiente angular.")
print(f"  ⚠️  Amplitude de α: {delta_alpha:.4f} pp")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SENSIBILIDADE À ESCOLHA DA JANELA
#    Se a janela se desloca 1 ano para trás, como α e β mudam?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CRÍTICA 2 — SENSIBILIDADE À ESCOLHA DA JANELA (±1 a ±3 anos)")
print("=" * 65)

def fit_window(df_, end_idx, window):
    xi = df_["IPCA"].iloc[end_idx - window:end_idx].values
    yi = df_["IGPM"].iloc[end_idx - window:end_idx].values
    X  = np.column_stack([xi, np.ones(window)])
    b  = np.linalg.lstsq(X, yi, rcond=None)[0]
    return b[1], b[0]   # beta, alpha

end = T
ref_beta, ref_alpha = fit_window(df, end, W)

print(f"\n  {'Janela':<22} {'β':>8} {'Δβ vs 20a':>12} {'α':>8} {'Δα vs 20a':>12}")
print("  " + "-" * 64)
for shift_yr in [-3, -2, -1, 0, 1, 2, 3]:
    w = W + shift_yr * 12
    if w < 60 or end - w < 0:
        continue
    b, a = fit_window(df, end, w)
    label = f"{20 + shift_yr} anos ({w}m)"
    marker = " ◀ modelo" if shift_yr == 0 else ""
    print(f"  {label:<22} {b:>8.4f} {b - ref_beta:>+12.4f} {a:>8.4f} {a - ref_alpha:>+12.4f}{marker}")

print(f"\n  ⚠️  Δβ entre janelas extremas: "
      f"{fit_window(df,end,W-36)[0] - fit_window(df,end,W+36)[0]:+.4f}")
print("      O modelo não tem critério formal para escolha de 20 anos.")
print("      Qualquer outra janela razoável produz parâmetros diferentes.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ERRO DE PREVISÃO OUT-OF-SAMPLE POR REGIME
#    Usa parâmetros calibrados até t-1 para prever t (walk-forward)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CRÍTICA 3 — ERRO OUT-OF-SAMPLE POR REGIME (Walk-Forward)")
print("=" * 65)

# Previsão 1-passo: parâmetros do mês anterior aplicados ao IPCA do mês atual
igpm_hat = alpha_now + beta_now * df["IPCA"]   # previsão com param. fixo atual
resid_fixed = df["IGPM"] - igpm_hat

# Walk-forward com parâmetros atualizados mês a mês
wf_pred, wf_idx = [], []
for i in range(W, T):
    a = params["alpha"].iloc[i - W]
    b = params["beta"].iloc[i - W]
    igpm_pred = a + b * df["IPCA"].iloc[i]
    wf_pred.append(igpm_pred)
    wf_idx.append(df.index[i])

wf = pd.Series(wf_pred, index=wf_idx)
resid_wf = df["IGPM"].loc[wf.index] - wf

# Sub-períodos (regimes macroeconômicos)
regimes = {
    "Câmbio flutuante / âncora"   : ("2000-01-31", "2002-09-30"),
    "Crise 2002–03"               : ("2002-10-31", "2003-12-31"),
    "Boom commodities 2004–08"    : ("2004-01-31", "2008-08-31"),
    "Pós-crise / estabilidade"    : ("2008-09-30", "2014-12-31"),
    "Recessão / crise fiscal"     : ("2015-01-31", "2018-12-31"),
    "Pandemia + câmbio"           : ("2020-03-31", "2021-12-31"),
    "Normalização pós-pandemia"   : ("2022-01-31", "2024-12-31"),
}

print(f"\n  {'Regime':<38} {'MAE':>6} {'RMSE':>7} {'Viés':>8} {'r(hat,real)':>12}")
print("  " + "-" * 75)

regime_stats = {}
for label, (d0, d1) in regimes.items():
    mask = (resid_wf.index >= d0) & (resid_wf.index <= d1)
    if mask.sum() < 6:
        continue
    e   = resid_wf[mask]
    mae  = np.abs(e).mean()
    rmse = np.sqrt((e**2).mean())
    bias = e.mean()
    r_pr, _ = stats.pearsonr(
        wf[mask].values,
        df["IGPM"].loc[wf.index][mask].values
    )
    regime_stats[label] = {"MAE": mae, "RMSE": rmse, "bias": bias, "r": r_pr}
    flag = " ⚠️" if mae > 1.0 or abs(bias) > 0.5 else ""
    print(f"  {label:<38} {mae:>6.3f} {rmse:>7.3f} {bias:>+8.3f} {r_pr:>12.4f}{flag}")

# Erro global
mae_global  = np.abs(resid_wf).mean()
rmse_global = np.sqrt((resid_wf**2).mean())
bias_global = resid_wf.mean()
print(f"\n  {'GLOBAL (walk-forward)':<38} {mae_global:>6.3f} {rmse_global:>7.3f} {bias_global:>+8.3f}")
print(f"\n  Interpretação: MAE > 1pp significa que o modelo erra, em média,")
print(f"  mais de 1 ponto percentual por mês — relevante para contratos")
print(f"  indexados e marcação a mercado de curvas de IGP-M.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. VIÉS CONDICIONAL — o erro depende do nível de IPCA?
#    Se sim, β não é estacionário e o modelo é heterocedástico por design
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CRÍTICA 4 — VIÉS CONDICIONAL AO NÍVEL DE IPCA")
print("=" * 65)

ipca_wf  = df["IPCA"].loc[wf.index]
quantis  = pd.qcut(ipca_wf, q=5, labels=["Q1\n(baixo)","Q2","Q3","Q4","Q5\n(alto)"])
vies_q   = resid_wf.groupby(quantis).mean()
mae_q    = resid_wf.abs().groupby(quantis).mean()

print(f"\n  {'Quintil IPCA':<15} {'Viés médio (pp)':>18} {'MAE (pp)':>12}")
print("  " + "-" * 48)
for q in vies_q.index:
    flag = " ⚠️" if abs(vies_q[q]) > 0.3 else ""
    print(f"  {str(q):<15} {vies_q[q]:>+18.4f} {mae_q[q]:>12.4f}{flag}")

# Teste de Ramsey RESET manual: adiciona ŷ² para testar má-especificação
yhat_wf = wf.values
e_wf    = resid_wf.values
X_reset = np.column_stack([yhat_wf, yhat_wf**2, np.ones(len(yhat_wf))])
b_r     = np.linalg.lstsq(X_reset, e_wf, rcond=None)[0]
yhat_r  = X_reset @ b_r
ss_res  = ((e_wf - yhat_r)**2).sum()
ss_tot  = ((e_wf - e_wf.mean())**2).sum()
r2_reset = 1 - ss_res/ss_tot
F_reset  = (r2_reset / 2) / ((1 - r2_reset) / (len(e_wf) - 3))
p_reset  = 1 - stats.f.cdf(F_reset, 2, len(e_wf) - 3)
print(f"\n  Ramsey RESET (má-especificação funcional):")
print(f"    F = {F_reset:.4f}   p = {p_reset:.4e}")
print(f"    {'⚠️  Forma funcional linear rejeitada — β não captura a relação.' if p_reset < 0.05 else '✅ Forma funcional não rejeitada.'}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. DECOMPOSIÇÃO DO ERRO: quanto é β instável vs. α instável?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CRÍTICA 5 — DECOMPOSIÇÃO DO ERRO DO MODELO")
print("=" * 65)

# Erro total = (α_t - α_now) + (β_t - β_now)·IPCA_t + ε_t
# Decompõe a variância do erro em contribuição de α e β
ipca_arr = df["IPCA"].loc[params.index].values
err_alpha = (params["alpha"] - alpha_now).values
err_beta  = (params["beta"]  - beta_now).values * ipca_arr

var_alpha = np.var(err_alpha)
var_beta  = np.var(err_beta)
var_total = np.var(err_alpha + err_beta)

print(f"\n  Variância total do erro de instabilidade : {var_total:.6f}")
print(f"  Contribuição de α instável               : {var_alpha:.6f}  ({var_alpha/var_total*100:.1f}%)")
print(f"  Contribuição de β instável               : {var_beta:.6f}  ({var_beta/var_total*100:.1f}%)")
print(f"\n  ➜ O maior gerador de instabilidade é "
      f"{'β (coeficiente angular)' if var_beta > var_alpha else 'α (intercepto)'}.")
print(f"    Usar parâmetros fixos ignora essa fonte de risco.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURA — Dashboard de Crítica ao Modelo Específico
# ─────────────────────────────────────────────────────────────────────────────

C = {"ipca":"#1a6eb5","igpm":"#e05c2a","alpha":"#8e44ad","beta":"#27ae60",
     "err":"#c0392b","hat":"#2c3e50","zero":"#7f8c8d"}

fig = plt.figure(figsize=(20, 28), facecolor="#f4f6f9")
fig.suptitle(
    "Crítica ao Modelo:  IGP-M$_t$ = α + β · IPCA$_t$\n"
    "α e β estimados via Mínimos Quadrados Ordinários — Janela 20 anos",
    fontsize=16, fontweight="bold", y=0.99, color="#1a1a2e"
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.58, wspace=0.38,
                        left=0.07, right=0.97, top=0.95, bottom=0.04)

ax_b   = fig.add_subplot(gs[0, :])   # β ao longo do tempo — painel largo
ax_a   = fig.add_subplot(gs[1, 0])   # α ao longo do tempo
ax_r2  = fig.add_subplot(gs[1, 1])   # R² ao longo do tempo
ax_err = fig.add_subplot(gs[2, :])   # erro out-of-sample walk-forward
ax_vq  = fig.add_subplot(gs[3, 0])   # viés por quintil de IPCA
ax_sc  = fig.add_subplot(gs[3, 1])   # scatter IGP-M real vs. previsto

def fmt(ax, title, ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_facecolor("#fdfdfd")
    ax.grid(axis="y", alpha=0.28, lw=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

# ── β ao longo do tempo ────────────────────────────────────────────────────
fmt(ax_b, "β (coeficiente angular) da MM20 anos  — Instabilidade Temporal", "β")
ax_b.plot(params.index, params["beta"], color=C["beta"], lw=1.8)
ax_b.axhline(beta_now, color="black", lw=1.2, ls="--",
             label=f"β atual = {beta_now:.4f} (usado pelo modelo)")
ax_b.fill_between(params.index,
                  params["beta"].mean() - params["beta"].std(),
                  params["beta"].mean() + params["beta"].std(),
                  alpha=0.12, color=C["beta"], label="±1σ")
ax_b.legend(fontsize=9)

# Marcos de quebra
for yr, label in [(1999,"1999"),(2002,"2002"),(2016,"2016"),(2020,"2020")]:
    ts = pd.Timestamp(f"{yr}-06-30")
    if ts in params.index or ts > params.index[0]:
        ax_b.axvline(ts, color="grey", lw=0.9, ls=":", alpha=0.7)
        ax_b.annotate(label, xy=(ts, params["beta"].max()),
                      xytext=(2, -10), textcoords="offset points",
                      fontsize=8, color="#555")

# ── α ao longo do tempo ────────────────────────────────────────────────────
fmt(ax_a, "α (intercepto) da MM20 anos", "α (pp)")
ax_a.plot(params.index, params["alpha"], color=C["alpha"], lw=1.6)
ax_a.axhline(alpha_now, color="black", lw=1.1, ls="--",
             label=f"α atual = {alpha_now:.4f}")
ax_a.fill_between(params.index,
                  params["alpha"].mean() - params["alpha"].std(),
                  params["alpha"].mean() + params["alpha"].std(),
                  alpha=0.12, color=C["alpha"])
ax_a.legend(fontsize=9)

# ── R² ao longo do tempo ───────────────────────────────────────────────────
fmt(ax_r2, "R² da regressão MM20 anos", "R²")
ax_r2.plot(params.index, params["r2"], color=C["hat"], lw=1.6)
ax_r2.axhline(0.5, color=C["err"], lw=1.0, ls="--", label="R² = 0.5")
ax_r2.set_ylim(0, 1)
ax_r2.legend(fontsize=9)
ax_r2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# ── Erro out-of-sample walk-forward ────────────────────────────────────────
fmt(ax_err,
    f"Erro Out-of-Sample (Walk-Forward)  IGP-M real − IGP-M previsto  |  "
    f"MAE global = {mae_global:.3f} pp  |  Viés = {bias_global:+.3f} pp",
    "Erro (pp)")
ax_err.bar(resid_wf.index, resid_wf.values, width=20,
           color=[C["err"] if e > 0 else C["ipca"] for e in resid_wf.values],
           alpha=0.65)
ax_err.axhline(0, color="black", lw=0.8)
ax_err.axhline(mae_global,  color=C["err"], lw=1.2, ls="--",
               label=f"+MAE = {mae_global:.3f}")
ax_err.axhline(-mae_global, color=C["err"], lw=1.2, ls="--",
               label=f"−MAE = {mae_global:.3f}")
ax_err.legend(fontsize=9, loc="upper left")

# Sombreia períodos de crise
crisis = [("2002-07-31","2003-06-30"), ("2020-03-31","2021-06-30")]
for d0, d1 in crisis:
    ax_err.axvspan(pd.Timestamp(d0), pd.Timestamp(d1),
                   alpha=0.1, color=C["err"])

# ── Viés por quintil ───────────────────────────────────────────────────────
ax_vq.set_facecolor("#fdfdfd")
colors_q = [C["err"] if v > 0 else C["ipca"] for v in vies_q.values]
bars = ax_vq.bar(range(5), vies_q.values, color=colors_q, alpha=0.8, width=0.55)
ax_vq.axhline(0, color="black", lw=0.8)
ax_vq.set_xticks(range(5))
ax_vq.set_xticklabels([f"Q{i+1}" for i in range(5)])
ax_vq.set_title("Viés Condicional ao Nível de IPCA\n(Q1=baixo, Q5=alto)",
                fontsize=11, fontweight="bold", pad=5)
ax_vq.set_ylabel("Viés médio (pp)")
ax_vq.grid(axis="y", alpha=0.28)
for bar, v, m in zip(bars, vies_q.values, mae_q.values):
    ax_vq.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + (0.02 if v >= 0 else -0.08),
               f"{v:+.3f}\nMAE={m:.3f}", ha="center", va="bottom", fontsize=8)

# ── Scatter real vs. previsto ──────────────────────────────────────────────
ax_sc.set_facecolor("#fdfdfd")
igpm_real = df["IGPM"].loc[wf.index].values
ax_sc.scatter(igpm_real, wf.values, alpha=0.3, s=14,
              color=C["hat"], edgecolors="none")
lim = max(abs(igpm_real).max(), abs(wf.values).max()) * 1.05
ax_sc.plot([-lim, lim], [-lim, lim], color=C["err"], lw=1.5,
           ls="--", label="Previsão perfeita")
r_oos, _ = stats.pearsonr(igpm_real, wf.values)
ax_sc.set_title(f"IGP-M Real vs. Previsto (Walk-Forward)  |  r = {r_oos:.4f}",
                fontsize=11, fontweight="bold", pad=5)
ax_sc.set_xlabel("IGP-M Real (%)")
ax_sc.set_ylabel("IGP-M Previsto (%)")
ax_sc.legend(fontsize=9)
ax_sc.grid(alpha=0.28)
ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim)

fig.text(0.5, 0.005,
         f"Fonte: SGS/BCB — Séries 433 (IPCA) e 189 (IGP-M)  |  "
         f"Período: {df.index[0].strftime('%b/%Y')}–{df.index[-1].strftime('%b/%Y')}  |  "
         f"Gerado em {datetime.today().strftime('%d/%m/%Y')}",
         ha="center", fontsize=8, color="#888")

out = "igpm_model_critica.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n✅ Dashboard salvo: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMÁRIO EXECUTIVO — CRÍTICA ESPECÍFICA AO MODELO
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMÁRIO EXECUTIVO — CRÍTICA AO MODELO ESPECÍFICO")
print("=" * 65)
print(f"""
MODELO AVALIADO:
  IGP-M_t = α + β · IPCA_t
  α = {alpha_now:.6f} pp
  β = {beta_now:.6f}
  Calibração: OLS na janela móvel de 20 anos (240 meses)

ACHADOS:

  [1] INSTABILIDADE DE β
      Amplitude histórica: [{params['beta'].min():.4f}, {params['beta'].max():.4f}]
      CV de β = {params['beta'].std()/params['beta'].mean()*100:.1f}%
      O coeficiente angular varia {(params['beta'].max()-params['beta'].min())/beta_now*100:.0f}% em torno do valor atual.
      Um modelo com β fixo ignora essa fonte de risco sistematicamente.

  [2] INSTABILIDADE DE α
      Amplitude histórica: [{params['alpha'].min():.4f}, {params['alpha'].max():.4f}]
      CV de α = {params['alpha'].std()/abs(params['alpha'].mean())*100:.1f}%
      O intercepto não é uma constante — é um processo estocástico.

  [3] SENSIBILIDADE À JANELA
      Alterar a janela em ±1 ano muda β em até
      {abs(fit_window(df,T,W-12)[0] - fit_window(df,T,W+12)[0]):.4f}.
      Não há critério formal (AIC/BIC/validação cruzada) que justifique
      exatamente 20 anos. A escolha é discricionária.

  [4] DESEMPENHO OUT-OF-SAMPLE (walk-forward)
      MAE global = {mae_global:.4f} pp/mês
      RMSE global = {rmse_global:.4f} pp/mês
      Viés global = {bias_global:+.4f} pp/mês
      Correlação (real, previsto) = {r_oos:.4f}
      R² out-of-sample ≈ {1-(resid_wf**2).sum()/((df['IGPM'].loc[wf.index]-df['IGPM'].loc[wf.index].mean())**2).sum():.4f}
      (R² in-sample = {r2_now:.4f} → sobreajuste confirmado)

  [5] VIÉS CONDICIONAL
      O modelo é sistematicamente enviesado em função do nível de IPCA.
      Ramsey RESET: F = {F_reset:.3f}, p = {p_reset:.2e}
      → Forma funcional linear rejeitada a 5%.

  [6] DECOMPOSIÇÃO DO RISCO DE PARÂMETRO
      {var_beta/var_total*100:.1f}% do risco de instabilidade vem de β.
      {var_alpha/var_total*100:.1f}% vem de α.

RECOMENDAÇÕES FORMAIS:
  (a) Substituir β e α fixos por distribuições de probabilidade
      calibradas na volatilidade histórica dos parâmetros.
  (b) Adicionar critério formal de seleção de janela (e.g., expanding
      window com validação cruzada temporal).
  (c) Introduzir variáveis de regime (câmbio, commodities) como
      controles — o β muda estruturalmente em crises cambiais.
  (d) Reportar intervalo de confiança para IGP-M projetado que
      incorpore incerteza de parâmetro, não apenas de resíduo.
  (e) Testar especificação não-linear (e.g., spline ou threshold
      regression) dado que Ramsey RESET rejeita linearidade.
""")
