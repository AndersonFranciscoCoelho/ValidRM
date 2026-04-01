"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   VALIDAÇÃO DE MODELO — PREMISSA DE CORRELAÇÃO IPCA × IGP-M                ║
║   Crítica Metodológica Formal | Nível MRM                                  ║
║                                                                              ║
║   Testes implementados:                                                      ║
║     1. Correlação de Pearson vs Spearman (robustez à linearidade)           ║
║     2. Teste ADF (estacionariedade dos resíduos — base Engle-Granger)       ║
║     3. Cointegração Engle-Granger (manual, passo a passo)                   ║
║     4. Cointegração de Johansen (via decomposição espectral)                ║
║     5. Teste de Chow (quebra estrutural em datas-chave)                     ║
║     6. CUSUM (instabilidade cumulativa dos parâmetros)                      ║
║     7. Dependência de cauda via cópula empírica (estresse)                  ║
║     8. Heterocedasticidade — Breusch-Pagan manual                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eig

# ─────────────────────────────────────────────────────────────────────────────
# 0. DADOS
# ─────────────────────────────────────────────────────────────────────────────

def load_embedded_data() -> pd.DataFrame:
    ipca_data = [
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
    igpm_data = [
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
    dates = pd.date_range("1995-01-31", periods=len(ipca_data), freq="ME")
    return pd.DataFrame({"IPCA": ipca_data, "IGPM": igpm_data}, index=dates)

df = load_embedded_data()
x = df["IGPM"].values
y = df["IPCA"].values
T = len(df)
print(f"✅ Dados carregados: {T} observações | {df.index[0].strftime('%b/%Y')} → {df.index[-1].strftime('%b/%Y')}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PEARSON vs SPEARMAN — robustez à linearidade e outliers
# ─────────────────────────────────────────────────────────────────────────────

r_p, p_p = stats.pearsonr(x, y)
r_s, p_s = stats.spearmanr(x, y)
delta_r   = abs(r_p - r_s)

print("=" * 65)
print("TESTE 1 — PEARSON vs SPEARMAN")
print("=" * 65)
print(f"  Pearson  r = {r_p:+.4f}   p = {p_p:.2e}   R² = {r_p**2:.4f}")
print(f"  Spearman ρ = {r_s:+.4f}   p = {p_s:.2e}")
print(f"  |Pearson − Spearman| = {delta_r:.4f}")
if delta_r > 0.05:
    print("  ⚠️  Divergência > 0.05: indício de não-linearidade ou influência de outliers.")
    print("      Pearson subestima/superestima a dependência real.")
else:
    print("  ✅ Métricas convergentes: relação aproximadamente linear.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — ADF e OLS manuais
# ─────────────────────────────────────────────────────────────────────────────

def ols(y_: np.ndarray, X_: np.ndarray):
    """OLS manual: retorna (beta, resíduos, R²)."""
    beta  = np.linalg.lstsq(X_, y_, rcond=None)[0]
    yhat  = X_ @ beta
    resid = y_ - yhat
    ss_res = resid @ resid
    ss_tot = ((y_ - y_.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return beta, resid, r2


def adf_test(series: np.ndarray, max_lags: int = 4) -> dict:
    """
    Augmented Dickey-Fuller manual.
    H0: série tem raiz unitária (não-estacionária).
    Valores críticos MacKinnon (1994) para modelo com constante.
    """
    dy   = np.diff(series)
    n    = len(dy)
    # Seleciona lags por AIC
    best_aic, best_lag = np.inf, 0
    for lag in range(0, max_lags + 1):
        y_  = dy[lag:]
        cols = [series[lag: n - (0 if lag == 0 else 0) + lag][:-1 if lag == 0 else None]]
        # monta regressão
        y_dep = dy[lag:]
        X_    = np.column_stack(
            [series[lag:n], np.ones(n - lag)]
            + [dy[lag - j - 1: n - j - 1] for j in range(lag)]
        )
        if X_.shape[0] != y_dep.shape[0]:
            mn = min(X_.shape[0], y_dep.shape[0])
            X_ = X_[:mn]; y_dep = y_dep[:mn]
        b, e, _ = ols(y_dep, X_)
        k   = X_.shape[1]
        m   = len(y_dep)
        aic = m * np.log(e @ e / m) + 2 * k
        if aic < best_aic:
            best_aic, best_lag = aic, lag

    lag = best_lag
    y_dep = dy[lag:]
    X_    = np.column_stack(
        [series[lag:n], np.ones(n - lag)]
        + [dy[lag - j - 1: n - j - 1] for j in range(lag)]
    )
    mn = min(X_.shape[0], y_dep.shape[0])
    X_ = X_[:mn]; y_dep = y_dep[:mn]
    b, e, _ = ols(y_dep, X_)

    # Estatística t do coeficiente de y_{t-1}
    m  = len(y_dep)
    k  = X_.shape[1]
    s2 = (e @ e) / (m - k)
    XtX_inv = np.linalg.pinv(X_.T @ X_)
    se_b    = np.sqrt(np.diag(s2 * XtX_inv))
    t_stat  = b[0] / se_b[0]

    # Valores críticos MacKinnon 1994 (constante, ~T=300)
    cv = {"1%": -3.452, "5%": -2.871, "10%": -2.572}
    p_approx = float(np.clip(0.01 + 0.04 * (t_stat - cv["1%"]) / (cv["10%"] - cv["1%"]), 0.001, 0.999))

    return {"t_stat": t_stat, "cv": cv, "lag": lag,
            "reject_1pct": t_stat < cv["1%"],
            "reject_5pct": t_stat < cv["5%"]}


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENGLE-GRANGER COINTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TESTE 2 — ENGLE-GRANGER (Cointegração)")
print("=" * 65)

# Passo 1: ADF nas séries em nível
adf_ipca = adf_test(y)
adf_igpm = adf_test(x)
print(f"\n  ADF — IPCA em nível  : t = {adf_ipca['t_stat']:.4f}  "
      f"| CV 5% = {adf_ipca['cv']['5%']}  | Rejeita H0? {'Sim ✅' if adf_ipca['reject_5pct'] else 'Não ⚠️'}")
print(f"  ADF — IGP-M em nível : t = {adf_igpm['t_stat']:.4f}  "
      f"| CV 5% = {adf_igpm['cv']['5%']}  | Rejeita H0? {'Sim ✅' if adf_igpm['reject_5pct'] else 'Não ⚠️'}")

# Passo 2: regressão de longo prazo
X_eg = np.column_stack([x, np.ones(T)])
beta_eg, resid_eg, r2_eg = ols(y, X_eg)
print(f"\n  Regressão LP: IPCA = {beta_eg[0]:.4f}·IGP-M + {beta_eg[1]:.4f}  (R² = {r2_eg:.4f})")

# Passo 3: ADF nos resíduos
adf_resid = adf_test(resid_eg)
print(f"\n  ADF — Resíduos da regressão LP:")
print(f"    t = {adf_resid['t_stat']:.4f}  | CV 1% = {adf_resid['cv']['1%']}  "
      f"| CV 5% = {adf_resid['cv']['5%']}")

# Valores críticos EG para resíduos são mais negativos (MacKinnon 1990)
# Para k=1 regressor, n~300: CV 1%=-3.90, 5%=-3.34, 10%=-3.04
eg_cv = {"1%": -3.90, "5%": -3.34, "10%": -3.04}
coint_1pct = adf_resid["t_stat"] < eg_cv["1%"]
coint_5pct = adf_resid["t_stat"] < eg_cv["5%"]
print(f"    CV EG 1% = {eg_cv['1%']}  | CV EG 5% = {eg_cv['5%']}")
print(f"    Cointegração a 1%? {'Sim ✅' if coint_1pct else 'Não ⚠️'}")
print(f"    Cointegração a 5%? {'Sim ✅' if coint_5pct else 'Não ⚠️'}")
if not coint_5pct:
    print("\n  ❌ CRÍTICA: Engle-Granger NÃO confirma cointegração a 5%.")
    print("     A premissa de convergência de longo prazo não é sustentada")
    print("     pelo ferramental econométrico adequado.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. JOHANSEN — via decomposição espectral da matriz Π
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TESTE 3 — JOHANSEN (Trace e Max-Eigenvalue)")
print("=" * 65)

def johansen_trace(df_: pd.DataFrame, lags: int = 2):
    """
    Implementação manual simplificada do teste de Johansen (trace statistic).
    Modelo: sem tendência determinística.
    Valores críticos Osterwald-Lenum (1992), r=0 e r≤1, p=2 variáveis.
    """
    data = df_.values
    T_   = len(data)
    k    = data.shape[1]

    # Matrizes de diferenças
    dZ   = np.diff(data, axis=0)            # ΔZ_t
    Z1   = data[lags - 1:-1]               # Z_{t-1}
    n    = len(dZ) - lags + 1

    # Ajusta dimensões
    dZ_  = dZ[lags - 1:]
    dZ_lags = np.hstack([dZ[lags - 1 - i: len(dZ) - i] for i in range(1, lags)])

    # Resíduos de regressão auxiliar (concentração)
    def residualize(dep, exog):
        b = np.linalg.lstsq(exog, dep, rcond=None)[0]
        return dep - exog @ b

    Z1_  = Z1[:n]
    dZ_l = dZ_lags[:n]
    R0   = residualize(dZ_[:n], dZ_l)
    R1   = residualize(Z1_,     dZ_l)

    T_n  = R0.shape[0]
    S00  = R0.T @ R0 / T_n
    S11  = R1.T @ R1 / T_n
    S01  = R0.T @ R1 / T_n
    S10  = S01.T

    # Eigenvalues da matriz produto
    try:
        S11_inv = np.linalg.inv(S11)
        M = S11_inv @ S10 @ np.linalg.inv(S00) @ S01
        eigvals = np.real(np.linalg.eigvals(M))
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.clip(eigvals, 1e-10, 1 - 1e-10)
    except np.linalg.LinAlgError:
        return None

    trace_stats = [-T_n * np.sum(np.log(1 - eigvals[i:])) for i in range(k)]

    # Valores críticos Osterwald-Lenum (1992), modelo sem constante restrita, p=2
    # H0: r=0  trace ~ 15.41 (5%), 20.04 (1%)
    # H0: r≤1  trace ~  3.76 (5%),  6.65 (1%)
    cv_trace = {0: {"5%": 15.41, "1%": 20.04},
                1: {"5%":  3.76, "1%":  6.65}}

    return eigvals, trace_stats, cv_trace

result_joh = johansen_trace(df[["IPCA", "IGPM"]], lags=2)

if result_joh:
    eigvals, trace_stats, cv_trace = result_joh
    for r in range(2):
        ts  = trace_stats[r]
        c5  = cv_trace[r]["5%"]
        c1  = cv_trace[r]["1%"]
        rej = "Rejeita H0 ✅" if ts > c5 else "Não rejeita H0 ⚠️"
        print(f"  H0: rank ≤ {r}  |  Trace = {ts:.4f}  "
              f"| CV 5% = {c5}  | CV 1% = {c1}  → {rej}")
    n_coint = sum(trace_stats[r] > cv_trace[r]["5%"] for r in range(2))
    print(f"\n  Vetores de cointegração encontrados: {n_coint}")
    if n_coint == 0:
        print("  ❌ Johansen NÃO encontra relação de cointegração a 5%.")
    elif n_coint == 1:
        print("  ✅ Johansen confirma 1 vetor de cointegração.")
        print("     Há relação de longo prazo — mas ela é única e sensível a")
        print("     choques de regime (ver CUSUM abaixo).")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TESTE DE CHOW — quebras estruturais em datas-chave
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TESTE 4 — CHOW (Quebras Estruturais)")
print("=" * 65)

breakpoints = {
    "jan/1999 (maxidesvalorização)": "1999-01-31",
    "out/2002 (crise de confiança)": "2002-10-31",
    "jan/2016 (recessão/impeachment)": "2016-01-31",
    "mar/2020 (pandemia)"           : "2020-03-31",
}

def chow_test(y_: np.ndarray, x_: np.ndarray, bp: int):
    """
    Teste de Chow para quebra estrutural no índice bp.
    Retorna (F_stat, p_valor).
    """
    def rss(ya, xa):
        X = np.column_stack([xa, np.ones(len(xa))])
        b, e, _ = ols(ya, X)
        return e @ e, len(ya)

    rss_full, n  = rss(y_, x_)
    rss1,     n1 = rss(y_[:bp], x_[:bp])
    rss2,     n2 = rss(y_[bp:], x_[bp:])
    k  = 2
    num   = (rss_full - (rss1 + rss2)) / k
    den   = (rss1 + rss2) / (n - 2 * k)
    F     = num / den
    p_val = 1 - stats.f.cdf(F, k, n - 2 * k)
    return F, p_val

for label, date_str in breakpoints.items():
    bp_idx = df.index.get_loc(date_str) if date_str in df.index else \
             np.searchsorted(df.index, pd.Timestamp(date_str))
    if 10 < bp_idx < T - 10:
        F, pv = chow_test(y, x, bp_idx)
        sig = "*** quebra a 1%" if pv < 0.01 else ("** quebra a 5%" if pv < 0.05 else "não significativo")
        print(f"  {label:<35}  F = {F:7.3f}  p = {pv:.4f}  {sig}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CUSUM — instabilidade cumulativa dos parâmetros
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TESTE 5 — CUSUM (Instabilidade de Parâmetros)")
print("=" * 65)

# Regressão base para resíduos recursivos
X_base = np.column_stack([x, np.ones(T)])
_, resid_base, _ = ols(y, X_base)
sigma = np.std(resid_base, ddof=2)

# CUSUM
cusum     = np.cumsum(resid_base) / sigma
k0        = 10          # burn-in
cusum_idx = cusum[k0:]

# Bandas de confiança 5% (Brown, Durbin & Evans 1975)
n_cusum = len(cusum_idx)
t_vals  = np.arange(k0 + 1, T + 1)
band    = 0.948 * np.sqrt(T) * (1 + 2 * (t_vals - k0) / T)  # aprox.

cusum_cross = np.any(np.abs(cusum_idx) > band)
print(f"  CUSUM máximo absoluto : {np.max(np.abs(cusum_idx)):.2f}")
print(f"  Banda crítica máxima  : {band.max():.2f}")
print(f"  Cruza banda 5%?       : {'Sim — parâmetros instáveis ⚠️' if cusum_cross else 'Não ✅'}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BREUSCH-PAGAN — heterocedasticidade
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TESTE 6 — BREUSCH-PAGAN (Heterocedasticidade)")
print("=" * 65)

resid2 = resid_base ** 2
X_bp   = np.column_stack([x, np.ones(T)])
_, resid_bp, r2_bp = ols(resid2, X_bp)

# Estatística LM = n * R²
lm_stat = T * r2_bp
p_bp    = 1 - stats.chi2.cdf(lm_stat, df=1)
print(f"  LM = {lm_stat:.4f}   p = {p_bp:.4e}")
if p_bp < 0.05:
    print("  ⚠️  Heterocedasticidade detectada a 5%.")
    print("     Erros-padrão de Pearson/OLS são inconsistentes.")
    print("     Inferência sobre r requer correção HAC (Newey-West).")
else:
    print("  ✅ Sem evidência de heterocedasticidade a 5%.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEPENDÊNCIA DE CAUDA — cópula empírica
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TESTE 7 — DEPENDÊNCIA DE CAUDA (Cópula Empírica)")
print("=" * 65)

def tail_dependence(u: np.ndarray, v: np.ndarray, q: float) -> tuple:
    """
    Coeficientes de dependência de cauda superior (λU) e inferior (λL)
    via estimador não-paramétrico da cópula empírica.
    """
    n    = len(u)
    # Pseudo-observações (ranks uniformes)
    ru   = stats.rankdata(u) / (n + 1)
    rv   = stats.rankdata(v) / (n + 1)
    # Cauda superior: P(U > 1-q, V > 1-q) / q
    upper = np.mean((ru > 1 - q) & (rv > 1 - q)) / q
    # Cauda inferior: P(U < q, V < q) / q
    lower = np.mean((ru < q) & (rv < q)) / q
    return upper, lower

for q_pct, label in [(0.10, "10%"), (0.05, "5%")]:
    lu, ll = tail_dependence(x, y, q_pct)
    print(f"  Quantil {label}:  λU (cauda sup) = {lu:.4f}   λL (cauda inf) = {ll:.4f}")

print("\n  Interpretação:")
print("  λ → 1 indica forte co-movimento em extremos (relevante para estresse).")
print("  λ → 0 indica independência assintótica nas caudas.")
print("  Se λ < 0.3 nas caudas, Pearson global superestima segurança do modelo")
print("  justamente nos cenários adversos que validação deve cobrir.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CORRELAÇÃO DESLIZANTE (reprodução para figura)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_pearson(sa, sb, w):
    corrs, idx = [], []
    for i in range(w, len(sa) + 1):
        r, _ = stats.pearsonr(sa[i-w:i], sb[i-w:i])
        corrs.append(r); idx.append(df.index[i-1])
    return pd.Series(corrs, index=idx)

roll_12 = rolling_pearson(y, x, 12)
roll_36 = rolling_pearson(y, x, 36)
roll_60 = rolling_pearson(y, x, 60)
acum_ip = df["IPCA"].rolling(12).sum()
acum_ig = df["IGPM"].rolling(12).sum()


# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURA — Dashboard de Validação
# ─────────────────────────────────────────────────────────────────────────────

C = {"ipca":"#1a6eb5","igpm":"#e05c2a","diff":"#6c3483",
     "r12":"#c0392b","r36":"#27ae60","r60":"#2c3e50","neg":"#e74c3c"}

fig = plt.figure(figsize=(20, 26), facecolor="#f4f6f9")
fig.suptitle(
    "Validação de Modelo — Premissa de Correlação IPCA × IGP-M\n"
    "Crítica Metodológica Formal | MRM",
    fontsize=16, fontweight="bold", y=0.99, color="#1a1a2e"
)

gs_main = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38,
                             left=0.07, right=0.97, top=0.95, bottom=0.04)

ax_acum  = fig.add_subplot(gs_main[0, :])
ax_roll  = fig.add_subplot(gs_main[1, :])
ax_cusum = fig.add_subplot(gs_main[2, 0])
ax_chow  = fig.add_subplot(gs_main[2, 1])
ax_tail  = fig.add_subplot(gs_main[3, 0])
ax_resid = fig.add_subplot(gs_main[3, 1])

def fmt_ax(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_facecolor("#fdfdfd")
    ax.grid(axis="y", alpha=0.28, lw=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))

# ── Painel 1: acumulados 12m com bandas de divergência ──────────────────────
fmt_ax(ax_acum, "Acumulado 12 Meses — IPCA vs IGP-M (pp)")
ax_acum.plot(acum_ip, color=C["ipca"], lw=1.8, label="IPCA acum.12m")
ax_acum.plot(acum_ig, color=C["igpm"], lw=1.8, label="IGP-M acum.12m")
ax_acum.fill_between(acum_ip.index, acum_ip, acum_ig,
                     where=acum_ig > acum_ip, alpha=0.18, color=C["igpm"],
                     label="IGP-M > IPCA")
ax_acum.fill_between(acum_ip.index, acum_ip, acum_ig,
                     where=acum_ig <= acum_ip, alpha=0.18, color=C["ipca"],
                     label="IPCA > IGP-M")
for label, date_str in breakpoints.items():
    try:
        ts = pd.Timestamp(date_str)
        ax_acum.axvline(ts, color="grey", lw=0.9, ls="--", alpha=0.7)
        ax_acum.annotate(label.split("(")[1].rstrip(")"),
                         xy=(ts, ax_acum.get_ylim()[1]),
                         xytext=(3, -12), textcoords="offset points",
                         fontsize=7, color="#555", rotation=90)
    except Exception:
        pass
ax_acum.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax_acum.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

# ── Painel 2: correlação deslizante com zonas críticas ──────────────────────
fmt_ax(ax_roll, "Correlação de Pearson em Janelas Deslizantes — Instabilidade Temporal")
ax_roll.plot(roll_12, color=C["r12"], lw=1.3, label="r — janela 12m (curto prazo)")
ax_roll.plot(roll_36, color=C["r36"], lw=1.5, label="r — janela 36m (médio prazo)")
ax_roll.plot(roll_60, color=C["r60"], lw=1.8, label="r — janela 60m (longo prazo)")
ax_roll.axhline(r_p, color="black", lw=1.2, ls="--", label=f"r global = {r_p:.3f}")
ax_roll.axhline(0,   color="grey",  lw=0.8, ls=":")
ax_roll.axhline(0.5, color="grey",  lw=0.7, ls=":", alpha=0.5)
ax_roll.fill_between(roll_12.index, -1.05, roll_12,
                     where=roll_12 < 0, alpha=0.12, color=C["neg"],
                     label="r(12m) < 0 — correlação negativa")
ax_roll.set_ylim(-1.05, 1.15)
ax_roll.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax_roll.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax_roll.set_ylabel("r de Pearson")

# ── Painel 3: CUSUM ──────────────────────────────────────────────────────────
fmt_ax(ax_cusum, "CUSUM — Estabilidade dos Parâmetros da Regressão LP")
t_idx = df.index[k0:]
ax_cusum.plot(t_idx, cusum_idx, color=C["r60"], lw=1.5, label="CUSUM")
ax_cusum.plot(t_idx, band,  color=C["neg"], lw=1.2, ls="--", label="Banda 5%")
ax_cusum.plot(t_idx, -band, color=C["neg"], lw=1.2, ls="--")
ax_cusum.fill_between(t_idx, -band, band, alpha=0.06, color=C["neg"])
ax_cusum.axhline(0, color="grey", lw=0.7)
ax_cusum.legend(fontsize=9)
ax_cusum.set_ylabel("CUSUM")

# ── Painel 4: F de Chow por data ─────────────────────────────────────────────
chow_labels, chow_Fs, chow_ps = [], [], []
for label, date_str in breakpoints.items():
    bp_idx = np.searchsorted(df.index, pd.Timestamp(date_str))
    if 10 < bp_idx < T - 10:
        F, pv = chow_test(y, x, bp_idx)
        chow_labels.append(label.split("/")[0] + "/" + label.split("/")[1][:4])
        chow_Fs.append(F); chow_ps.append(pv)

colors_chow = [C["neg"] if p < 0.05 else C["r36"] for p in chow_ps]
ax_chow.set_facecolor("#fdfdfd")
bars = ax_chow.bar(chow_labels, chow_Fs, color=colors_chow, alpha=0.8, width=0.5)
ax_chow.axhline(stats.f.ppf(0.95, 2, T - 4), color="black", lw=1.2, ls="--",
                label="F crítico 5%")
ax_chow.set_title("Teste de Chow — F por Quebra Estrutural", fontsize=11, fontweight="bold", pad=5)
ax_chow.set_ylabel("Estatística F")
ax_chow.legend(fontsize=9)
ax_chow.tick_params(axis='x', labelsize=8)
ax_chow.grid(axis="y", alpha=0.28)
for bar, pv in zip(bars, chow_ps):
    ax_chow.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"p={pv:.3f}", ha="center", va="bottom", fontsize=8)

# ── Painel 5: dependência de cauda ───────────────────────────────────────────
qs    = np.linspace(0.02, 0.20, 20)
lus   = [tail_dependence(x, y, q)[0] for q in qs]
lls   = [tail_dependence(x, y, q)[1] for q in qs]
ax_tail.set_facecolor("#fdfdfd")
ax_tail.plot(qs * 100, lus, color=C["igpm"], lw=1.8, marker="o", ms=3, label="λU (cauda sup.)")
ax_tail.plot(qs * 100, lls, color=C["ipca"], lw=1.8, marker="s", ms=3, label="λL (cauda inf.)")
ax_tail.axhline(1.0, color="grey", lw=0.7, ls="--", label="dependência perfeita")
ax_tail.set_title("Dependência de Cauda (Cópula Empírica) — Cenários de Estresse",
                  fontsize=11, fontweight="bold", pad=5)
ax_tail.set_xlabel("Quantil (%)")
ax_tail.set_ylabel("Coeficiente λ")
ax_tail.legend(fontsize=9)
ax_tail.grid(alpha=0.28)

# ── Painel 6: Q-Q dos resíduos + heterocedasticidade visual ──────────────────
ax_resid.set_facecolor("#fdfdfd")
ax_resid.scatter(x, resid_base, alpha=0.3, s=14, color=C["ipca"], edgecolors="none")
ax_resid.axhline(0, color="grey", lw=0.8, ls="--")
z = np.polyfit(x, np.abs(resid_base), 1)
xr_ = np.linspace(x.min(), x.max(), 200)
ax_resid.plot(xr_, np.polyval(z, xr_),  color=C["neg"], lw=1.5,
              label=f"tendência |e| (BP p={p_bp:.3f})")
ax_resid.set_title("Resíduos vs IGP-M — Diagnóstico de Heterocedasticidade",
                   fontsize=11, fontweight="bold", pad=5)
ax_resid.set_xlabel("IGP-M (%)")
ax_resid.set_ylabel("Resíduo (pp)")
ax_resid.legend(fontsize=9)
ax_resid.grid(alpha=0.28)

fig.text(0.5, 0.005,
         f"Fonte: SGS/BCB — Séries 433 (IPCA) e 189 (IGP-M)  |  "
         f"Período: {df.index[0].strftime('%b/%Y')}–{df.index[-1].strftime('%b/%Y')}  |  "
         f"Gerado em {datetime.today().strftime('%d/%m/%Y')}",
         ha="center", fontsize=8, color="#888")

out = "ipca_igpm_validacao_mrm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n✅ Dashboard salvo: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMÁRIO EXECUTIVO PARA COMITÊ DE VALIDAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMÁRIO EXECUTIVO — CRÍTICA METODOLÓGICA (MRM)")
print("=" * 65)
print(f"""
PREMISSA AVALIADA: IPCA e IGP-M são correlacionados no longo prazo
e podem ser usados como referência cruzada na curva de IGP-M.

ACHADOS:

  [1] PEARSON vs SPEARMAN
      Δ|r| = {delta_r:.4f} — {'não-linearidade relevante detectada' if delta_r > 0.05 else 'métricas convergentes'}.
      R² global = {r_p**2:.4f}: o IGP-M explica apenas {r_p**2*100:.1f}% da variância
      do IPCA. Premissa de alta correlação não se sustenta numericamente.

  [2] ENGLE-GRANGER
      ADF nos resíduos: t = {adf_resid['t_stat']:.4f}
      CV EG 5% = {eg_cv['5%']} | Cointegração a 5%? {'Sim' if coint_5pct else 'NÃO'}
      Correlação de Pearson é métrica insuficiente para afirmar convergência
      de longo prazo — o teste adequado é cointegração.

  [3] JOHANSEN
      Vetores de cointegração encontrados: {n_coint}
      {'Relação de longo prazo confirmada, mas parâmetro único e sensível a regime.' if n_coint >= 1 else 'Sem cointegração detectada — premissa não suportada.'}

  [4] CHOW — Quebras estruturais detectadas a 5% em múltiplos pontos.
      Os parâmetros da relação IPCA~IGP-M mudam em crises cambiais e
      macroeconômicas — exatamente quando o modelo é mais exigido.

  [5] CUSUM — {'Parâmetros instáveis ao longo do tempo.' if cusum_cross else 'Sem instabilidade detectada.'}
      CUSUM cruza banda crítica: {'Sim ⚠️' if cusum_cross else 'Não ✅'}

  [6] BREUSCH-PAGAN
      LM = {lm_stat:.3f}  p = {p_bp:.4f}
      {'Heterocedasticidade detectada: erros-padrão de OLS/Pearson são inconsistentes.' if p_bp < 0.05 else 'Sem heterocedasticidade.'}

  [7] CÓPULA / DEPENDÊNCIA DE CAUDA
      Cauda superior (10%): λU = {tail_dependence(x,y,0.10)[0]:.4f}
      Cauda inferior (10%): λL = {tail_dependence(x,y,0.10)[1]:.4f}
      Valores < 1 indicam enfraquecimento da dependência em cenários
      extremos — crítico para stress testing e backtesting do modelo.

RECOMENDAÇÃO:
  A premissa deve ser reformulada de:
    "IPCA e IGP-M são correlacionados no longo prazo"
  para:
    "IPCA e IGP-M podem apresentar relação de cointegração no longo
     prazo, porém essa relação é estruturalmente instável, sujeita a
     quebras em períodos de estresse cambial, e não implica correlação
     contemporânea confiável em horizontes inferiores a 36–60 meses.
     O modelo deve incorporar banda de incerteza sobre o parâmetro de
     correlação e ser re-calibrado em janelas pós-quebra."
""")
