# -*- coding: utf-8 -*-
"""
Coril SAB — Optimizador de Portafolios Black-Litterman
Interfaz comercial (Streamlit) — MOTOR INTEGRADO v3

Cambios v3:
  - Múltiples benchmarks (dinámicos, se actualizan al redescargar)
  - Monto de inversión inicial configurable (slider sidebar, hasta 1M)
  - VaR / CVaR paramétrico e histórico
  - Betas por regresión OLS
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer import (
    RiskProfile, ForcedAsset, View, BLConfig, run_profile,
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(page_title="Coril · Optimizador BL", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

RF_ANNUAL   = 0.02
PPY         = 52

FICO_TICKER = "FICCMP13"
FICO = ForcedAsset(ret_annual=0.065, vol_annual=0.010, beta=0.30,
                   sector="Factoring", region="Perú", moneda="USD",
                   instrumento="Fondo de inversión")

PERFILES = {
    "Perfil A (30/70)": (0.30, 0.70),
    "Perfil B (40/60)": (0.40, 0.60),
    "Perfil C (50/50)": (0.50, 0.50),
    "Perfil D (60/40)": (0.60, 0.40),
    "Perfil E (70/30)": (0.70, 0.30),
}

COL_RV, COL_RF, COL_BMK, COL_OPT = "#2E5E8C", "#2CA02C", "#888888", "#D6604D"
BMK_COLORS = ["#888888", "#E377C2", "#FF7F0E", "#9467BD", "#17BECF"]

# =============================================================================
# ESTADO
# =============================================================================
def init_state():
    defaults = {
        "tickers": [], "benchmarks": ["^GSPC"], "views": [],
        "optimized": False, "result": None, "manual_weights": None,
        "returns": None, "bench_rets": None, "betas": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

# =============================================================================
# DATOS
# =============================================================================
@st.cache_data(show_spinner=False)
def download_equity(tickers, period="5y", interval="1wk"):
    """Descarga precios equity y retorna log-returns."""
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].copy()
        px.columns = list(tickers)[:1]
    px = px.dropna(how="all").ffill()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")


@st.cache_data(show_spinner=False)
def download_benchmarks(bench_tickers, period="5y", interval="1wk"):
    """Descarga uno o más benchmarks, retorna dict de {ticker: Series}."""
    import yfinance as yf
    result = {}
    for bk in bench_tickers:
        bk = bk.strip().upper()
        if not bk:
            continue
        try:
            raw = yf.download(bk, period=period, interval=interval,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            pb = raw["Close"]
            if isinstance(pb, pd.DataFrame):
                pb = pb.iloc[:, 0]
            pb.index = pd.to_datetime(pb.index).tz_localize(None)
            lr = np.log(pb / pb.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
            lr.name = bk
            result[bk] = lr
        except Exception:
            pass
    return result


def calc_betas(returns, bench_ret):
    """Betas por regresión OLS contra el primer benchmark."""
    common = returns.index.intersection(bench_ret.index)
    bmk = bench_ret.loc[common].values
    bmk_var = np.var(bmk, ddof=1)
    betas = {}
    for tk in returns.columns:
        tk_vals = returns.loc[common, tk].values
        mask = np.isfinite(tk_vals) & np.isfinite(bmk)
        if mask.sum() > 10 and bmk_var > 1e-12:
            cov = np.cov(tk_vals[mask], bmk[mask], ddof=1)[0, 1]
            b = cov / bmk_var
            betas[tk] = round(float(b), 3) if np.isfinite(b) and b > 0 else 1.0
        else:
            betas[tk] = 1.0
    return pd.Series(betas)


def optimize(tickers, views_cfg, equity_target, fico_target, primary_bench):
    returns   = st.session_state.returns
    betas     = st.session_state.betas.copy()
    betas[FICO_TICKER] = FICO.beta

    views = []
    for v in views_cfg:
        if v["type"] == "absolute":
            views.append(View(kind="absolute", asset=v["asset"],
                              q=v["q"], confidence=v["confidence"]))
        else:
            views.append(View(kind="relative", long=v["long"], short=v["short"],
                              q=v["q"], confidence=v["confidence"]))

    profile = RiskProfile.for_split(equity_target, fico_target)
    config  = BLConfig(rf_annual=RF_ANNUAL, periods_per_year=PPY, tau=0.05,
                       max_weight_equity=0.25, gamma_beta=5.0)

    ok_tickers = [t for t in tickers if t in returns.columns]
    return run_profile(
        returns=returns, equity_assets=ok_tickers, forced_assets={FICO_TICKER: FICO},
        profile=profile, views=views, config=config,
        benchmark_returns=primary_bench, betas=betas,
    )


def wealth_and_dd(weights, returns, bench_dict, capital):
    """Retornos portafolio + wealth index para portafolio y todos los benchmarks."""
    if not bench_dict or not isinstance(bench_dict, dict):
        bench_dict = {}

    eq_cols = [a for a in weights.index if a in returns.columns and a != FICO_TICKER]
    w_eq = weights.reindex(eq_cols).fillna(0.0)

    port_ret = pd.Series(0.0, index=returns.index)
    for col in eq_cols:
        port_ret = port_ret + w_eq[col] * returns[col].fillna(0.0)
    if FICO_TICKER in weights.index and weights[FICO_TICKER] > 1e-8:
        port_ret = port_ret + weights[FICO_TICKER] * (np.log(1 + FICO.ret_annual) / PPY)
    port_ret = port_ret.fillna(0.0)

    # Alinear con TODOS los benchmarks
    common = port_ret.index
    for bk_ret in bench_dict.values():
        common = common.intersection(bk_ret.index)
    port_ret = port_ret.loc[common]

    wealth = np.exp(port_ret.cumsum()) * capital
    dd     = wealth / wealth.cummax() - 1.0

    bench_wealths = {}
    bench_dds     = {}
    for name, bk_ret in bench_dict.items():
        br = bk_ret.loc[common].fillna(0.0)
        bw = np.exp(br.cumsum()) * capital
        bench_wealths[name] = bw
        bench_dds[name]     = bw / bw.cummax() - 1.0

    return port_ret, wealth, dd, bench_wealths, bench_dds


def calc_var_cvar(returns_series, confidence=0.95):
    clean = returns_series.dropna()
    if len(clean) < 10:
        return {"VaR_hist": np.nan, "CVaR_hist": np.nan,
                "VaR_param": np.nan, "CVaR_param": np.nan}
    from scipy.stats import norm
    mu  = clean.mean() * PPY
    sig = clean.std(ddof=1) * np.sqrt(PPY)
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    var_param  = -(mu + z * sig)
    cvar_param = -(mu - sig * norm.pdf(z) / alpha)
    ann_rets = clean * PPY
    var_hist  = -float(np.percentile(ann_rets, alpha * 100))
    cvar_hist = -float(ann_rets[ann_rets <= -var_hist].mean()) if (ann_rets <= -var_hist).any() else var_hist
    return {"VaR_hist": var_hist, "CVaR_hist": cvar_hist,
            "VaR_param": var_param, "CVaR_param": cvar_param}


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("Coril · BL")
    st.caption("Optimizador institucional")
    st.divider()

    perfil_sel = st.selectbox("Perfil de inversión", list(PERFILES.keys()), index=2)
    eq_t, fi_t = PERFILES[perfil_sel]
    _prof = RiskProfile.for_split(eq_t, fi_t)

    c1, c2 = st.columns(2)
    c1.metric("Renta variable", f"{eq_t:.0%}")
    c2.metric("Renta fija", f"{fi_t:.0%}")

    st.divider()
    st.caption(f"Perfil base: **{_prof.base.value}**")
    st.caption(f"Beta objetivo · {0.5*(_prof.beta_min+_prof.beta_max):.2f} "
               f"(rango {_prof.beta_min:.2f}–{_prof.beta_max:.2f})")
    st.caption(f"Máx. drawdown · {_prof.max_drawdown:.0%}")

    st.divider()
    st.caption(f"**Renta fija forzada:** {FICO_TICKER}")
    st.caption(f"Retorno {FICO.ret_annual:.2%} · Vol {FICO.vol_annual:.2%}")

    st.divider()
    capital_inicial = st.slider(
        "Monto de inversión (USD)", min_value=1_000, max_value=1_000_000,
        value=100_000, step=1_000, format="$%d",
    )

    st.divider()
    st.subheader("Benchmarks")

    # Form garantiza que el text_input se lee al pulsar Añadir
    with st.form("add_bk_form", clear_on_submit=True):
        bk_new = st.text_input("Ticker", placeholder="Ej: SPY, QQQ, EEM…")
        bk_submit = st.form_submit_button("Añadir benchmark", use_container_width=True)
        if bk_submit and bk_new.strip():
            bk = bk_new.strip().upper()
            if bk not in st.session_state.benchmarks:
                st.session_state.benchmarks.append(bk)

    # Lista actual con botón de quitar
    if st.session_state.benchmarks:
        for i, bk in enumerate(st.session_state.benchmarks):
            bc1, bc2 = st.columns([5, 1])
            bc1.caption(f"**{bk}**")
            if bc2.button("✕", key=f"rmbk_{i}"):
                st.session_state.benchmarks.pop(i)
                st.rerun()
    else:
        st.caption("Añade al menos un benchmark.")

    # Variable auxiliar para usar en el resto de la app
    benchmarks_list = list(st.session_state.benchmarks)

# =============================================================================
# CUERPO
# =============================================================================
st.title("Optimizador Black-Litterman")
st.caption(f"Mandato: **{perfil_sel}** · {eq_t:.0%} RV / {fi_t:.0%} RF "
           f" · Inversión: ${capital_inicial:,.0f}")

tab1, tab2, tab3 = st.tabs(["1 · Activos", "2 · Views", "3 · Optimización"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Universo de renta variable")
    c_in, c_add = st.columns([4, 1])
    nt = c_in.text_input("Ticker", placeholder="AAPL, MSFT…",
                         label_visibility="collapsed", key="in_tk")
    if c_add.button("Añadir", use_container_width=True):
        t = nt.strip().upper()
        if t and t not in st.session_state.tickers:
            st.session_state.tickers.append(t)
            st.rerun()

    if st.session_state.tickers:
        for i, t in enumerate(st.session_state.tickers):
            a, b = st.columns([6, 1])
            a.write(f"• {t}")
            if b.button("Quitar", key=f"rm{i}", use_container_width=True):
                st.session_state.tickers.pop(i)
                st.rerun()
    else:
        st.info("Añade al menos un ticker de renta variable.")

    st.divider()
    st.caption(f"**Descargará:** {len(st.session_state.tickers)} activos · "
               f"{len(benchmarks_list)} benchmark(s): {', '.join(benchmarks_list) if benchmarks_list else '—'}")

    # Detectar si benchmarks cambiaron desde última descarga
    if (isinstance(st.session_state.bench_rets, dict)
            and set(st.session_state.bench_rets.keys()) != set(benchmarks_list)):
        st.warning("Los benchmarks cambiaron. Pulsa Descargar datos para actualizar.")

    if st.button("Descargar datos",
                 disabled=(not st.session_state.tickers or not benchmarks_list),
                 type="primary"):
        with st.spinner("Descargando precios…"):
            log_ret = download_equity(tuple(st.session_state.tickers))
        if log_ret is None or log_ret.empty:
            st.error("No se pudieron descargar precios. Verifica los tickers.")
        else:
            with st.spinner("Descargando benchmarks…"):
                bench_dict = download_benchmarks(tuple(benchmarks_list))

            if not bench_dict:
                st.error("No se pudieron descargar los benchmarks.")
            else:
                # Alinear todo a fechas comunes
                common = log_ret.index
                for bret in bench_dict.values():
                    common = common.intersection(bret.index)
                st.session_state.returns   = log_ret.loc[common]
                st.session_state.bench_rets = {k: v.loc[common] for k, v in bench_dict.items()}

                # Betas contra el primer benchmark
                primary = list(bench_dict.values())[0]
                betas = calc_betas(log_ret.loc[common], primary.loc[common])
                st.session_state.betas = betas

                ok = [t for t in st.session_state.tickers if t in log_ret.columns]
                st.success(f"{len(common)} semanas × {len(ok)} activos · "
                           f"{len(bench_dict)} benchmark(s).")
                falt = set(st.session_state.tickers) - set(ok)
                if falt:
                    st.warning(f"Sin datos (ignorados): {falt}")

                st.write("**Betas (regresión OLS vs "
                         f"{list(bench_dict.keys())[0]}):**")
                st.dataframe(betas.rename("Beta").to_frame().T.style.format("{:.3f}"),
                             use_container_width=True)

                # Resetear optimización al redescargar
                st.session_state.optimized = False
                st.session_state.result = None

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Views del analista")
    if not st.session_state.tickers:
        st.info("Primero añade tickers.")
    else:
        vt = st.radio("Tipo", ["Absoluta", "Relativa"], horizontal=True)
        if vt == "Absoluta":
            c1, c2, c3 = st.columns([3, 2, 2])
            a = c1.selectbox("Activo", st.session_state.tickers, key="a_abs")
            q = c2.number_input("Retorno anual", value=0.10, step=0.01,
                               format="%.2f", key="q_abs")
            cf = c3.slider("Confianza", 0.1, 1.0, 0.5, 0.1, key="c_abs")
            if st.button("Añadir view absoluta"):
                st.session_state.views.append({"type": "absolute", "asset": a,
                                               "q": float(q), "confidence": float(cf)})
                st.rerun()
        else:
            c1, c2, c3, c4 = st.columns(4)
            lg = c1.selectbox("Sobrepondera", st.session_state.tickers, key="l_rel")
            sh = c2.selectbox("Subpondera", st.session_state.tickers, key="s_rel")
            q  = c3.number_input("Diferencial", value=0.05, step=0.01,
                                format="%.2f", key="q_rel")
            cf = c4.slider("Confianza", 0.1, 1.0, 0.5, 0.1, key="c_rel")
            if st.button("Añadir view relativa"):
                if lg == sh:
                    st.warning("Deben ser activos distintos.")
                else:
                    st.session_state.views.append({"type": "relative", "long": lg,
                                                   "short": sh, "q": float(q),
                                                   "confidence": float(cf)})
                    st.rerun()

        st.divider()
        if st.session_state.views:
            for i, v in enumerate(st.session_state.views):
                a, b = st.columns([6, 1])
                if v["type"] == "absolute":
                    a.write(f"• **{v['asset']}** → {v['q']:.2%} (conf. {v['confidence']:.0%})")
                else:
                    a.write(f"• **{v['long']}** > **{v['short']}** por {v['q']:.2%} "
                            f"(conf. {v['confidence']:.0%})")
                if b.button("Quitar", key=f"rmv{i}", use_container_width=True):
                    st.session_state.views.pop(i)
                    st.rerun()
        else:
            st.info("Sin views: se usa solo el equilibrio de mercado.")

# ── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Optimización")
    ready = (st.session_state.returns is not None
             and isinstance(st.session_state.bench_rets, dict)
             and len(st.session_state.bench_rets) > 0)
    if not ready:
        st.info("Descarga los datos en la pestaña 1 antes de optimizar.")

    if st.button("Optimizar portafolio", type="primary", disabled=not ready,
                 use_container_width=True):
        primary_bench = list(st.session_state.bench_rets.values())[0]
        with st.spinner("Optimizando…"):
            res = optimize(st.session_state.tickers, st.session_state.views,
                           eq_t, fi_t, primary_bench)
        st.session_state.result = res
        st.session_state.manual_weights = res.weights.copy()
        st.session_state.optimized = True
        if res.feasible:
            st.success("Optimización completada.")
        else:
            st.warning(f"Solución con advertencias: {res.feasibility_report}")

    if (st.session_state.optimized and st.session_state.result is not None
            and isinstance(st.session_state.bench_rets, dict)):
        res = st.session_state.result

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retorno esperado (BL)", f"{res.exp_return:.2%}")
        m2.metric("Volatilidad anual", f"{res.volatility:.2%}")
        m3.metric("Sharpe", f"{res.sharpe:.3f}")
        m4.metric("Beta portafolio", f"{res.beta:.3f}")

        st.divider()
        st.subheader("Ajuste manual de pesos")
        st.caption("Mueve los pesos; se renormalizan a 100%.")

        cs, csum = st.columns([3, 2])
        with cs:
            nuevos = {}
            for a in res.weights.index:
                es_fico = a == FICO_TICKER
                nuevos[a] = st.slider(f"{a} · {'RF' if es_fico else 'RV'}",
                                      0.0, 1.0, float(res.weights[a]), 0.01,
                                      key=f"sl_{a}")
            wn = pd.Series(nuevos)
            total = wn.sum()
            wnorm = wn / total if total > 0 else wn
            st.session_state.manual_weights = wnorm

        with csum:
            eqw = float(wnorm[[a for a in wnorm.index if a != FICO_TICKER]].sum())
            fiw = float(wnorm.get(FICO_TICKER, 0.0))
            st.metric("RV", f"{eqw:.1%}", delta=f"{eqw-eq_t:+.1%} vs target")
            st.metric("RF", f"{fiw:.1%}", delta=f"{fiw-fi_t:+.1%} vs target")
            if abs(eqw - eq_t) > 0.05:
                st.warning(f"Split desviado {abs(eqw-eq_t):.1%} del mandato.")

        # ── Gráficos ─────────────────────────────────────────────────────────
        st.divider()
        g1, g2 = st.columns(2)

        with g1:
            st.caption("Composición del portafolio")
            wshow = wnorm[wnorm > 1e-4]
            colors = [COL_RF if a == FICO_TICKER else COL_RV for a in wshow.index]
            fig = go.Figure(go.Bar(x=wshow.values, y=wshow.index, orientation="h",
                                   marker_color=colors))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                             xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.caption("Exposición por clase")
            fig = go.Figure(go.Pie(labels=["Renta variable", "Renta fija"],
                                   values=[eqw, fiw],
                                   marker_colors=[COL_RV, COL_RF], hole=0.5))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # ── Wealth index + drawdown (multi-benchmark) ────────────────────────
        st.caption(f"Evolución de capital (base ${capital_inicial:,.0f}) y drawdown")
        pr, wealth, dd, bwealths, bdds = wealth_and_dd(
            wnorm, st.session_state.returns, st.session_state.bench_rets,
            capital_inicial)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.7, 0.3], vertical_spacing=0.05)
        # Portafolio
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, name="Portafolio",
                                line=dict(color=COL_RV, width=2)), row=1, col=1)
        # Benchmarks
        for idx, (bname, bw) in enumerate(bwealths.items()):
            color = BMK_COLORS[idx % len(BMK_COLORS)]
            fig.add_trace(go.Scatter(x=bw.index, y=bw.values, name=bname,
                                    line=dict(color=color, dash="dash")), row=1, col=1)

        # Drawdown portafolio
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="DD portafolio",
                                fill="tozeroy", line=dict(color=COL_OPT)), row=2, col=1)
        # Drawdown benchmarks
        for idx, (bname, bdd) in enumerate(bdds.items()):
            color = BMK_COLORS[idx % len(BMK_COLORS)]
            fig.add_trace(go.Scatter(x=bdd.index, y=bdd.values, name=f"DD {bname}",
                                    line=dict(color=color, dash="dot", width=1)),
                         row=2, col=1)
        fig.add_hline(y=-res.profile.max_drawdown, line_dash="dot",
                     line_color="black", row=2, col=1,
                     annotation_text=f"Límite {res.profile.max_drawdown:.0%}")
        fig.update_yaxes(tickformat=".0%", row=2, col=1)
        fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0),
                         legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

        # ── Métricas ex-post + VaR/CVaR ──────────────────────────────────────
        ann_ret = np.exp(pr.mean() * PPY) - 1
        ann_vol = pr.std(ddof=1) * np.sqrt(PPY)
        risk = calc_var_cvar(pr, confidence=0.95)

        st.divider()
        st.subheader("Métricas históricas (ex-post)")
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Retorno anualizado", f"{ann_ret:.2%}")
        h2.metric("Volatilidad anual", f"{ann_vol:.2%}")
        h3.metric("Máx. drawdown", f"{dd.min():.2%}")
        h4.metric("VaR 95%", f"{risk['VaR_param']:.2%}")
        h5.metric("CVaR 95%", f"{risk['CVaR_param']:.2%}")

        with st.expander("Detalle VaR / CVaR"):
            st.write("Valores anualizados al 95% de confianza.")
            risk_df = pd.DataFrame({
                "Paramétrico (Normal)": [risk["VaR_param"], risk["CVaR_param"]],
                "Histórico": [risk["VaR_hist"], risk["CVaR_hist"]],
            }, index=["VaR 95%", "CVaR 95%"])
            st.dataframe(risk_df.style.format("{:.2%}"), use_container_width=True)
            st.caption(
                "VaR: pérdida máxima esperada en el 95% de los escenarios. "
                "CVaR: pérdida promedio en el peor 5% de escenarios."
            )

        with st.expander("Retornos BL posterior vs equilibrio"):
            bl_df = pd.DataFrame({
                "Equilibrio (Π)": res.equilibrium,
                "BL posterior": res.bl_returns,
                "Δ (BL − Π)": res.bl_returns - res.equilibrium,
                "Peso": res.weights,
            }).sort_values("Peso", ascending=False)
            st.dataframe(bl_df.style.format({
                "Equilibrio (Π)": "{:.2%}", "BL posterior": "{:.2%}",
                "Δ (BL − Π)": "{:+.2%}", "Peso": "{:.2%}",
            }), use_container_width=True)
