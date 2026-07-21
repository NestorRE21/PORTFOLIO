# -*- coding: utf-8 -*-
"""Coril SAB — Optimizador de Portafolios Black-Litterman — v5"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optimizer import RiskProfile, ForcedAsset, View, BLConfig, run_profile
from projections import monte_carlo, stress_test, CRISIS_PERIODS

# ═══════════════════════════ CONFIG ═══════════════════════════════════════════
st.set_page_config(page_title="Coril · Portafolios", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
RF_ANNUAL, PPY = 0.02, 52
FICO_TICKER = "FICCMP13"
FICO = ForcedAsset(ret_annual=0.065, vol_annual=0.010, beta=0.30,
                   sector="Factoring", region="Perú", moneda="USD",
                   instrumento="Fondo de inversión")
PERFILES = {
    "Conservador (30/70)": (0.30, 0.70),
    "Moderado-bajo (40/60)": (0.40, 0.60),
    "Moderado (50/50)": (0.50, 0.50),
    "Crecimiento (60/40)": (0.60, 0.40),
    "Agresivo (70/30)": (0.70, 0.30),
}
PERFIL_DESC = {
    "Conservador (30/70)": "Prioriza estabilidad. Ideal para preservar capital.",
    "Moderado-bajo (40/60)": "Leve inclinación a crecimiento con colchón de renta fija.",
    "Moderado (50/50)": "Balance entre crecimiento y protección.",
    "Crecimiento (60/40)": "Mayor exposición a mercado para horizontes largos.",
    "Agresivo (70/30)": "Máxima exposición a renta variable. Mayor volatilidad.",
}
EJEMPLO_TK = ["AAPL", "MSFT", "NVDA", "JNJ", "KO", "QQQ"]
COL_RV, COL_RF, COL_OPT = "#2E5E8C", "#2CA02C", "#D6604D"
BMK_COLORS = ["#888888", "#E377C2", "#FF7F0E", "#9467BD", "#17BECF"]

# ═══════════════════════════ STATE ════════════════════════════════════════════
for k, v in {"tickers":[],"views":[],"optimized":False,"result":None,
             "manual_weights":None,"returns":None,"bench_rets":None,
             "betas":None,"sectors":None,"returns_full":None,
             "bench_full":None,"downloaded_period":None,"data_range":""}.items():
    st.session_state.setdefault(k, v)

# ═══════════════════════════ BACKEND ══════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=600)
def download_equity(tickers, period="5y", interval="1wk"):
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)
    if raw is None or raw.empty: return None
    px = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": list(tickers)[0]})
    px = px.dropna(how="all").ffill()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")

@st.cache_data(show_spinner=False, ttl=600)
def download_benchmarks(tks, period="5y", interval="1wk"):
    import yfinance as yf
    out = {}
    for bk in tks:
        bk = bk.strip().upper()
        if not bk: continue
        try:
            raw = yf.download(bk, period=period, interval=interval, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
            pb = raw["Close"]
            if isinstance(pb, pd.DataFrame): pb = pb.iloc[:, 0]
            pb.index = pd.to_datetime(pb.index).tz_localize(None)
            lr = np.log(pb / pb.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
            lr.name = bk; out[bk] = lr
        except Exception: pass
    return out

def calc_betas(rets, bench):
    common = rets.index.intersection(bench.index)
    bv = bench.loc[common].values; bvar = np.var(bv, ddof=1)
    betas = {}
    for tk in rets.columns:
        tv = rets.loc[common, tk].values; m = np.isfinite(tv) & np.isfinite(bv)
        if m.sum() > 10 and bvar > 1e-12:
            b = np.cov(tv[m], bv[m], ddof=1)[0, 1] / bvar
            betas[tk] = round(float(b), 3) if np.isfinite(b) and b > 0 else 1.0
        else: betas[tk] = 1.0
    return pd.Series(betas)

@st.cache_data(show_spinner=False, ttl=600)
def fetch_sectors(tickers):
    import yfinance as yf
    out = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info or {}
            sec = info.get("sector", "")
            if sec: out[tk] = sec
            elif info.get("quoteType") == "ETF": out[tk] = f"ETF · {info.get('category','') or info.get('longName',tk)[:30]}"
            else: out[tk] = info.get("industry","") or "Sin clasificar"
        except Exception: out[tk] = "Sin clasificar"
    return pd.Series(out)

@st.cache_data(show_spinner=False, ttl=300)
def search_yahoo(query):
    import requests
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                         params={"q": query, "quotesCount": 8, "newsCount": 0},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        return [{"Ticker": q["symbol"], "Nombre": q.get("shortname") or q.get("longname",""),
                 "Tipo": q.get("quoteType",""), "Bolsa": q.get("exchange","")}
                for q in r.json().get("quotes", []) if q.get("symbol")]
    except Exception: return []

def do_optimize(tickers, views_cfg, eq_t, fi_t, primary_bench):
    betas = st.session_state.betas.copy(); betas[FICO_TICKER] = FICO.beta
    views = []
    for v in views_cfg:
        if v["type"] == "absolute": views.append(View(kind="absolute", asset=v["asset"], q=v["q"], confidence=v["confidence"]))
        else: views.append(View(kind="relative", long=v["long"], short=v["short"], q=v["q"], confidence=v["confidence"]))
    config = BLConfig(rf_annual=RF_ANNUAL, periods_per_year=PPY, tau=0.05, max_weight_equity=0.25, gamma_beta=5.0)
    ok = [t for t in tickers if t in st.session_state.returns.columns]
    return run_profile(returns=st.session_state.returns, equity_assets=ok, forced_assets={FICO_TICKER: FICO},
                       profile=RiskProfile.for_split(eq_t, fi_t), views=views, config=config,
                       benchmark_returns=primary_bench, betas=betas)

def wealth_and_dd(w, rets, bd, cap):
    if not bd or not isinstance(bd, dict): bd = {}
    eq = [a for a in w.index if a in rets.columns and a != FICO_TICKER]
    pr = pd.Series(0.0, index=rets.index)
    for c in eq: pr += w.get(c, 0) * rets[c].fillna(0)
    if FICO_TICKER in w.index and w[FICO_TICKER] > 1e-8:
        pr += w[FICO_TICKER] * (np.log(1 + FICO.ret_annual) / PPY)
    pr = pr.fillna(0)
    common = pr.index
    for v in bd.values(): common = common.intersection(v.index)
    pr = pr.loc[common]; wl = np.exp(pr.cumsum()) * cap; dd = wl / wl.cummax() - 1
    bw, bdd = {}, {}
    for n, v in bd.items():
        br = v.loc[common].fillna(0); bw[n] = np.exp(br.cumsum()) * cap; bdd[n] = bw[n] / bw[n].cummax() - 1
    return pr, wl, dd, bw, bdd

def calc_risk(rs):
    c = rs.dropna()
    if len(c) < 10: return {"VaR": np.nan, "CVaR": np.nan}
    from scipy.stats import norm
    mu = c.mean()*PPY; sig = c.std(ddof=1)*np.sqrt(PPY); z = norm.ppf(0.05)
    return {"VaR": -(mu+z*sig), "CVaR": -(mu-sig*norm.pdf(z)/0.05)}

# ═══════════════════════════ SIDEBAR ══════════════════════════════════════════
with st.sidebar:
    st.title("📈 Coril")
    perfil_sel = st.selectbox("Perfil", list(PERFILES.keys()), index=2)
    eq_t, fi_t = PERFILES[perfil_sel]
    st.caption(PERFIL_DESC[perfil_sel])
    c1, c2 = st.columns(2)
    c1.metric("RV", f"{eq_t:.0%}"); c2.metric("RF", f"{fi_t:.0%}")
    st.divider()
    capital = st.slider("Inversión (USD)", 1_000, 1_000_000, 100_000, 1_000, format="$%d")
    period = st.selectbox("Historia", ["1y","2y","3y","5y","10y","max"], index=3)
    st.divider()
    bmk_raw = st.text_area("Benchmarks (uno por línea)", value="^GSPC\nSPY", height=80)
    bmk_list = [b.strip().upper() for b in bmk_raw.split("\n") if b.strip()]
    if bmk_list: st.caption(f"{len(bmk_list)}: {', '.join(bmk_list)}")
    st.divider()
    with st.expander("⚙️ Avanzado"):
        st.caption(f"Renta fija: {FICO_TICKER} · {FICO.ret_annual:.2%}")
        _p = RiskProfile.for_split(eq_t, fi_t)
        st.caption(f"Beta: {_p.beta_min:.2f}–{_p.beta_max:.2f} · DD máx: {_p.max_drawdown:.0%}")
        if st.button("🗑️ Limpiar caché", use_container_width=True,
                     help="Borra los datos descargados guardados en memoria. "
                          "Útil si cambiaste el período o los benchmarks y quieres "
                          "forzar una descarga fresca."):
            st.cache_data.clear(); st.toast("Caché limpiado ✓")

# ═══════════════════════════ HEADER ═══════════════════════════════════════════
st.title("Optimizador de portafolios")
# Progress (se recalcula en cada interacción)
_ht = len(st.session_state.tickers) > 0
_hd = st.session_state.returns is not None
_ho = st.session_state.optimized and st.session_state.result is not None
p1 = "✅" if _ht else "1️⃣"; p2 = "✅" if _hd else ("2️⃣" if _ht else "⬜")
p3 = "✅" if _ho else ("3️⃣" if _hd else "⬜"); p4 = "4️⃣" if _ho else "⬜"
st.caption(f"{p1} Activos → {p2} Datos → {p3} Portafolio → {p4} Proyecciones")

tab1, tab2, tab3, tab4 = st.tabs([
    f"{p1} Activos", f"{p2} Datos y views", f"{p3} Portafolio", f"{p4} Proyecciones"])

# ═══════════════════════════ TAB 1: ACTIVOS ═══════════════════════════════════
with tab1:
    st.subheader("Selecciona los activos")
    if not st.session_state.tickers:
        st.info("👋 **¿Primera vez?** Busca activos abajo o carga un ejemplo.")
        if st.button("🚀 Cargar ejemplo (6 activos US)", type="primary"):
            st.session_state.tickers = list(EJEMPLO_TK)

    q = st.text_input("🔍 Buscar por nombre o ticker", placeholder="Visa, Apple, NVDA…")
    if q.strip():
        res = search_yahoo(q.strip())
        if res:
            st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)
            opts = [f"{r['Ticker']} — {r['Nombre']}" for r in res]
            sel = st.selectbox("Selecciona", [""] + opts, format_func=lambda x: "Elige…" if x=="" else x)
            if sel:
                tk = sel.split(" — ")[0].strip()
                if st.button(f"➕ Añadir {tk}", type="primary"):
                    if tk not in st.session_state.tickers:
                        st.session_state.tickers.append(tk)
                    else: st.toast(f"{tk} ya está en la lista")
        else: st.warning("Sin resultados.")

    if st.session_state.tickers:
        st.divider()
        st.write(f"**Tu universo ({len(st.session_state.tickers)}):**")
        cols = st.columns(min(len(st.session_state.tickers), 5))
        for i, t in enumerate(st.session_state.tickers):
            with cols[i % len(cols)]:
                if st.button(f"❌ {t}", key=f"rm{i}", use_container_width=True):
                    st.session_state.tickers.pop(i)

# ═══════════════════════════ TAB 2: DATOS + VIEWS ═════════════════════════════
with tab2:
    _has_tk = len(st.session_state.tickers) > 0
    if not _has_tk:
        st.info("⬅️ Añade activos en la pestaña anterior.")
    else:
        st.subheader("Descargar datos")
        st.caption(f"{len(st.session_state.tickers)} activos · "
                   f"{len(bmk_list)} benchmarks · {period} de historia")
        if st.session_state.data_range:
            st.success(f"📦 Cargado: {st.session_state.data_range}")

        if st.button("📥 Descargar datos", type="primary", use_container_width=True):
            with st.spinner("Descargando…"):
                lr = download_equity(tuple(st.session_state.tickers), period=period)
            if lr is None or lr.empty:
                st.error("Error descargando precios.")
            else:
                with st.spinner("Benchmarks…"):
                    bd = download_benchmarks(tuple(bmk_list), period=period)
                if not bd: st.error("Error descargando benchmarks.")
                else:
                    common = lr.index
                    for v in bd.values(): common = common.intersection(v.index)
                    st.session_state.returns = lr.loc[common]
                    st.session_state.bench_rets = {k: v.loc[common] for k, v in bd.items()}
                    st.session_state.returns_full = lr
                    st.session_state.bench_full = bd
                    primary = list(bd.values())[0]
                    st.session_state.betas = calc_betas(lr.loc[common], primary.loc[common])
                    ok = [t for t in st.session_state.tickers if t in lr.columns]
                    with st.spinner("Sectores…"):
                        sec = fetch_sectors(tuple(ok)); sec[FICO_TICKER] = FICO.sector
                        st.session_state.sectors = sec
                    st.session_state.downloaded_period = period
                    d1 = lr.index.min().strftime("%Y-%m-%d"); d2 = lr.index.max().strftime("%Y-%m-%d")
                    st.session_state.data_range = f"{d1} → {d2}"
                    st.session_state.optimized = False; st.session_state.result = None
                    st.success(f"✅ {len(common)} semanas · {d1} → {d2}")

        # Views (solo si hay datos)
        _has_data = st.session_state.returns is not None
        if _has_data:
            st.divider()
            st.subheader("Expectativas (opcional)")
            st.caption("¿Tienes una opinión sobre algún activo? Añádela aquí.")
            vt = st.radio("Tipo", ["Retorno de un activo", "Un activo vs otro"], horizontal=True)
            if vt == "Retorno de un activo":
                c1, c2, c3 = st.columns([3, 2, 2])
                va = c1.selectbox("Activo", st.session_state.tickers, key="va")
                vq = c2.number_input("Retorno anual", value=0.10, step=0.01, format="%.2f", key="vq")
                vc = c3.slider("Confianza", 0.1, 1.0, 0.5, 0.1, key="vc")
                if st.button("Añadir"):
                    st.session_state.views.append({"type":"absolute","asset":va,"q":float(vq),"confidence":float(vc)})
            else:
                c1,c2,c3,c4 = st.columns(4)
                vl = c1.selectbox("Ganador", st.session_state.tickers, key="vl")
                vs = c2.selectbox("Perdedor", st.session_state.tickers, key="vs")
                vq = c3.number_input("Diferencia", value=0.05, step=0.01, format="%.2f", key="vqr")
                vc = c4.slider("Confianza", 0.1, 1.0, 0.5, 0.1, key="vcr")
                if st.button("Añadir"):
                    if vl==vs: st.warning("Deben ser distintos.")
                    else: st.session_state.views.append({"type":"relative","long":vl,"short":vs,"q":float(vq),"confidence":float(vc)})
            if st.session_state.views:
                for i, v in enumerate(st.session_state.views):
                    a, b = st.columns([6, 1])
                    txt = f"📌 **{v['asset']}** → {v['q']:.0%}" if v["type"]=="absolute" else f"📌 **{v['long']}** > **{v['short']}** por {v['q']:.0%}"
                    a.write(txt + f" (confianza {v['confidence']:.0%})")
                    if b.button("✕", key=f"rv{i}"): st.session_state.views.pop(i)

# ═══════════════════════════ TAB 3: PORTAFOLIO ════════════════════════════════
with tab3:
    _has_data = st.session_state.returns is not None
    if not _has_data:
        st.info("⬅️ Descarga datos en la pestaña anterior.")
    else:
        st.subheader("Optimizar portafolio")
        st.caption(f"**{perfil_sel}** · ${capital:,.0f}")
        if st.button("🔄 Optimizar", type="primary", use_container_width=True):
            pb = list(st.session_state.bench_rets.values())[0]
            with st.spinner("Calculando…"):
                r = do_optimize(st.session_state.tickers, st.session_state.views, eq_t, fi_t, pb)
            st.session_state.result = r
            st.session_state.manual_weights = r.weights.copy()
            st.session_state.optimized = True
            if r.feasible: st.success("✅ Optimizado")
            else: st.warning(f"⚠️ {r.feasibility_report}")

        _has_opt = st.session_state.optimized and st.session_state.result is not None
        if _has_opt:
            res = st.session_state.result
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Retorno esperado", f"{res.exp_return:.2%}", help="Proyección anual del modelo.")
            m2.metric("Riesgo", f"{res.volatility:.2%}", help="Volatilidad anualizada.")
            m3.metric("Sharpe", f"{res.sharpe:.2f}", help="Retorno por unidad de riesgo.")
            m4.metric("Beta", f"{res.beta:.2f}", help="Sensibilidad al mercado.")

            st.divider()
            st.subheader("Ajustar pesos")
            st.caption("Mueve los sliders. La columna derecha muestra el peso real (suma 100%).")
            cs, csum = st.columns([3, 2])
            with cs:
                nw = {}
                for a in res.weights.index:
                    ef = a == FICO_TICKER
                    nw[a] = st.slider(f"{a} {'🟢RF' if ef else '🔵RV'}", 0.0, 1.0,
                                      float(res.weights[a]), 0.01, key=f"s_{a}")
                wn = pd.Series(nw); tot = wn.sum()
                wnorm = wn / tot if tot > 0 else wn
                st.session_state.manual_weights = wnorm
            with csum:
                st.markdown("**Peso final:**")
                for a in wnorm.index:
                    ic = "🟢" if a == FICO_TICKER else "🔵"
                    st.write(f"{ic} {a}: **{wnorm[a]:.1%}**")
                st.divider()
                eqw = float(wnorm[[a for a in wnorm.index if a != FICO_TICKER]].sum())
                fiw = float(wnorm.get(FICO_TICKER, 0))
                st.metric("RV", f"{eqw:.1%}", delta=f"{eqw-eq_t:+.1%} vs obj")
                st.metric("RF", f"{fiw:.1%}", delta=f"{fiw-fi_t:+.1%} vs obj")

            st.divider()
            g1, g2, g3 = st.columns(3)
            with g1:
                ws = wnorm[wnorm > 1e-4]
                fig = go.Figure(go.Bar(x=ws.values, y=ws.index, orientation="h",
                    marker_color=[COL_RF if a==FICO_TICKER else COL_RV for a in ws.index]))
                fig.update_layout(height=260, margin=dict(l=0,r=0,t=5,b=0), xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                fig = go.Figure(go.Pie(labels=["RV","RF"], values=[eqw,fiw],
                    marker_colors=[COL_RV,COL_RF], hole=.5))
                fig.update_layout(height=260, margin=dict(l=0,r=0,t=5,b=0))
                st.plotly_chart(fig, use_container_width=True)
            with g3:
                sec = st.session_state.sectors
                if sec is not None and not sec.empty:
                    sw = {}
                    for a in wnorm.index:
                        if wnorm[a]>1e-4: s=sec.get(a,"?"); sw[s]=sw.get(s,0)+wnorm[a]
                    fig = go.Figure(go.Pie(labels=list(sw.keys()), values=list(sw.values()), hole=.5))
                    fig.update_layout(height=260, margin=dict(l=0,r=0,t=5,b=0))
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader(f"Evolución histórica (${capital:,.0f})")
            pr, wl, dd, bw, bdd = wealth_and_dd(wnorm, st.session_state.returns,
                                                  st.session_state.bench_rets, capital)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.7,.3], vertical_spacing=.05)
            fig.add_trace(go.Scatter(x=wl.index, y=wl.values, name="Portafolio",
                                    line=dict(color=COL_RV, width=2.5)), row=1, col=1)
            for i,(n,v) in enumerate(bw.items()):
                fig.add_trace(go.Scatter(x=v.index, y=v.values, name=n,
                    line=dict(color=BMK_COLORS[i%len(BMK_COLORS)], dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                                    fill="tozeroy", line=dict(color=COL_OPT)), row=2, col=1)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
            fig.update_yaxes(tickformat=".0%", row=2, col=1)
            fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
                             legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

            ann_r = np.exp(pr.mean()*PPY)-1; ann_v = pr.std(ddof=1)*np.sqrt(PPY)
            risk = calc_risk(pr)
            h1,h2,h3,h4,h5 = st.columns(5)
            h1.metric("Ret. histórico", f"{ann_r:.2%}")
            h2.metric("Volatilidad", f"{ann_v:.2%}")
            h3.metric("Peor caída", f"{dd.min():.2%}")
            h4.metric("VaR 95%", f"{risk['VaR']:.2%}", help="Pérdida máx en 95% de casos.")
            h5.metric("CVaR 95%", f"{risk['CVaR']:.2%}", help="Pérdida prom. en el peor 5%.")

# ═══════════════════════════ TAB 4: PROYECCIONES ══════════════════════════════
with tab4:
    _has_opt = st.session_state.optimized and st.session_state.result is not None
    if not _has_opt:
        st.info("⬅️ Optimiza el portafolio primero.")
    else:
        res = st.session_state.result
        wnorm = st.session_state.manual_weights if st.session_state.manual_weights is not None else res.weights

        st.subheader("🎲 Proyección Monte Carlo")
        st.caption("Simula miles de futuros posibles para ver el rango de resultados.")
        c1,c2,c3 = st.columns(3)
        mh = c1.selectbox("Horizonte", [1,2,3,5,10], index=2, format_func=lambda x: f"{x} año{'s' if x>1 else ''}")
        mn = c2.selectbox("Simulaciones", [1000,5000,10000], index=1)
        mt = c3.number_input("Objetivo", value=int(capital*1.2), step=10_000, format="%d")

        if st.button("▶️ Simular", type="primary", use_container_width=True):
            with st.spinner(f"Simulando {mn:,} futuros…"):
                mc = monte_carlo(wnorm, res.bl_returns, res.cov_matrix, capital, mh, PPY, mn, mt)
            st.session_state["mc"] = mc

        if "mc" in st.session_state and st.session_state["mc"] is not None:
            mc = st.session_state["mc"]
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Capital proyectado", f"${mc.median_path[-1]:,.0f}", help="Mediana.")
            m2.metric("Prob. pérdida", f"{mc.prob_loss:.1%}")
            m3.metric(f"Prob. ≥${mc.target:,.0f}", f"{mc.prob_target:.1%}")
            m4.metric("VaR terminal", f"${mc.var_terminal:,.0f}")
            m5.metric("CVaR terminal", f"${mc.cvar_terminal:,.0f}")
            st.info(f"💡 Escenario central: **${mc.median_path[-1]:,.0f}** en {mc.horizon_years:.0f} año(s). "
                    f"Prob. de pérdida: **{mc.prob_loss:.1%}**. Prob. de alcanzar objetivo: **{mc.prob_target:.1%}**.")
            fig = go.Figure()
            x = mc.dates
            for lo,hi,cl in [(5,95,"rgba(46,94,140,0.08)"),(10,90,"rgba(46,94,140,0.12)"),(25,75,"rgba(46,94,140,0.18)")]:
                fig.add_trace(go.Scatter(x=list(x)+list(x[::-1]),y=list(mc.percentiles[hi])+list(mc.percentiles[lo][::-1]),
                    fill="toself",fillcolor=cl,line=dict(width=0),name=f"P{lo}–P{hi}"))
            fig.add_trace(go.Scatter(x=x,y=mc.median_path,name="Mediana",line=dict(color=COL_RV,width=2.5)))
            fig.add_hline(y=mc.capital,line_dash="dot",line_color="gray",annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target!=mc.capital: fig.add_hline(y=mc.target,line_dash="dot",line_color=COL_RF,annotation_text=f"Objetivo ${mc.target:,.0f}")
            fig.update_yaxes(tickprefix="$",tickformat=",.0f")
            fig.update_layout(height=400,margin=dict(l=0,r=0,t=10,b=0),legend=dict(orientation="h",y=-0.1))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📊 Tabla de escenarios"):
                labels = ["Pesimista","Conservador","Probable bajo","Central","Probable alto","Optimista","Mejor caso"]
                pct_df = pd.DataFrame({"Escenario":labels,
                    "Capital final":[f"${mc.percentiles[p][-1]:,.0f}" for p in [5,10,25,50,75,90,95]],
                    "Retorno":[f"{mc.percentiles[p][-1]/mc.capital-1:.1%}" for p in [5,10,25,50,75,90,95]]})
                st.dataframe(pct_df, use_container_width=True, hide_index=True)

        # ── STRESS ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔥 Pruebas de estrés")
        st.caption("¿Qué pasaría si se repitiera una crisis del pasado?")
        ret_st = st.session_state.returns_full if st.session_state.returns_full is not None else st.session_state.returns
        if ret_st is not None:
            st.caption(f"📅 Datos: {ret_st.index.min().strftime('%Y-%m-%d')} → {ret_st.index.max().strftime('%Y-%m-%d')}")
        bench_st = st.session_state.bench_full if st.session_state.bench_full is not None else (st.session_state.bench_rets if isinstance(st.session_state.bench_rets, dict) else {})
        if st.button("▶️ Correr estrés", use_container_width=True):
            pb = list(bench_st.values())[0] if bench_st else None
            with st.spinner("Analizando…"):
                stres = stress_test(wnorm, ret_st, CRISIS_PERIODS, capital, {FICO_TICKER:FICO}, PPY, pb)
            st.session_state["stress"] = stres
        if "stress" in st.session_state and st.session_state["stress"]:
            stres = st.session_state["stress"]
            avail = [s for s in stres if s.available]
            if not avail: st.warning("Sin datos en tu rango. Usa más años de historia.")
            else:
                worst = min(avail, key=lambda s: s.port_return)
                beats = sum(1 for s in avail if s.port_return > s.benchmark_return)
                st.info(f"💡 Peor crisis: **{worst.name}** ({worst.port_return:+.2%}). "
                        f"Superas al benchmark en **{beats}/{len(avail)}** escenarios.")
                # Gráfico de barras
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[s.name for s in avail], y=[s.port_return for s in avail],
                                     name="Portafolio", marker_color=COL_OPT))
                fig.add_trace(go.Bar(x=[s.name for s in avail], y=[s.benchmark_return for s in avail],
                                     name="Benchmark", marker_color="#888"))
                fig.update_yaxes(tickformat=".1%")
                fig.update_layout(barmode="group", height=320, margin=dict(l=0,r=0,t=10,b=0),
                                 legend=dict(orientation="h", y=1.08))
                st.plotly_chart(fig, use_container_width=True)
                # Detalle
                for s in avail:
                    ic = "🔴" if s.port_return < 0 else "🟢"
                    diff = s.port_return - s.benchmark_return
                    cA, cB = st.columns([3, 1])
                    with cA:
                        st.markdown(f"{ic} **{s.name}** · {s.start} → {s.end}")
                        st.caption(s.description)
                    with cB:
                        st.metric("Impacto", f"{s.port_return:+.2%}", delta=f"${s.port_loss:,.0f}", delta_color="off")
                    mejor = "mejor" if diff > 0 else "peor"
                    st.caption(f"→ **{mejor.upper()}** que benchmark ({s.port_return:+.2%} vs {s.benchmark_return:+.2%}). "
                               f"Caída máx: {s.max_drawdown:.2%}.")
                    if not s.asset_returns.empty:
                        ar = s.asset_returns.sort_values()
                        st.caption(f"→ Más golpeado: **{ar.index[0]}** ({ar.iloc[0]:+.2%}). "
                                   f"Más resiliente: **{ar.index[-1]}** ({ar.iloc[-1]:+.2%}).")
                    st.divider()
            missing = [s for s in stres if not s.available]
            if missing:
                with st.expander(f"ℹ️ {len(missing)} crisis fuera del rango"):
                    for s in missing: st.caption(f"**{s.name}** ({s.start}→{s.end}): {s.description}")
