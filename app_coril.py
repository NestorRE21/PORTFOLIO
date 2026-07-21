# -*- coding: utf-8 -*-
"""
Coril SAB — Optimizador de Portafolios Black-Litterman
Interfaz comercial (Streamlit) — v4 UX simplificada
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optimizer import RiskProfile, ForcedAsset, View, BLConfig, run_profile
from projections import monte_carlo, stress_test, CRISIS_PERIODS

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
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
PERFIL_DESCRIP = {
    "Conservador (30/70)": "Prioriza estabilidad. Ideal para preservar capital.",
    "Moderado-bajo (40/60)": "Leve inclinación a crecimiento con colchón de renta fija.",
    "Moderado (50/50)": "Balance entre crecimiento y protección.",
    "Crecimiento (60/40)": "Mayor exposición a mercado para horizontes largos.",
    "Agresivo (70/30)": "Máxima exposición a renta variable. Mayor volatilidad.",
}
EJEMPLO_TICKERS = ["AAPL", "MSFT", "NVDA", "JNJ", "KO", "QQQ"]
COL_RV, COL_RF, COL_BMK, COL_OPT = "#2E5E8C", "#2CA02C", "#888888", "#D6604D"
BMK_COLORS = ["#888888", "#E377C2", "#FF7F0E", "#9467BD", "#17BECF"]

# ═══════════════════════════════════════════════════════════════════════════════
# ESTADO
# ═══════════════════════════════════════════════════════════════════════════════
def init_state():
    for k, v in {
        "tickers": [], "views": [], "optimized": False, "result": None,
        "manual_weights": None, "returns": None, "bench_rets": None,
        "betas": None, "sectors": None, "returns_full": None,
        "bench_full": None, "downloaded_period": None, "data_range": "",
    }.items():
        st.session_state.setdefault(k, v)
init_state()

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE DATOS (backend — no tocar)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=600)
def download_equity(tickers, period="5y", interval="1wk"):
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty: return None
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw[["Close"]].copy(); px.columns = list(tickers)[:1]
    px = px.dropna(how="all").ffill()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")

@st.cache_data(show_spinner=False, ttl=600)
def download_benchmarks(bench_tickers, period="5y", interval="1wk"):
    import yfinance as yf
    result = {}
    for bk in bench_tickers:
        bk = bk.strip().upper()
        if not bk: continue
        try:
            raw = yf.download(bk, period=period, interval=interval,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            pb = raw["Close"]
            if isinstance(pb, pd.DataFrame): pb = pb.iloc[:, 0]
            pb.index = pd.to_datetime(pb.index).tz_localize(None)
            lr = np.log(pb / pb.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
            lr.name = bk; result[bk] = lr
        except Exception: pass
    return result

def calc_betas(returns, bench_ret):
    common = returns.index.intersection(bench_ret.index)
    bmk = bench_ret.loc[common].values; bmk_var = np.var(bmk, ddof=1)
    betas = {}
    for tk in returns.columns:
        tv = returns.loc[common, tk].values
        mask = np.isfinite(tv) & np.isfinite(bmk)
        if mask.sum() > 10 and bmk_var > 1e-12:
            b = np.cov(tv[mask], bmk[mask], ddof=1)[0, 1] / bmk_var
            betas[tk] = round(float(b), 3) if np.isfinite(b) and b > 0 else 1.0
        else: betas[tk] = 1.0
    return pd.Series(betas)

@st.cache_data(show_spinner=False, ttl=600)
def fetch_sectors(tickers):
    import yfinance as yf
    sectors = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info or {}
            qt = info.get("quoteType", "")
            sec = info.get("sector", "")
            cat = info.get("category", "")
            if sec: sectors[tk] = sec
            elif qt == "ETF": sectors[tk] = f"ETF · {cat or info.get('longName', tk)[:30]}"
            elif qt == "MUTUALFUND": sectors[tk] = f"Fondo · {cat or 'Mixto'}"
            else: sectors[tk] = info.get("industry", "") or "Sin clasificar"
        except Exception: sectors[tk] = "Sin clasificar"
    return pd.Series(sectors)

@st.cache_data(show_spinner=False, ttl=300)
def search_yahoo(query):
    import requests
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                         params={"q": query, "quotesCount": 8, "newsCount": 0},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        return [{"Ticker": q.get("symbol",""), "Nombre": q.get("shortname") or q.get("longname",""),
                 "Tipo": q.get("quoteType",""), "Bolsa": q.get("exchange","")}
                for q in r.json().get("quotes", []) if q.get("symbol")]
    except Exception: return []

def optimize(tickers, views_cfg, eq_t, fi_t, primary_bench):
    returns = st.session_state.returns
    betas = st.session_state.betas.copy(); betas[FICO_TICKER] = FICO.beta
    views = []
    for v in views_cfg:
        if v["type"] == "absolute":
            views.append(View(kind="absolute", asset=v["asset"], q=v["q"], confidence=v["confidence"]))
        else:
            views.append(View(kind="relative", long=v["long"], short=v["short"], q=v["q"], confidence=v["confidence"]))
    profile = RiskProfile.for_split(eq_t, fi_t)
    config = BLConfig(rf_annual=RF_ANNUAL, periods_per_year=PPY, tau=0.05,
                      max_weight_equity=0.25, gamma_beta=5.0)
    ok = [t for t in tickers if t in returns.columns]
    return run_profile(returns=returns, equity_assets=ok, forced_assets={FICO_TICKER: FICO},
                       profile=profile, views=views, config=config,
                       benchmark_returns=primary_bench, betas=betas)

def wealth_and_dd(weights, returns, bench_dict, capital):
    if not bench_dict or not isinstance(bench_dict, dict): bench_dict = {}
    eq_cols = [a for a in weights.index if a in returns.columns and a != FICO_TICKER]
    w_eq = weights.reindex(eq_cols).fillna(0.0)
    port_ret = pd.Series(0.0, index=returns.index)
    for col in eq_cols: port_ret = port_ret + w_eq[col] * returns[col].fillna(0.0)
    if FICO_TICKER in weights.index and weights[FICO_TICKER] > 1e-8:
        port_ret = port_ret + weights[FICO_TICKER] * (np.log(1 + FICO.ret_annual) / PPY)
    port_ret = port_ret.fillna(0.0)
    common = port_ret.index
    for bk_ret in bench_dict.values(): common = common.intersection(bk_ret.index)
    port_ret = port_ret.loc[common]
    wealth = np.exp(port_ret.cumsum()) * capital; dd = wealth / wealth.cummax() - 1.0
    bw, bd = {}, {}
    for name, bk_ret in bench_dict.items():
        br = bk_ret.loc[common].fillna(0.0)
        bw[name] = np.exp(br.cumsum()) * capital; bd[name] = bw[name] / bw[name].cummax() - 1.0
    return port_ret, wealth, dd, bw, bd

def calc_var_cvar(rs, confidence=0.95):
    c = rs.dropna()
    if len(c) < 10: return {"VaR_param": np.nan, "CVaR_param": np.nan}
    from scipy.stats import norm
    mu = c.mean() * PPY; sig = c.std(ddof=1) * np.sqrt(PPY)
    alpha = 1 - confidence; z = norm.ppf(alpha)
    return {"VaR_param": -(mu + z * sig), "CVaR_param": -(mu - sig * norm.pdf(z) / alpha)}

# ═══════════════════════════════════════════════════════════════════════════════
# INDICADOR DE PROGRESO
# ═══════════════════════════════════════════════════════════════════════════════
def step_status():
    has_tickers = len(st.session_state.tickers) > 0
    has_data = st.session_state.returns is not None
    has_opt = st.session_state.optimized and st.session_state.result is not None
    return has_tickers, has_data, has_opt

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — configuración limpia
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuración")

    perfil_sel = st.selectbox("Perfil de riesgo", list(PERFILES.keys()), index=2,
                              help="Define el balance entre renta variable y renta fija.")
    eq_t, fi_t = PERFILES[perfil_sel]
    st.caption(PERFIL_DESCRIP[perfil_sel])
    c1, c2 = st.columns(2)
    c1.metric("Renta variable", f"{eq_t:.0%}")
    c2.metric("Renta fija", f"{fi_t:.0%}")

    st.divider()
    capital_inicial = st.slider("Inversión inicial (USD)", 1_000, 1_000_000,
                                100_000, 1_000, format="$%d")
    data_period = st.selectbox("Años de historia", ["1y","2y","3y","5y","10y","max"],
                               index=3, help="Más años = mejor stress testing pero "
                               "descarga más lenta.")

    st.divider()
    benchmarks_raw = st.text_area("Benchmarks (uno por línea)",
                                  value="^GSPC\nSPY", height=80,
                                  help="Índices de comparación. Se descargan al "
                                  "pulsar el botón en la pestaña de activos.")
    benchmarks_list = [b.strip().upper() for b in benchmarks_raw.split("\n") if b.strip()]

    with st.expander("⚙️ Avanzado"):
        st.caption(f"**Renta fija forzada:** {FICO_TICKER} · "
                   f"Ret. {FICO.ret_annual:.2%} · Vol {FICO.vol_annual:.2%}")
        _prof = RiskProfile.for_split(eq_t, fi_t)
        st.caption(f"Beta objetivo: {0.5*(_prof.beta_min+_prof.beta_max):.2f} "
                   f"(rango {_prof.beta_min:.2f}–{_prof.beta_max:.2f})")
        st.caption(f"Máx. drawdown permitido: {_prof.max_drawdown:.0%}")
        if st.button("Limpiar caché de datos", use_container_width=True):
            st.cache_data.clear(); st.success("Caché limpiado.")

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER + PROGRESO
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📈 Optimizador de portafolios")

has_tk, has_data, has_opt = step_status()
p1 = "✅" if has_tk else "1️⃣"
p2 = "✅" if has_data else ("2️⃣" if has_tk else "⬜")
p3 = "✅" if has_opt else ("3️⃣" if has_data else "⬜")
p4 = "4️⃣" if has_opt else "⬜"
st.caption(f"{p1} Activos → {p2} Datos → {p3} Optimizar → {p4} Proyecciones")

tab1, tab2, tab3, tab4 = st.tabs([
    f"{p1} Activos", f"{p2} Datos y views", f"{p3} Portafolio", f"{p4} Proyecciones"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ACTIVOS (buscar + añadir)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Selecciona los activos de renta variable")

    if not st.session_state.tickers:
        st.info("👋 **¿Primera vez?** Busca activos por nombre o ticker abajo, "
                "o carga un ejemplo rápido para explorar la herramienta.")
        if st.button("🚀 Cargar ejemplo (6 activos US)", type="primary"):
            st.session_state.tickers = list(EJEMPLO_TICKERS)
            st.rerun()

    search_q = st.text_input("🔍 Buscar por nombre o ticker",
                             placeholder="Ej: Visa, Apple, semiconductores, NVDA…")
    if search_q.strip():
        results = search_yahoo(search_q.strip())
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            options = [f"{r['Ticker']} — {r['Nombre']}" for r in results]
            sel = st.selectbox("Selecciona para añadir", [""] + options, key="sel_tk",
                               format_func=lambda x: "Elige un activo…" if x == "" else x)
            if sel:
                tk = sel.split(" — ")[0].strip()
                if st.button(f"➕ Añadir {tk}", type="primary"):
                    if tk not in st.session_state.tickers:
                        st.session_state.tickers.append(tk); st.rerun()
                    else: st.warning(f"{tk} ya está en la lista.")
        else:
            st.warning("No se encontraron resultados. Prueba con otro término.")

    if st.session_state.tickers:
        st.divider()
        st.write(f"**Tu portafolio ({len(st.session_state.tickers)} activos):**")
        cols = st.columns(min(len(st.session_state.tickers), 4))
        for i, t in enumerate(st.session_state.tickers):
            with cols[i % len(cols)]:
                if st.button(f"❌ {t}", key=f"rm{i}", use_container_width=True):
                    st.session_state.tickers.pop(i); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATOS + VIEWS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not has_tk:
        st.info("⬅️ Primero añade activos en la pestaña anterior.")
    else:
        st.subheader("Descargar datos de mercado")
        st.caption(f"Se descargarán **{len(st.session_state.tickers)} activos** y "
                   f"**{len(benchmarks_list)} benchmark(s)** con **{data_period}** de historia.")

        if st.session_state.data_range:
            st.success(f"📦 Datos cargados: {st.session_state.data_range}")

        if st.button("📥 Descargar datos", type="primary", use_container_width=True):
            with st.spinner("Descargando precios y benchmarks…"):
                log_ret = download_equity(tuple(st.session_state.tickers), period=data_period)
            if log_ret is None or log_ret.empty:
                st.error("No se pudieron descargar precios.")
            else:
                with st.spinner("Descargando benchmarks…"):
                    bench_dict = download_benchmarks(tuple(benchmarks_list), period=data_period)
                if not bench_dict:
                    st.error("No se pudieron descargar benchmarks.")
                else:
                    common = log_ret.index
                    for bret in bench_dict.values(): common = common.intersection(bret.index)
                    st.session_state.returns = log_ret.loc[common]
                    st.session_state.bench_rets = {k: v.loc[common] for k, v in bench_dict.items()}
                    st.session_state.returns_full = log_ret
                    st.session_state.bench_full = bench_dict
                    primary = list(bench_dict.values())[0]
                    st.session_state.betas = calc_betas(log_ret.loc[common], primary.loc[common])
                    st.session_state.betas[FICO_TICKER] = FICO.beta
                    ok = [t for t in st.session_state.tickers if t in log_ret.columns]
                    with st.spinner("Obteniendo sectores…"):
                        sectors = fetch_sectors(tuple(ok))
                        sectors[FICO_TICKER] = FICO.sector
                        st.session_state.sectors = sectors
                    st.session_state.downloaded_period = data_period
                    dmin = st.session_state.returns.index.min().strftime("%Y-%m-%d")
                    dmax = st.session_state.returns.index.max().strftime("%Y-%m-%d")
                    st.session_state.data_range = f"{dmin} → {dmax}"
                    st.session_state.optimized = False; st.session_state.result = None
                    st.rerun()  # Refresca para que las pestañas se actualicen

        # ── Views ────────────────────────────────────────────────────────
        if has_data:
            st.divider()
            st.subheader("Expectativas del analista (opcional)")
            st.caption("¿Crees que un activo rendirá más o menos que lo que dice el mercado? "
                       "Añade tus expectativas aquí. Si no tienes, déjalo vacío.")

            vt = st.radio("Tipo", ["Retorno esperado de un activo",
                                    "Un activo rendirá más que otro"], horizontal=True)
            if vt == "Retorno esperado de un activo":
                c1, c2, c3 = st.columns([3, 2, 2])
                va = c1.selectbox("Activo", st.session_state.tickers, key="a_abs")
                vq = c2.number_input("Retorno anual esperado", value=0.10, step=0.01,
                                     format="%.2f", key="q_abs",
                                     help="Ej: 0.10 = esperas 10% anual.")
                vc = c3.slider("¿Qué tan seguro estás?", 0.1, 1.0, 0.5, 0.1, key="c_abs",
                               help="1.0 = muy seguro, 0.1 = apenas una intuición.")
                if st.button("Añadir expectativa"):
                    st.session_state.views.append({"type":"absolute","asset":va,
                                                   "q":float(vq),"confidence":float(vc)})
                    st.rerun()
            else:
                c1, c2, c3, c4 = st.columns(4)
                vl = c1.selectbox("Ganador", st.session_state.tickers, key="l_rel")
                vs = c2.selectbox("Perdedor", st.session_state.tickers, key="s_rel")
                vq = c3.number_input("Diferencia anual", value=0.05, step=0.01,
                                     format="%.2f", key="q_rel")
                vc = c4.slider("Confianza", 0.1, 1.0, 0.5, 0.1, key="c_rel")
                if st.button("Añadir expectativa"):
                    if vl == vs: st.warning("Deben ser activos distintos.")
                    else:
                        st.session_state.views.append({"type":"relative","long":vl,
                                                       "short":vs,"q":float(vq),
                                                       "confidence":float(vc)})
                        st.rerun()

            if st.session_state.views:
                for i, v in enumerate(st.session_state.views):
                    a, b = st.columns([6, 1])
                    if v["type"] == "absolute":
                        a.write(f"📌 **{v['asset']}** rendirá {v['q']:.0%} "
                                f"(confianza {v['confidence']:.0%})")
                    else:
                        a.write(f"📌 **{v['long']}** superará a **{v['short']}** "
                                f"por {v['q']:.0%} (confianza {v['confidence']:.0%})")
                    if b.button("✕", key=f"rmv{i}"):
                        st.session_state.views.pop(i); st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTAFOLIO (optimizar + resultados)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not has_data:
        st.info("⬅️ Primero descarga los datos en la pestaña anterior.")
    else:
        st.subheader("Optimizar portafolio")
        st.caption(f"Perfil: **{perfil_sel}** · Inversión: **${capital_inicial:,.0f}**")

        if st.button("🔄 Optimizar", type="primary", use_container_width=True):
            primary_bench = list(st.session_state.bench_rets.values())[0]
            with st.spinner("Calculando pesos óptimos…"):
                res = optimize(st.session_state.tickers, st.session_state.views,
                               eq_t, fi_t, primary_bench)
            st.session_state.result = res
            st.session_state.manual_weights = res.weights.copy()
            st.session_state.optimized = True
            st.rerun()  # Refresca para mostrar resultados inmediatamente

        if has_opt:
            res = st.session_state.result
            # ── Métricas principales ─────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retorno esperado", f"{res.exp_return:.2%}",
                      help="Retorno anual proyectado por el modelo Black-Litterman.")
            m2.metric("Riesgo (volatilidad)", f"{res.volatility:.2%}",
                      help="Desviación estándar anualizada. Mide cuánto puede variar.")
            m3.metric("Sharpe", f"{res.sharpe:.2f}",
                      help="Retorno por unidad de riesgo. Más alto = mejor.")
            m4.metric("Beta", f"{res.beta:.2f}",
                      help="Sensibilidad al mercado. 1 = se mueve igual que el benchmark.")

            # ── Pesos manuales ───────────────────────────────────────────
            st.divider()
            st.subheader("Ajustar pesos")
            st.caption("Mueve los sliders para cambiar la distribución. "
                       "La columna derecha muestra el peso real (siempre suma 100%).")
            cs, csum = st.columns([3, 2])
            with cs:
                nuevos = {}
                for a in res.weights.index:
                    ef = a == FICO_TICKER
                    nuevos[a] = st.slider(f"{a} {'🟢 RF' if ef else '🔵 RV'}",
                                          0.0, 1.0, float(res.weights[a]), 0.01,
                                          key=f"sl_{a}")
                wn = pd.Series(nuevos); total = wn.sum()
                wnorm = wn / total if total > 0 else wn
                st.session_state.manual_weights = wnorm

            with csum:
                st.markdown("**Peso final:**")
                for a in wnorm.index:
                    cl = "🟢" if a == FICO_TICKER else "🔵"
                    st.write(f"{cl} {a}: **{wnorm[a]:.1%}**")
                st.divider()
                eqw = float(wnorm[[a for a in wnorm.index if a != FICO_TICKER]].sum())
                fiw = float(wnorm.get(FICO_TICKER, 0.0))
                st.metric("Renta variable", f"{eqw:.1%}",
                          delta=f"{eqw-eq_t:+.1%} vs objetivo")
                st.metric("Renta fija", f"{fiw:.1%}",
                          delta=f"{fiw-fi_t:+.1%} vs objetivo")

            # ── Gráficos ─────────────────────────────────────────────────
            st.divider()
            g1, g2, g3 = st.columns(3)
            with g1:
                st.caption("Distribución por activo")
                ws = wnorm[wnorm > 1e-4]
                fig = go.Figure(go.Bar(x=ws.values, y=ws.index, orientation="h",
                    marker_color=[COL_RF if a == FICO_TICKER else COL_RV for a in ws.index]))
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=5,b=0), xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                st.caption("Renta variable vs fija")
                fig = go.Figure(go.Pie(labels=["Renta variable","Renta fija"],
                    values=[eqw,fiw], marker_colors=[COL_RV,COL_RF], hole=.5))
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=5,b=0))
                st.plotly_chart(fig, use_container_width=True)
            with g3:
                st.caption("Por sector")
                sectors = st.session_state.sectors
                if sectors is not None and not sectors.empty:
                    sw = {}
                    for a in wnorm.index:
                        if wnorm[a] > 1e-4:
                            s = sectors.get(a, "Sin clasificar")
                            sw[s] = sw.get(s, 0) + wnorm[a]
                    fig = go.Figure(go.Pie(labels=list(sw.keys()), values=list(sw.values()), hole=.5))
                    fig.update_layout(height=280, margin=dict(l=0,r=0,t=5,b=0))
                    st.plotly_chart(fig, use_container_width=True)

            # ── Evolución de capital ─────────────────────────────────────
            st.divider()
            st.subheader(f"Evolución histórica (base ${capital_inicial:,.0f})")
            pr, wealth, dd, bw, bd = wealth_and_dd(wnorm, st.session_state.returns,
                                                    st.session_state.bench_rets, capital_inicial)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[.7,.3], vertical_spacing=.05)
            fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, name="Tu portafolio",
                                    line=dict(color=COL_RV, width=2.5)), row=1, col=1)
            for idx, (bn, bwv) in enumerate(bw.items()):
                fig.add_trace(go.Scatter(x=bwv.index, y=bwv.values, name=bn,
                    line=dict(color=BMK_COLORS[idx%len(BMK_COLORS)], dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Caída máxima",
                                    fill="tozeroy", line=dict(color=COL_OPT)), row=2, col=1)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
            fig.update_yaxes(tickformat=".0%", row=2, col=1)
            fig.update_layout(height=450, margin=dict(l=0,r=0,t=10,b=0),
                             legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

            # ── Métricas históricas ──────────────────────────────────────
            ann_ret = np.exp(pr.mean() * PPY) - 1
            ann_vol = pr.std(ddof=1) * np.sqrt(PPY)
            risk = calc_var_cvar(pr)
            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("Retorno histórico", f"{ann_ret:.2%}")
            h2.metric("Volatilidad", f"{ann_vol:.2%}")
            h3.metric("Peor caída", f"{dd.min():.2%}")
            h4.metric("VaR 95%", f"{risk['VaR_param']:.2%}",
                      help="Pérdida máxima en el 95% de escenarios (anual).")
            h5.metric("CVaR 95%", f"{risk['CVaR_param']:.2%}",
                      help="Pérdida promedio en el peor 5% (anual).")

            with st.expander("📋 Detalle: retornos del modelo vs mercado"):
                bl_df = pd.DataFrame({
                    "Mercado (equilibrio)": res.equilibrium,
                    "Modelo (BL)": res.bl_returns,
                    "Diferencia": res.bl_returns - res.equilibrium,
                    "Peso": res.weights,
                }).sort_values("Peso", ascending=False)
                st.dataframe(bl_df.style.format({"Mercado (equilibrio)":"{:.2%}",
                    "Modelo (BL)":"{:.2%}","Diferencia":"{:+.2%}","Peso":"{:.2%}"}),
                    use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROYECCIONES Y ESCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not has_opt:
        st.info("⬅️ Primero optimiza el portafolio en la pestaña anterior.")
    else:
        res = st.session_state.result
        wnorm = st.session_state.manual_weights if st.session_state.manual_weights is not None else res.weights

        # ── MONTE CARLO ──────────────────────────────────────────────────
        st.subheader("🎲 Proyección Monte Carlo")
        st.caption("Simula miles de futuros posibles para tu portafolio, "
                   "mostrando el rango de resultados probables.")

        cc1, cc2, cc3 = st.columns(3)
        mc_h = cc1.selectbox("Horizonte", [1,2,3,5,10], index=2,
                             format_func=lambda x: f"{x} año{'s' if x>1 else ''}")
        mc_n = cc2.selectbox("Simulaciones", [1000,5000,10000], index=1)
        mc_t = cc3.number_input("Capital objetivo", value=int(capital_inicial*1.2),
                                step=10_000, format="%d")

        if st.button("▶️ Simular", type="primary", use_container_width=True):
            with st.spinner(f"Simulando {mc_n:,} futuros a {mc_h} años…"):
                mc = monte_carlo(wnorm, res.bl_returns, res.cov_matrix,
                                 capital_inicial, mc_h, PPY, mc_n, mc_t)
            st.session_state["mc_result"] = mc

        if "mc_result" in st.session_state and st.session_state["mc_result"]:
            mc = st.session_state["mc_result"]
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Capital proyectado", f"${mc.median_path[-1]:,.0f}",
                      help="Mediana: la mitad de los escenarios termina arriba, la mitad abajo.")
            m2.metric("Prob. de pérdida", f"{mc.prob_loss:.1%}")
            m3.metric(f"Prob. ≥ ${mc.target:,.0f}", f"{mc.prob_target:.1%}")
            m4.metric("Pérdida máx. probable", f"${mc.var_terminal:,.0f}",
                      help="VaR 95%: en el 95% de los casos pierdes menos que esto.")
            m5.metric("Pérdida promedio extrema", f"${mc.cvar_terminal:,.0f}",
                      help="CVaR 95%: pérdida promedio en el peor 5%.")

            st.info(f"💡 Con ${mc.capital:,.0f} a {mc.horizon_years:.0f} año(s), "
                    f"el escenario central proyecta **${mc.median_path[-1]:,.0f}**. "
                    f"Hay un **{mc.prob_loss:.1%}** de probabilidad de terminar en pérdida, "
                    f"y un **{mc.prob_target:.1%}** de alcanzar el objetivo de ${mc.target:,.0f}.")

            fig = go.Figure()
            x = mc.dates
            for lo,hi,col in [(5,95,"rgba(46,94,140,0.08)"),
                              (10,90,"rgba(46,94,140,0.12)"),
                              (25,75,"rgba(46,94,140,0.18)")]:
                fig.add_trace(go.Scatter(x=list(x)+list(x[::-1]),
                    y=list(mc.percentiles[hi])+list(mc.percentiles[lo][::-1]),
                    fill="toself",fillcolor=col,line=dict(width=0),name=f"P{lo}–P{hi}"))
            fig.add_trace(go.Scatter(x=x,y=mc.median_path,name="Mediana (escenario central)",
                                    line=dict(color="#2E5E8C",width=2.5)))
            fig.add_hline(y=mc.capital,line_dash="dot",line_color="gray",
                         annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target != mc.capital:
                fig.add_hline(y=mc.target,line_dash="dot",line_color="#2CA02C",
                             annotation_text=f"Objetivo ${mc.target:,.0f}")
            fig.update_yaxes(tickprefix="$",tickformat=",.0f")
            fig.update_layout(height=420,margin=dict(l=0,r=0,t=10,b=0),
                             legend=dict(orientation="h",y=-0.1))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📊 Tabla de escenarios"):
                pct_df = pd.DataFrame({
                    "Escenario": ["Pesimista (5%)","Conservador (10%)","Probable bajo (25%)",
                                  "Central (50%)","Probable alto (75%)","Optimista (90%)","Mejor caso (95%)"],
                    "Capital final": [f"${mc.percentiles[p][-1]:,.0f}" for p in [5,10,25,50,75,90,95]],
                    "Retorno total": [f"{mc.percentiles[p][-1]/mc.capital-1:.1%}" for p in [5,10,25,50,75,90,95]],
                })
                st.dataframe(pct_df, use_container_width=True, hide_index=True)

        # ── STRESS TESTING ───────────────────────────────────────────────
        st.divider()
        st.subheader("🔥 Pruebas de estrés históricas")
        st.caption("¿Qué le pasaría a tu portafolio actual si se repitiera una crisis del pasado?")

        ret_stress = st.session_state.returns_full if st.session_state.returns_full is not None else st.session_state.returns
        if ret_stress is not None:
            smin = ret_stress.index.min().strftime("%Y-%m-%d")
            smax = ret_stress.index.max().strftime("%Y-%m-%d")
            st.caption(f"📅 Datos disponibles: **{smin} → {smax}**")

        if st.button("▶️ Correr pruebas de estrés", use_container_width=True):
            bench_st = st.session_state.bench_full if st.session_state.bench_full is not None else (st.session_state.bench_rets if isinstance(st.session_state.bench_rets, dict) else {})
            primary = list(bench_st.values())[0] if bench_st else None
            with st.spinner("Analizando escenarios…"):
                stress = stress_test(wnorm, ret_stress, CRISIS_PERIODS, capital_inicial,
                                     {FICO_TICKER: FICO}, PPY, primary)
            st.session_state["stress_result"] = stress

        if "stress_result" in st.session_state and st.session_state["stress_result"]:
            stress = st.session_state["stress_result"]
            available = [s for s in stress if s.available]

            if not available:
                st.warning("Ningún escenario tiene datos en tu rango. "
                           "Intenta con más años de historia.")
            else:
                worst = min(available, key=lambda s: s.port_return)
                best = max(available, key=lambda s: s.port_return)
                beats = sum(1 for s in available if s.port_return > s.benchmark_return)
                st.info(f"💡 De {len(available)} crisis analizadas, la peor para tu portafolio "
                        f"es **{worst.name}** ({worst.port_return:+.2%}). "
                        f"Tu portafolio supera al benchmark en **{beats} de {len(available)}** escenarios.")

                for s in available:
                    icon = "🔴" if s.port_return < 0 else "🟢"
                    diff = s.port_return - s.benchmark_return
                    with st.container():
                        cA, cB = st.columns([3, 1])
                        with cA:
                            st.markdown(f"{icon} **{s.name}** · {s.start} → {s.end}")
                            st.caption(s.description)
                        with cB:
                            st.metric("Impacto", f"{s.port_return:+.2%}",
                                      delta=f"${s.port_loss:,.0f}", delta_color="off")
                        mejor = "mejor" if diff > 0 else "peor"
                        st.caption(f"→ Tu portafolio se comportó **{mejor}** que el benchmark "
                                   f"({s.port_return:+.2%} vs {s.benchmark_return:+.2%}). "
                                   f"Caída máxima en el período: {s.max_drawdown:.2%}.")
                        if not s.asset_returns.empty:
                            ar = s.asset_returns.sort_values()
                            st.caption(f"→ Más golpeado: **{ar.index[0]}** ({ar.iloc[0]:+.2%}). "
                                       f"Más resiliente: **{ar.index[-1]}** ({ar.iloc[-1]:+.2%}).")
                        st.divider()

            missing = [s for s in stress if not s.available]
            if missing:
                with st.expander(f"ℹ️ {len(missing)} escenarios fuera del rango de datos"):
                    for s in missing:
                        st.caption(f"**{s.name}** ({s.start} → {s.end}): {s.description}")
