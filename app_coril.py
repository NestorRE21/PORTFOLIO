# -*- coding: utf-8 -*-
"""Coril SAB — Optimizador BL v6 — UX directa"""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optimizer import RiskProfile, ForcedAsset, View, BLConfig, run_profile
from projections import monte_carlo, stress_test, CRISIS_PERIODS

# ══════════════════════════════ CONFIG ═════════════════════════════════════════
st.set_page_config(page_title="Coril · Portafolios", page_icon="📈", layout="wide")
RF, PPY = 0.02, 52
FICO_TK = "FICCMP13"
FICO = ForcedAsset(ret_annual=0.065, vol_annual=0.010, beta=0.30,
                   sector="Factoring", region="Perú", moneda="USD", instrumento="Fondo")
PERFILES = {"Conservador (30/70)":(0.30,0.70),"Moderado-bajo (40/60)":(0.40,0.60),
            "Moderado (50/50)":(0.50,0.50),"Crecimiento (60/40)":(0.60,0.40),
            "Agresivo (70/30)":(0.70,0.30)}
PERFIL_DESC = {"Conservador (30/70)":"Preservar capital.","Moderado-bajo (40/60)":"Leve crecimiento.",
               "Moderado (50/50)":"Balance.","Crecimiento (60/40)":"Mayor exposición.",
               "Agresivo (70/30)":"Máxima renta variable."}
EJEMPLO = ["AAPL","MSFT","NVDA","JNJ","KO","QQQ"]
C_RV,C_RF,C_OPT = "#2E5E8C","#2CA02C","#D6604D"
BMK_C = ["#888888","#E377C2","#FF7F0E","#9467BD","#17BECF"]

# ══════════════════════════════ STATE ══════════════════════════════════════════
for k,v in {"tickers":[],"benchmarks":["^GSPC"],"views":[],"optimized":False,
            "result":None,"manual_weights":None,"returns":None,"bench_rets":None,
            "betas":None,"sectors":None,"returns_full":None,"bench_full":None,
            "last_period":None,"data_range":""}.items():
    st.session_state.setdefault(k,v)

# ══════════════════════════════ BACKEND ════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=600)
def dl_equity(tickers, period="5y"):
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval="1wk", auto_adjust=True, progress=False)
    if raw is None or raw.empty: return None
    px = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":list(tickers)[0]})
    px = px.dropna(how="all").ffill(); px.index = pd.to_datetime(px.index).tz_localize(None)
    return np.log(px/px.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(how="all")

@st.cache_data(show_spinner=False, ttl=600)
def dl_bench(tks, period="5y"):
    import yfinance as yf
    out = {}
    for bk in tks:
        bk=bk.strip().upper()
        if not bk: continue
        try:
            raw=yf.download(bk,period=period,interval="1wk",auto_adjust=True,progress=False)
            if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
            pb=raw["Close"];
            if isinstance(pb,pd.DataFrame): pb=pb.iloc[:,0]
            pb.index=pd.to_datetime(pb.index).tz_localize(None)
            lr=np.log(pb/pb.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(); lr.name=bk; out[bk]=lr
        except: pass
    return out

def calc_betas(r,b):
    c=r.index.intersection(b.index); bv=b.loc[c].values; bvar=np.var(bv,ddof=1)
    return pd.Series({t:round(float(np.cov(r.loc[c,t].values[m:=np.isfinite(r.loc[c,t].values)&np.isfinite(bv)],bv[m],ddof=1)[0,1]/bvar),3) if (m:=np.isfinite(r.loc[c,t].values)&np.isfinite(bv)).sum()>10 and bvar>1e-12 else 1.0 for t in r.columns})

@st.cache_data(show_spinner=False, ttl=600)
def fetch_sectors(tickers):
    import yfinance as yf
    out={}
    for tk in tickers:
        try:
            i=yf.Ticker(tk).info or {}; s=i.get("sector","")
            if s: out[tk]=s
            elif i.get("quoteType")=="ETF": out[tk]=f"ETF · {i.get('category','') or i.get('longName',tk)[:30]}"
            else: out[tk]=i.get("industry","") or "Sin clasificar"
        except: out[tk]="Sin clasificar"
    return pd.Series(out)

@st.cache_data(show_spinner=False, ttl=300)
def search_yahoo(q):
    import requests
    try:
        r=requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                       params={"q":q,"quotesCount":6,"newsCount":0},
                       headers={"User-Agent":"Mozilla/5.0"},timeout=5)
        return [{"tk":x["symbol"],"name":x.get("shortname") or x.get("longname",""),
                 "type":x.get("quoteType",""),"ex":x.get("exchange","")}
                for x in r.json().get("quotes",[]) if x.get("symbol")]
    except: return []

def do_optimize(tickers,views_cfg,eq_t,fi_t,pb):
    betas=st.session_state.betas.copy(); betas[FICO_TK]=FICO.beta
    views=[]
    for v in views_cfg:
        if v["type"]=="absolute": views.append(View(kind="absolute",asset=v["asset"],q=v["q"],confidence=v["confidence"]))
        else: views.append(View(kind="relative",long=v["long"],short=v["short"],q=v["q"],confidence=v["confidence"]))
    ok=[t for t in tickers if t in st.session_state.returns.columns]
    return run_profile(returns=st.session_state.returns,equity_assets=ok,forced_assets={FICO_TK:FICO},
                       profile=RiskProfile.for_split(eq_t,fi_t),views=views,
                       config=BLConfig(rf_annual=RF,periods_per_year=PPY,tau=0.05,max_weight_equity=0.25,gamma_beta=5.0),
                       benchmark_returns=pb,betas=betas)

def wealth_dd(w,rets,bd,cap):
    if not bd or not isinstance(bd,dict): bd={}
    eq=[a for a in w.index if a in rets.columns and a!=FICO_TK]
    pr=pd.Series(0.0,index=rets.index)
    for c in eq: pr+=w.get(c,0)*rets[c].fillna(0)
    if FICO_TK in w.index and w[FICO_TK]>1e-8: pr+=w[FICO_TK]*(np.log(1+FICO.ret_annual)/PPY)
    pr=pr.fillna(0); common=pr.index
    for v in bd.values(): common=common.intersection(v.index)
    pr=pr.loc[common]; wl=np.exp(pr.cumsum())*cap; dd=wl/wl.cummax()-1
    bw,bdd={},{}
    for n,v in bd.items(): br=v.loc[common].fillna(0); bw[n]=np.exp(br.cumsum())*cap; bdd[n]=bw[n]/bw[n].cummax()-1
    return pr,wl,dd,bw,bdd

def calc_risk(rs):
    c=rs.dropna()
    if len(c)<10: return {"VaR":np.nan,"CVaR":np.nan}
    from scipy.stats import norm
    mu=c.mean()*PPY; sig=c.std(ddof=1)*np.sqrt(PPY); z=norm.ppf(0.05)
    return {"VaR":-(mu+z*sig),"CVaR":-(mu-sig*norm.pdf(z)/0.05)}

# ══════════════════════════ DESCARGA AUTO ═════════════════════════════════════
def run_download(period):
    """Descarga datos y guarda en session_state."""
    tks = st.session_state.tickers; bks = st.session_state.benchmarks
    if not tks or not bks: return False
    lr = dl_equity(tuple(tks), period=period)
    if lr is None or lr.empty: return False
    bd = dl_bench(tuple(bks), period=period)
    if not bd: return False
    common=lr.index
    for v in bd.values(): common=common.intersection(v.index)
    st.session_state.returns=lr.loc[common]
    st.session_state.bench_rets={k:v.loc[common] for k,v in bd.items()}
    st.session_state.returns_full=lr; st.session_state.bench_full=bd
    primary=list(bd.values())[0]
    betas=calc_betas(lr.loc[common],primary.loc[common]); betas[FICO_TK]=FICO.beta
    st.session_state.betas=betas
    ok=[t for t in tks if t in lr.columns]
    sec=fetch_sectors(tuple(ok)); sec[FICO_TK]=FICO.sector; st.session_state.sectors=sec
    st.session_state.last_period=period
    d1=lr.index.min().strftime("%Y-%m-%d"); d2=lr.index.max().strftime("%Y-%m-%d")
    st.session_state.data_range=f"{d1} → {d2}"
    # Limpiar resultados anteriores
    for k in list(st.session_state.keys()):
        if k.startswith("s_"): del st.session_state[k]
    st.session_state.optimized=False; st.session_state.result=None
    st.session_state.manual_weights=None
    if "mc" in st.session_state: del st.session_state["mc"]
    if "stress" in st.session_state: del st.session_state["stress"]
    return True

# ══════════════════════════════ SIDEBAR ════════════════════════════════════════
with st.sidebar:
    st.title("📈 Coril")
    perfil_sel=st.selectbox("Perfil",list(PERFILES.keys()),index=2)
    eq_t,fi_t=PERFILES[perfil_sel]
    st.caption(PERFIL_DESC[perfil_sel])
    c1,c2=st.columns(2); c1.metric("RV",f"{eq_t:.0%}"); c2.metric("RF",f"{fi_t:.0%}")
    st.divider()
    capital=st.slider("Inversión (USD)",1_000,1_000_000,100_000,1_000,format="$%d")
    period=st.selectbox("Historia",["1y","2y","3y","5y","10y","max"],index=3)
    with st.expander("⚙️ Avanzado"):
        st.caption(f"RF forzada: {FICO_TK} · {FICO.ret_annual:.2%}")
        _p=RiskProfile.for_split(eq_t,fi_t)
        st.caption(f"Beta: {_p.beta_min:.2f}–{_p.beta_max:.2f} · DD máx: {_p.max_drawdown:.0%}")
        if st.button("🗑️ Limpiar caché",use_container_width=True,
                     help="Borra datos guardados para forzar descarga fresca."):
            st.cache_data.clear(); st.toast("Caché limpiado ✓")

# ══════════════════════════ AUTO-DESCARGA ═════════════════════════════════════
# Si el período cambió y hay tickers+benchmarks, descargar automáticamente
if (st.session_state.tickers and st.session_state.benchmarks
        and st.session_state.last_period != period
        and st.session_state.last_period is not None):
    with st.spinner(f"Actualizando datos a {period}…"):
        run_download(period)

# ══════════════════════════════ HEADER ═════════════════════════════════════════
st.title("Optimizador de portafolios")
_hd=st.session_state.returns is not None
_ho=st.session_state.optimized and st.session_state.result is not None
st.caption(f"{'✅' if st.session_state.tickers else '1️⃣'} Activos → "
           f"{'✅' if _hd else '2️⃣'} Datos → "
           f"{'✅' if _ho else '3️⃣'} Portafolio → 4️⃣ Proyecciones")

tab1,tab2,tab3,tab4=st.tabs(["1 · Activos y datos","2 · Expectativas","3 · Portafolio","4 · Proyecciones"])

# ══════════════════════════ TAB 1: ACTIVOS + BENCHMARKS + DESCARGA ════════════
with tab1:
    # ── Buscador unificado ───────────────────────────────────────────────
    col_search, col_target = st.columns([3,1])
    with col_target:
        add_to = st.radio("Añadir como", ["🔵 Activo","📊 Benchmark"], horizontal=False,
                          label_visibility="visible")
    with col_search:
        q = st.text_input("🔍 Buscar por nombre o ticker",
                          placeholder="Visa, Apple, S&P 500, QQQ…")

    if q.strip():
        results = search_yahoo(q.strip())
        if results:
            # Resultados como botones directos — un clic para añadir
            cols = st.columns(min(len(results), 3))
            for i, r in enumerate(results):
                with cols[i % len(cols)]:
                    label = f"**{r['tk']}**\n{r['name'][:25]}\n_{r['type']} · {r['ex']}_"
                    if st.button(f"➕ {r['tk']} — {r['name'][:20]}", key=f"add_{r['tk']}",
                                 use_container_width=True):
                        tk = r['tk']
                        if add_to == "🔵 Activo":
                            if tk not in st.session_state.tickers:
                                st.session_state.tickers.append(tk)
                                st.toast(f"✓ {tk} añadido como activo")
                            else: st.toast(f"{tk} ya está en activos")
                        else:
                            if tk not in st.session_state.benchmarks:
                                st.session_state.benchmarks.append(tk)
                                st.toast(f"✓ {tk} añadido como benchmark")
                            else: st.toast(f"{tk} ya está en benchmarks")
        else:
            st.caption("Sin resultados. Prueba otro término.")

    if not st.session_state.tickers:
        st.divider()
        st.info("👋 **¿Primera vez?** Busca arriba o carga un ejemplo.")
        if st.button("🚀 Cargar ejemplo (6 activos US)", type="primary"):
            st.session_state.tickers = list(EJEMPLO)
            st.toast("Ejemplo cargado ✓")

    # ── Listas actuales ──────────────────────────────────────────────────
    st.divider()
    la, lb = st.columns(2)
    with la:
        st.write(f"**🔵 Activos ({len(st.session_state.tickers)})**")
        if st.session_state.tickers:
            for i,t in enumerate(st.session_state.tickers):
                c1,c2=st.columns([5,1])
                c1.write(t)
                if c2.button("✕",key=f"ra{i}"): st.session_state.tickers.pop(i); st.toast(f"{t} eliminado")
        else:
            st.caption("Ninguno aún.")

    with lb:
        st.write(f"**📊 Benchmarks ({len(st.session_state.benchmarks)})**")
        if st.session_state.benchmarks:
            for i,b in enumerate(st.session_state.benchmarks):
                c1,c2=st.columns([5,1])
                c1.write(b)
                if c2.button("✕",key=f"rb{i}"): st.session_state.benchmarks.pop(i); st.toast(f"{b} eliminado")
        else:
            st.caption("Ninguno aún.")

    # ── Descarga ─────────────────────────────────────────────────────────
    st.divider()
    can_dl = bool(st.session_state.tickers and st.session_state.benchmarks)
    if st.session_state.data_range:
        st.success(f"📦 Datos: {st.session_state.data_range} ({st.session_state.last_period})")

    if st.button("📥 Descargar datos", type="primary", use_container_width=True, disabled=not can_dl):
        with st.spinner("Descargando…"):
            ok = run_download(period)
        if ok: st.success(f"✅ Listo · {st.session_state.data_range}")
        else: st.error("Error al descargar. Verifica los tickers.")

# ══════════════════════════ TAB 2: VIEWS ══════════════════════════════════════
with tab2:
    _hd2 = st.session_state.returns is not None
    if not _hd2:
        st.info("⬅️ Descarga datos primero.")
    else:
        st.subheader("Expectativas del analista (opcional)")
        st.caption("¿Tienes una opinión sobre algún activo? Si no, déjalo vacío.")
        vt=st.radio("Tipo",["Retorno de un activo","Un activo vs otro"],horizontal=True)
        if vt=="Retorno de un activo":
            c1,c2,c3=st.columns([3,2,2])
            va=c1.selectbox("Activo",st.session_state.tickers,key="va")
            vq=c2.number_input("Retorno anual",value=0.10,step=0.01,format="%.2f",key="vq",help="0.10 = 10%")
            vc=c3.slider("Confianza",0.1,1.0,0.5,0.1,key="vc",help="1=seguro, 0.1=intuición")
            if st.button("Añadir"):
                st.session_state.views.append({"type":"absolute","asset":va,"q":float(vq),"confidence":float(vc)})
                st.toast("Expectativa añadida ✓")
        else:
            c1,c2,c3,c4=st.columns(4)
            vl=c1.selectbox("Ganador",st.session_state.tickers,key="vl")
            vs=c2.selectbox("Perdedor",st.session_state.tickers,key="vs")
            vq=c3.number_input("Diferencia",value=0.05,step=0.01,format="%.2f",key="vqr")
            vc=c4.slider("Confianza",0.1,1.0,0.5,0.1,key="vcr")
            if st.button("Añadir"):
                if vl==vs: st.warning("Deben ser distintos.")
                else:
                    st.session_state.views.append({"type":"relative","long":vl,"short":vs,"q":float(vq),"confidence":float(vc)})
                    st.toast("Expectativa añadida ✓")
        if st.session_state.views:
            st.divider()
            for i,v in enumerate(st.session_state.views):
                a,b=st.columns([6,1])
                txt=f"📌 **{v['asset']}** → {v['q']:.0%}" if v["type"]=="absolute" else f"📌 **{v['long']}** > **{v['short']}** por {v['q']:.0%}"
                a.write(txt+f" (confianza {v['confidence']:.0%})")
                if b.button("✕",key=f"rv{i}"): st.session_state.views.pop(i); st.toast("Eliminada")

# ══════════════════════════ TAB 3: PORTAFOLIO ═════════════════════════════════
with tab3:
    if st.session_state.returns is None:
        st.info("⬅️ Descarga datos primero.")
    else:
        st.subheader("Optimizar")
        st.caption(f"**{perfil_sel}** · ${capital:,.0f}")
        if st.button("🔄 Optimizar",type="primary",use_container_width=True):
            pb=list(st.session_state.bench_rets.values())[0]
            with st.spinner("Calculando…"):
                r=do_optimize(st.session_state.tickers,st.session_state.views,eq_t,fi_t,pb)
            for k in list(st.session_state.keys()):
                if k.startswith("s_"): del st.session_state[k]
            st.session_state.result=r; st.session_state.manual_weights=r.weights.copy()
            st.session_state.optimized=True
            if r.feasible: st.success("✅ Optimizado")
            else: st.warning(f"⚠️ {r.feasibility_report}")

        if st.session_state.optimized and st.session_state.result is not None:
            res=st.session_state.result

            # ── Pesos editables ──────────────────────────────────────────
            st.divider()
            st.subheader("Pesos (%)")
            st.caption("Cambia porcentajes. Métricas se actualizan automáticamente.")
            cs,csum=st.columns([3,2])
            with cs:
                nw={}
                for a in res.weights.index:
                    ef=a==FICO_TK; ic="🟢 RF" if ef else "🔵 RV"
                    nw[a]=st.number_input(f"{a} · {ic}",0.0,100.0,
                        round(float(res.weights[a])*100,1),0.5,"%.1f",key=f"s_{a}")
                wn_pct=pd.Series(nw); tot=wn_pct.sum()
                wnorm=wn_pct/tot if tot>0 else wn_pct/100
                st.session_state.manual_weights=wnorm
                if abs(tot-100)<0.1: st.success(f"✅ Suma: {tot:.1f}%")
                else: st.warning(f"⚠️ Suma: {tot:.1f}% → normalizado")
            with csum:
                st.markdown("**Peso final:**")
                for a in wnorm.index:
                    ic="🟢" if a==FICO_TK else "🔵"
                    st.write(f"{ic} {a}: **{wnorm[a]:.1%}**")
                st.divider()
                eqw=float(wnorm[[a for a in wnorm.index if a!=FICO_TK]].sum())
                fiw=float(wnorm.get(FICO_TK,0))
                st.metric("RV",f"{eqw:.1%}",delta=f"{eqw-eq_t:+.1%} vs obj")
                st.metric("RF",f"{fiw:.1%}",delta=f"{fiw-fi_t:+.1%} vs obj")

            # ── Métricas dinámicas ───────────────────────────────────────
            st.divider()
            w_np=wnorm.reindex(res.bl_returns.index).fillna(0).to_numpy()
            mu_np=res.bl_returns.to_numpy(); S_np=res.cov_matrix.to_numpy()
            b_np=st.session_state.betas.reindex(res.bl_returns.index).fillna(1).to_numpy()
            p_ret=float(w_np@mu_np); p_vol=float(np.sqrt(max(w_np@S_np@w_np,1e-10)))
            p_sh=(p_ret-RF)/p_vol if p_vol>1e-10 else 0; p_bt=float(w_np@b_np)
            st.subheader("Métricas del portafolio")
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Retorno",f"{p_ret:.2%}",help="Pesos actuales × retornos BL.")
            m2.metric("Riesgo",f"{p_vol:.2%}",help="Volatilidad anualizada.")
            m3.metric("Sharpe",f"{p_sh:.2f}",help="Retorno/Riesgo.")
            m4.metric("Beta",f"{p_bt:.2f}",help="Sensibilidad al mercado.")

            # ── Gráficos ─────────────────────────────────────────────────
            st.divider()
            g1,g2,g3=st.columns(3)
            with g1:
                ws=wnorm[wnorm>1e-4]
                fig=go.Figure(go.Bar(x=ws.values,y=ws.index,orientation="h",
                    marker_color=[C_RF if a==FICO_TK else C_RV for a in ws.index]))
                fig.update_layout(height=260,margin=dict(l=0,r=0,t=5,b=0),xaxis_tickformat=".0%")
                st.plotly_chart(fig,use_container_width=True)
            with g2:
                fig=go.Figure(go.Pie(labels=["RV","RF"],values=[eqw,fiw],
                    marker_colors=[C_RV,C_RF],hole=.5))
                fig.update_layout(height=260,margin=dict(l=0,r=0,t=5,b=0))
                st.plotly_chart(fig,use_container_width=True)
            with g3:
                sec=st.session_state.sectors
                if sec is not None and not sec.empty:
                    sw={}
                    for a in wnorm.index:
                        if wnorm[a]>1e-4: s=sec.get(a,"?"); sw[s]=sw.get(s,0)+wnorm[a]
                    fig=go.Figure(go.Pie(labels=list(sw.keys()),values=list(sw.values()),hole=.5))
                    fig.update_layout(height=260,margin=dict(l=0,r=0,t=5,b=0))
                    st.plotly_chart(fig,use_container_width=True)

            # ── Evolución ────────────────────────────────────────────────
            st.divider()
            st.subheader(f"Evolución histórica (${capital:,.0f})")
            pr,wl,dd,bw,bdd=wealth_dd(wnorm,st.session_state.returns,st.session_state.bench_rets,capital)
            fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.7,.3],vertical_spacing=.05)
            fig.add_trace(go.Scatter(x=wl.index,y=wl.values,name="Portafolio",line=dict(color=C_RV,width=2.5)),row=1,col=1)
            for i,(n,v) in enumerate(bw.items()):
                fig.add_trace(go.Scatter(x=v.index,y=v.values,name=n,line=dict(color=BMK_C[i%len(BMK_C)],dash="dash")),row=1,col=1)
            fig.add_trace(go.Scatter(x=dd.index,y=dd.values,name="Drawdown",fill="tozeroy",line=dict(color=C_OPT)),row=2,col=1)
            fig.update_yaxes(tickprefix="$",tickformat=",.0f",row=1,col=1)
            fig.update_yaxes(tickformat=".0%",row=2,col=1)
            fig.update_layout(height=420,margin=dict(l=0,r=0,t=10,b=0),legend=dict(orientation="h",y=1.08))
            st.plotly_chart(fig,use_container_width=True)

            ann_r=np.exp(pr.mean()*PPY)-1; ann_v=pr.std(ddof=1)*np.sqrt(PPY); risk=calc_risk(pr)
            h1,h2,h3,h4,h5=st.columns(5)
            h1.metric("Ret. hist.",f"{ann_r:.2%}"); h2.metric("Vol.",f"{ann_v:.2%}")
            h3.metric("Peor caída",f"{dd.min():.2%}")
            h4.metric("VaR 95%",f"{risk['VaR']:.2%}"); h5.metric("CVaR 95%",f"{risk['CVaR']:.2%}")

# ══════════════════════════ TAB 4: PROYECCIONES ═══════════════════════════════
with tab4:
    if not (st.session_state.optimized and st.session_state.result is not None):
        st.info("⬅️ Optimiza primero.")
    else:
        res=st.session_state.result
        wnorm=st.session_state.manual_weights if st.session_state.manual_weights is not None else res.weights

        st.subheader("🎲 Monte Carlo")
        st.caption("Simula miles de futuros posibles.")
        c1,c2,c3=st.columns(3)
        mh=c1.selectbox("Horizonte",[1,2,3,5,10],index=2,format_func=lambda x:f"{x} año{'s' if x>1 else ''}")
        mn=c2.selectbox("Simulaciones",[1000,5000,10000],index=1)
        mt=c3.number_input("Objetivo",value=int(capital*1.2),step=10_000,format="%d")
        if st.button("▶️ Simular",type="primary",use_container_width=True):
            with st.spinner(f"Simulando {mn:,} futuros…"):
                mc=monte_carlo(wnorm,res.bl_returns,res.cov_matrix,capital,mh,PPY,mn,mt)
            st.session_state["mc"]=mc
        if "mc" in st.session_state and st.session_state["mc"] is not None:
            mc=st.session_state["mc"]
            m1,m2,m3,m4,m5=st.columns(5)
            m1.metric("Proyectado",f"${mc.median_path[-1]:,.0f}",help="Mediana.")
            m2.metric("P(pérdida)",f"{mc.prob_loss:.1%}")
            m3.metric(f"P(≥${mc.target:,.0f})",f"{mc.prob_target:.1%}")
            m4.metric("VaR term.",f"${mc.var_terminal:,.0f}"); m5.metric("CVaR term.",f"${mc.cvar_terminal:,.0f}")
            st.info(f"💡 Escenario central: **${mc.median_path[-1]:,.0f}** en {mc.horizon_years:.0f} año(s). "
                    f"Prob. pérdida: **{mc.prob_loss:.1%}**. Prob. objetivo: **{mc.prob_target:.1%}**.")
            fig=go.Figure(); x=mc.dates
            for lo,hi,cl in [(5,95,"rgba(46,94,140,0.08)"),(10,90,"rgba(46,94,140,0.12)"),(25,75,"rgba(46,94,140,0.18)")]:
                fig.add_trace(go.Scatter(x=list(x)+list(x[::-1]),y=list(mc.percentiles[hi])+list(mc.percentiles[lo][::-1]),
                    fill="toself",fillcolor=cl,line=dict(width=0),name=f"P{lo}–P{hi}"))
            fig.add_trace(go.Scatter(x=x,y=mc.median_path,name="Mediana",line=dict(color=C_RV,width=2.5)))
            fig.add_hline(y=mc.capital,line_dash="dot",line_color="gray",annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target!=mc.capital: fig.add_hline(y=mc.target,line_dash="dot",line_color=C_RF,annotation_text=f"Objetivo ${mc.target:,.0f}")
            fig.update_yaxes(tickprefix="$",tickformat=",.0f")
            fig.update_layout(height=400,margin=dict(l=0,r=0,t=10,b=0),legend=dict(orientation="h",y=-0.1))
            st.plotly_chart(fig,use_container_width=True)
            with st.expander("📊 Detalle por escenario"):
                pcts=[5,10,25,50,75,90,95]; cv=[mc.percentiles[p][-1] for p in pcts]
                pct_df=pd.DataFrame({
                    "Escenario":[f"Pesimista (P5)",f"Conservador (P10)",f"Probable bajo (P25)",
                                 f"Central (P50)",f"Probable alto (P75)",f"Optimista (P90)",f"Mejor caso (P95)"],
                    "Capital final":[f"${v:,.0f}" for v in cv],
                    "Retorno":[f"{v/mc.capital-1:+.1%}" for v in cv],
                    "Ganancia/Pérdida":[f"${v-mc.capital:+,.0f}" for v in cv],
                    "Interpretación":[
                        "Solo 5% termina peor. Pérdida máxima probable.",
                        "90% de probabilidad de superar este resultado.",
                        "75% de probabilidad de superar este resultado.",
                        "Escenario más probable (mitad arriba, mitad abajo).",
                        "Solo 25% supera este resultado.",
                        "Solo 10% es mejor.",
                        "Casi el mejor caso posible (top 5%)."]})
                st.dataframe(pct_df,use_container_width=True,hide_index=True)
                st.caption("💡 El percentil (P) indica qué % de escenarios termina por debajo de ese valor.")

        # ── STRESS ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔥 Pruebas de estrés")
        st.caption("¿Qué pasaría si se repitiera una crisis?")
        ret_st=st.session_state.returns_full if st.session_state.returns_full is not None else st.session_state.returns
        if ret_st is not None:
            st.caption(f"📅 Datos: {ret_st.index.min().strftime('%Y-%m-%d')} → {ret_st.index.max().strftime('%Y-%m-%d')}")
        bench_st=st.session_state.bench_full if st.session_state.bench_full is not None else (st.session_state.bench_rets if isinstance(st.session_state.bench_rets,dict) else {})
        if st.button("▶️ Correr estrés",use_container_width=True):
            pb=list(bench_st.values())[0] if bench_st else None
            with st.spinner("Analizando…"):
                stres=stress_test(wnorm,ret_st,CRISIS_PERIODS,capital,{FICO_TK:FICO},PPY,pb)
            st.session_state["stress"]=stres
        if "stress" in st.session_state and st.session_state["stress"]:
            stres=st.session_state["stress"]; avail=[s for s in stres if s.available]
            if not avail: st.warning("Sin datos. Usa más años de historia.")
            else:
                worst=min(avail,key=lambda s:s.port_return); beats=sum(1 for s in avail if s.port_return>s.benchmark_return)
                st.info(f"💡 Peor: **{worst.name}** ({worst.port_return:+.2%}). "
                        f"Superas benchmark en **{beats}/{len(avail)}**.")
                fig=go.Figure()
                fig.add_trace(go.Bar(x=[s.name for s in avail],y=[s.port_return for s in avail],name="Portafolio",marker_color=C_OPT))
                fig.add_trace(go.Bar(x=[s.name for s in avail],y=[s.benchmark_return for s in avail],name="Benchmark",marker_color="#888"))
                fig.update_yaxes(tickformat=".1%")
                fig.update_layout(barmode="group",height=320,margin=dict(l=0,r=0,t=10,b=0),legend=dict(orientation="h",y=1.08))
                st.plotly_chart(fig,use_container_width=True)
                for s in avail:
                    ic="🔴" if s.port_return<0 else "🟢"; diff=s.port_return-s.benchmark_return
                    cA,cB=st.columns([3,1])
                    with cA: st.markdown(f"{ic} **{s.name}** · {s.start} → {s.end}"); st.caption(s.description)
                    with cB: st.metric("Impacto",f"{s.port_return:+.2%}",delta=f"${s.port_loss:,.0f}",delta_color="off")
                    mejor="mejor" if diff>0 else "peor"
                    st.caption(f"→ **{mejor.upper()}** que benchmark ({s.port_return:+.2%} vs {s.benchmark_return:+.2%}). Drawdown: {s.max_drawdown:.2%}.")
                    if not s.asset_returns.empty:
                        ar=s.asset_returns.sort_values()
                        st.caption(f"→ Más golpeado: **{ar.index[0]}** ({ar.iloc[0]:+.2%}). Más resiliente: **{ar.index[-1]}** ({ar.iloc[-1]:+.2%}).")
                    st.divider()
            missing=[s for s in stres if not s.available]
            if missing:
                with st.expander(f"ℹ️ {len(missing)} crisis fuera del rango"):
                    for s in missing: st.caption(f"**{s.name}** ({s.start}→{s.end}): {s.description}")
