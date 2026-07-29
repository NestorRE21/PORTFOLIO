# -*- coding: utf-8 -*-
"""Coril SAB — Optimizador BL v7 — Compacto"""
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optimizer import RiskProfile, ForcedAsset, View, BLConfig, run_profile
from projections import monte_carlo, stress_test, CRISIS_PERIODS

st.set_page_config(page_title="Coril · Portafolios", page_icon="📈", layout="wide")
RF,PPY = 0.02,52
FICO_TK = "FICCMP13"
FICO = ForcedAsset(ret_annual=0.0625,vol_annual=0.010,beta=0.30,sector="Factoring",region="Perú",moneda="USD",instrumento="Fondo")
PERFILES = {"Conservador (30/70)":(0.30,0.70),"Moderado-bajo (40/60)":(0.40,0.60),
            "Moderado (50/50)":(0.50,0.50),"Crecimiento (60/40)":(0.60,0.40),"Agresivo (70/30)":(0.70,0.30)}
P_DESC = {"Conservador (30/70)":"Preservar capital.","Moderado-bajo (40/60)":"Leve crecimiento.",
          "Moderado (50/50)":"Balance.","Crecimiento (60/40)":"Mayor exposición.","Agresivo (70/30)":"Máxima RV."}
EJ = ["AAPL","MSFT","NVDA","JNJ","KO","QQQ"]
C_RV,C_RF,C_OPT = "#2E5E8C","#2CA02C","#D6604D"
BC = ["#888","#E377C2","#FF7F0E","#9467BD","#17BECF"]

for k,v in {"tickers":[],"rf_tickers":[],"include_fico":True,"benchmarks":["^GSPC"],"views":[],"optimized":False,"result":None,
            "manual_weights":None,"returns":None,"bench_rets":None,"betas":None,"sectors":None,
            "returns_full":None,"bench_full":None,"last_period":None,"data_range":""}.items():
    st.session_state.setdefault(k,v)

# ═══════════════════ BACKEND ══════════════════════════════════════════════════
def _yf_period(period):
    """Convierte períodos custom (como 15y) a parámetros de yfinance."""
    if period in ["1y","2y","3y","5y","10y","max","ytd"]:
        return {"period": period}
    # Períodos custom: extraer años y calcular fecha de inicio
    if period.endswith("y"):
        years = int(period.replace("y",""))
        start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        return {"start": start}
    return {"period": period}

@st.cache_data(show_spinner=False,ttl=600)
def dl_eq(tickers,period="15y"):
    import yfinance as yf
    params = _yf_period(period)
    raw=yf.download(tickers,**params,interval="1wk",auto_adjust=True,progress=False)
    if raw is None or raw.empty: return None
    px=raw["Close"].copy() if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":list(tickers)[0]})
    px=px.dropna(how="all").ffill(); px.index=pd.to_datetime(px.index).tz_localize(None)
    return np.log(px/px.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(how="all")

@st.cache_data(show_spinner=False,ttl=600)
def dl_bk(tks,period="15y"):
    import yfinance as yf
    out={}
    for b in tks:
        b=b.strip().upper()
        if not b: continue
        try:
            params = _yf_period(period)
            raw=yf.download(b,**params,interval="1wk",auto_adjust=True,progress=False)
            if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
            p=raw["Close"]; 
            if isinstance(p,pd.DataFrame): p=p.iloc[:,0]
            p.index=pd.to_datetime(p.index).tz_localize(None)
            lr=np.log(p/p.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(); lr.name=b; out[b]=lr
        except: pass
    return out

def calc_betas(r,b):
    c=r.index.intersection(b.index); bv=b.loc[c].values; bvar=np.var(bv,ddof=1)
    out={}
    for t in r.columns:
        tv=r.loc[c,t].values; m=np.isfinite(tv)&np.isfinite(bv)
        out[t]=round(float(np.cov(tv[m],bv[m],ddof=1)[0,1]/bvar),3) if m.sum()>10 and bvar>1e-12 else 1.0
    return pd.Series(out)

@st.cache_data(show_spinner=False,ttl=600)
def fetch_sec(tickers):
    import yfinance as yf
    out={}
    for t in tickers:
        try:
            i=yf.Ticker(t).info or {}; s=i.get("sector","")
            out[t]=s if s else (f"ETF · {i.get('category','')[:25]}" if i.get("quoteType")=="ETF" else i.get("industry","") or "–")
        except: out[t]="–"
    return pd.Series(out)

@st.cache_data(show_spinner=False,ttl=300)
def search_yf(q):
    import requests
    try:
        r=requests.get("https://query2.finance.yahoo.com/v1/finance/search",params={"q":q,"quotesCount":12,"newsCount":0},
                       headers={"User-Agent":"Mozilla/5.0"},timeout=5)
        return [{"tk":x["symbol"],"nm":x.get("shortname") or x.get("longname",""),
                 "tp":x.get("quoteType",""),"ex":x.get("exchange","")}
                for x in r.json().get("quotes",[]) if x.get("symbol")]
    except: return []

# Keywords en nombre que indican renta fija
_RF_KW = {"bond","treasury","income","fixed","aggregate","debt","govt","municipal",
          "corporate bond","tips","tbill","bill","note","yield","interest rate",
          "money market","short term","intermediate","long term","investment grade",
          "high yield","credit","inflation","sovereign","tlt","shy","ief","agg",
          "bnd","lqd","hyg","tip","bil","shv","govt","mbb","vcsh","vcit","vclt",
          "biv","bsv","blv","vgsh","vgit","vglt","scho","schr","schz","spab","usig",
          "igsb","igib","flot","stip","ltpz","zroz","edv","splb","sptl","spts",
          "bono","renta fija","deuda","tesor"}

def _is_rf_candidate(r):
    """Heurística: ¿el resultado parece renta fija?"""
    nm = (r.get("nm","") or "").lower()
    tk = (r.get("tk","") or "").upper()
    # ETFs/fondos con keywords de RF en nombre
    if any(kw in nm for kw in _RF_KW): return True
    if tk.lower() in _RF_KW: return True
    return False

def _is_rv_candidate(r):
    """Heurística: ¿el resultado parece renta variable?"""
    tp = r.get("tp","")
    # Acciones individuales → siempre RV
    if tp == "EQUITY": return True
    # ETFs/fondos → solo si NO parece RF
    if tp in ("ETF","MUTUALFUND"):
        return not _is_rf_candidate(r)
    # Índices → depende del contexto, los dejamos pasar
    if tp == "INDEX": return True
    return True  # por defecto permitir

def filter_search(results, category):
    """Filtra resultados de búsqueda según la categoría seleccionada."""
    if category == "🔵 Renta variable":
        return [r for r in results if _is_rv_candidate(r)]
    elif category == "🟢 Renta fija":
        return [r for r in results if r.get("tp") in ("ETF","MUTUALFUND","INDEX") or _is_rf_candidate(r)]
    else:  # Benchmark — no filtrar
        return results

def do_opt(eq_tickers, rf_tickers, include_fico, views_cfg, eq_t, fi_t, pb):
    betas=st.session_state.betas.copy()
    forced = {FICO_TK: FICO} if include_fico else {}
    if include_fico: betas[FICO_TK]=FICO.beta
    all_assets = set(eq_tickers) | set(rf_tickers) | (set(forced.keys()) if forced else set())
    views=[View(kind="absolute",asset=v["asset"],q=v["q"],confidence=v["confidence"]) if v["type"]=="absolute"
           else View(kind="relative",long=v["long"],short=v["short"],q=v["q"],confidence=v["confidence"])
           for v in views_cfg if (v["type"]=="absolute" and v.get("asset") in all_assets) or
                                  (v["type"]!="absolute" and v.get("long") in all_assets and v.get("short") in all_assets)]
    return run_profile(returns=st.session_state.returns,equity_assets=eq_tickers,
                       forced_assets=forced, rf_assets=rf_tickers,
                       profile=RiskProfile.for_split(eq_t,fi_t),views=views,
                       config=BLConfig(rf_annual=RF,periods_per_year=PPY,tau=0.05,max_weight_equity=0.25,gamma_beta=5.0),
                       benchmark_returns=pb,betas=betas)

def wdd(w,rets,bd,cap):
    if not isinstance(bd,dict): bd={}
    eq=[a for a in w.index if a in rets.columns and a!=FICO_TK]
    pr=sum(w.get(c,0)*rets[c].fillna(0) for c in eq) if eq else pd.Series(0,index=rets.index)
    if FICO_TK in w.index and w[FICO_TK]>1e-8: pr=pr+w[FICO_TK]*(np.log(1+FICO.ret_annual)/PPY)
    pr=pr.fillna(0); common=pr.index
    for v in bd.values(): common=common.intersection(v.index)
    pr=pr.loc[common]; wl=np.exp(pr.cumsum())*cap; dd=wl/wl.cummax()-1
    bw,bdd={},{}
    for n,v in bd.items():
        br=v.loc[common].fillna(0); bw[n]=np.exp(br.cumsum())*cap; bdd[n]=bw[n]/bw[n].cummax()-1
    return pr,wl,dd,bw,bdd

def run_dl(period):
    tks=st.session_state.tickers; bks=st.session_state.benchmarks
    rf_tks=st.session_state.rf_tickers
    if not tks or not bks: return False
    # Descargar equity + RF de mercado juntos
    all_market = list(set(tks + rf_tks))  # sin duplicados
    lr=dl_eq(tuple(all_market),period)
    if lr is None or lr.empty: return False
    bd=dl_bk(tuple(bks),period)
    if not bd: return False
    common=lr.index
    for v in bd.values(): common=common.intersection(v.index)
    st.session_state.returns=lr.loc[common]; st.session_state.bench_rets={k:v.loc[common] for k,v in bd.items()}
    st.session_state.returns_full=lr; st.session_state.bench_full=bd
    b=calc_betas(lr.loc[common],list(bd.values())[0].loc[common]); b[FICO_TK]=FICO.beta; st.session_state.betas=b
    ok=[t for t in tks if t in lr.columns]
    s=fetch_sec(tuple(ok)); s[FICO_TK]=FICO.sector; st.session_state.sectors=s
    st.session_state.last_period=period
    st.session_state.data_range=f"{lr.index.min().strftime('%Y-%m-%d')} → {lr.index.max().strftime('%Y-%m-%d')}"
    for k in list(st.session_state.keys()):
        if k.startswith("s_"): del st.session_state[k]
    st.session_state.optimized=False; st.session_state.result=None; st.session_state.manual_weights=None
    for x in ["mc","stress"]:
        if x in st.session_state: del st.session_state[x]
    return True

# ═══════════════════ SIDEBAR ══════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 Coril")
    ps=st.selectbox("Perfil",list(PERFILES.keys()),index=2); eq_t,fi_t=PERFILES[ps]
    st.caption(P_DESC[ps])
    c1,c2=st.columns(2); c1.metric("RV",f"{eq_t:.0%}"); c2.metric("RF",f"{fi_t:.0%}")
    st.divider()
    capital=st.slider("Inversión (USD)",1_000,1_000_000,100_000,1_000,format="$%d")
    chart_years=st.selectbox("Ver gráfico desde hace",["1y","2y","3y","5y","10y","15y"],index=3,
                             help="Solo afecta la vista del gráfico histórico. La optimización siempre usa 15 años.")
    with st.expander("⚙️ Avanzado"):
        _p=RiskProfile.for_split(eq_t,fi_t)
        st.caption(f"RF: {FICO_TK} · {FICO.ret_annual:.2%} | Beta: {_p.beta_min:.2f}–{_p.beta_max:.2f} | DD máx: {_p.max_drawdown:.0%}")
        st.caption("Optimización siempre usa **15 años** de datos.")
        if st.button("🗑️ Limpiar caché",use_container_width=True): st.cache_data.clear(); st.toast("✓")

# Descargar siempre con 15 años (fijo)
OPT_PERIOD = "15y"

# Auto-descarga si no hay datos aún (primera vez tras añadir tickers)
if st.session_state.tickers and st.session_state.benchmarks and st.session_state.last_period and st.session_state.last_period!=OPT_PERIOD:
    with st.spinner("Actualizando datos (15y)…"): run_dl(OPT_PERIOD)

# ═══════════════════ MAIN ═════════════════════════════════════════════════════
st.title("Optimizador de portafolios")
tab1,tab2,tab3,tab4=st.tabs(["1 · Activos","2 · Expectativas","3 · Portafolio","4 · Proyecciones"])

# ═══════════════════ TAB 1 ════════════════════════════════════════════════════
with tab1:
    col_s,col_t=st.columns([4,1])
    with col_t: add_to=st.radio("Añadir como",["🔵 Renta variable","🟢 Renta fija","📊 Benchmark"])
    with col_s: q=st.text_input("🔍 Buscar",placeholder="Apple, TLT, SHY, AGG, ^GSPC…")
    if q.strip():
        raw_res=search_yf(q.strip())
        res=filter_search(raw_res, add_to)
        if not res and raw_res:
            st.caption(f"ℹ️ No se encontraron resultados compatibles con **{add_to}**. "
                       f"Se encontraron {len(raw_res)} de otra clase.")
        if res:
            cols=st.columns(min(len(res[:6]),3))
            for i,r in enumerate(res[:6]):
                with cols[i%len(cols)]:
                    if st.button(f"➕ {r['tk']} — {r['nm'][:18]}",key=f"a_{r['tk']}",use_container_width=True):
                        tk=r['tk']
                        if add_to=="🔵 Renta variable":
                            if tk not in st.session_state.tickers: st.session_state.tickers.append(tk); st.toast(f"✓ {tk} → RV")
                        elif add_to=="🟢 Renta fija":
                            if tk not in st.session_state.rf_tickers: st.session_state.rf_tickers.append(tk); st.toast(f"✓ {tk} → RF")
                        else:
                            if tk not in st.session_state.benchmarks: st.session_state.benchmarks.append(tk); st.toast(f"✓ {tk} → Benchmark")
    if not st.session_state.tickers:
        if st.button("🚀 Cargar ejemplo",type="primary"):
            st.session_state.tickers=list(EJ); st.session_state.views=[]; st.session_state.rf_tickers=[]

    # ── Listas: RV + RF + Benchmarks ─────────────────────────────────────
    la,lb,lc=st.columns(3)
    with la:
        st.caption(f"**🔵 Renta variable ({len(st.session_state.tickers)})**")
        for i,t in enumerate(st.session_state.tickers):
            c1,c2=st.columns([5,1]); c1.write(t)
            if c2.button("✕",key=f"ra{i}"):
                rm=st.session_state.tickers.pop(i)
                st.session_state.views=[v for v in st.session_state.views if v.get("asset")!=rm and v.get("long")!=rm and v.get("short")!=rm]
    with lb:
        st.caption(f"**🟢 Renta fija ({len(st.session_state.rf_tickers)})**")
        # Toggle FICO
        include_fico = st.checkbox("Incluir FICO Coril (6.25%)", value=True, key="fico_toggle")
        st.session_state.include_fico = include_fico
        if include_fico:
            st.caption(f"✓ {FICO_TK} · {FICO.ret_annual:.2%} forzado")
        for i,t in enumerate(st.session_state.rf_tickers):
            c1,c2=st.columns([5,1]); c1.write(t)
            if c2.button("✕",key=f"rrf{i}"): st.session_state.rf_tickers.pop(i)
        if not st.session_state.rf_tickers and not include_fico:
            st.warning("Sin activos de renta fija.")
    with lc:
        st.caption(f"**📊 Benchmarks ({len(st.session_state.benchmarks)})**")
        for i,b in enumerate(st.session_state.benchmarks):
            c1,c2=st.columns([5,1]); c1.write(b)
            if c2.button("✕",key=f"rb{i}"): st.session_state.benchmarks.pop(i)

    # ── Descarga ─────────────────────────────────────────────────────────
    has_rf = bool(st.session_state.rf_tickers) or st.session_state.get("include_fico", True)
    can_dl = bool(st.session_state.tickers and st.session_state.benchmarks and has_rf)
    if st.session_state.data_range: st.success(f"📦 {st.session_state.data_range} ({st.session_state.last_period})")
    if st.button("📥 Descargar datos",type="primary",use_container_width=True, disabled=not can_dl):
        with st.spinner("Descargando…"):
            if run_dl(OPT_PERIOD): st.success(f"✅ {st.session_state.data_range}")
            else: st.error("Error. Verifica tickers.")

# ═══════════════════ TAB 2 ════════════════════════════════════════════════════
with tab2:
    if st.session_state.returns is None: st.info("⬅️ Descarga datos primero.")
    else:
        st.caption("Opcional: añade tus expectativas sobre algún activo.")
        vt=st.radio("",["Retorno de un activo","Un activo vs otro"],horizontal=True,label_visibility="collapsed")
        if vt=="Retorno de un activo":
            c1,c2,c3=st.columns([3,2,2])
            va=c1.selectbox("Activo",st.session_state.tickers,key="va")
            vq=c2.number_input("Ret. anual",value=0.10,step=0.01,format="%.2f",key="vq")
            vc=c3.slider("Confianza",0.1,1.0,0.5,0.1,key="vc")
            if st.button("Añadir"): st.session_state.views.append({"type":"absolute","asset":va,"q":float(vq),"confidence":float(vc)})
        else:
            c1,c2,c3,c4=st.columns(4)
            vl=c1.selectbox("Ganador",st.session_state.tickers,key="vl"); vs=c2.selectbox("Perdedor",st.session_state.tickers,key="vs")
            vq=c3.number_input("Dif.",value=0.05,step=0.01,format="%.2f",key="vqr"); vc=c4.slider("Conf.",0.1,1.0,0.5,0.1,key="vcr")
            if st.button("Añadir"):
                if vl!=vs: st.session_state.views.append({"type":"relative","long":vl,"short":vs,"q":float(vq),"confidence":float(vc)})
        for i,v in enumerate(st.session_state.views):
            c1,c2=st.columns([6,1])
            if v["type"]=="absolute":
                c1.caption(f"📌 {v['asset']} → {v['q']:.0%} (conf. {v['confidence']:.0%})")
            else:
                c1.caption(f"📌 {v['long']} > {v['short']} por {v['q']:.0%} (conf. {v['confidence']:.0%})")
            if c2.button("✕",key=f"rv{i}"): st.session_state.views.pop(i)

# ═══════════════════ TAB 3 ════════════════════════════════════════════════════
with tab3:
    if st.session_state.returns is None: st.info("⬅️ Descarga datos primero.")
    else:
        if st.button("🔄 Optimizar",type="primary",use_container_width=True):
            pb=list(st.session_state.bench_rets.values())[0]
            with st.spinner("Calculando…"):
                r=do_opt(st.session_state.tickers, st.session_state.rf_tickers,
                         st.session_state.get("include_fico",True),
                         st.session_state.views, eq_t, fi_t, pb)
            for k in list(st.session_state.keys()):
                if k.startswith("s_"): del st.session_state[k]
            st.session_state.result=r; st.session_state.manual_weights=r.weights.copy(); st.session_state.optimized=True

        if st.session_state.optimized and st.session_state.result:
            res=st.session_state.result
            # Pesos en 2 columnas lado a lado
            assets=list(res.weights.index); mid=len(assets)//2+len(assets)%2
            col_a,col_b,col_r=st.columns([2,2,1.5])
            nw={}
            with col_a:
                for a in assets[:mid]:
                    ic="🟢" if a==FICO_TK else "🔵"
                    nw[a]=st.number_input(f"{ic} {a}",0.0,100.0,round(float(res.weights[a])*100,1),0.5,"%.1f",key=f"s_{a}")
            with col_b:
                for a in assets[mid:]:
                    ic="🟢" if a==FICO_TK else "🔵"
                    nw[a]=st.number_input(f"{ic} {a}",0.0,100.0,round(float(res.weights[a])*100,1),0.5,"%.1f",key=f"s_{a}")
            wn=pd.Series(nw); tot=wn.sum(); wnorm=wn/tot if tot>0 else wn/100; st.session_state.manual_weights=wnorm
            eqw=float(wnorm[[a for a in wnorm.index if a!=FICO_TK]].sum()); fiw=float(wnorm.get(FICO_TK,0))
            with col_r:
                st.metric("RV",f"{eqw:.1%}",delta=f"{eqw-eq_t:+.1%}"); st.metric("RF",f"{fiw:.1%}",delta=f"{fiw-fi_t:+.1%}")
                st.caption(f"Suma: {tot:.0f}%{'✅' if abs(tot-100)<0.5 else ' ⚠️→100%'}")

            # Métricas dinámicas
            w_np=wnorm.reindex(res.bl_returns.index).fillna(0).to_numpy()
            mu_np=res.bl_returns.to_numpy(); S_np=res.cov_matrix.to_numpy()
            b_np=st.session_state.betas.reindex(res.bl_returns.index).fillna(1).to_numpy()
            p_r=float(w_np@mu_np); p_v=float(np.sqrt(max(w_np@S_np@w_np,1e-10)))
            p_sh=(p_r-RF)/p_v if p_v>1e-10 else 0; p_bt=float(w_np@b_np)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("Retorno",f"{p_r:.2%}"); m2.metric("Riesgo",f"{p_v:.2%}"); m3.metric("Sharpe",f"{p_sh:.2f}"); m4.metric("Beta",f"{p_bt:.2f}")

            # Gráficos: composición por activo + por sector
            import plotly.express as px
            g1,g2=st.columns(2)
            with g1:
                st.caption("Por activo")
                ws=wnorm[wnorm>1e-4]
                colors=px.colors.qualitative.Set2[:len(ws)]
                fig=go.Figure(go.Pie(labels=ws.index.tolist(),values=ws.values.tolist(),
                    marker_colors=colors,hole=.4,textinfo="label+percent"))
                fig.update_layout(height=280,margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                st.plotly_chart(fig,use_container_width=True)
            with g2:
                st.caption("Por sector")
                sec=st.session_state.sectors
                if sec is not None and not sec.empty:
                    sw={}
                    for a in wnorm.index:
                        if wnorm[a]>1e-4: s=sec.get(a,"–"); sw[s]=sw.get(s,0)+wnorm[a]
                    sec_colors=px.colors.qualitative.Pastel[:len(sw)]
                    fig=go.Figure(go.Pie(labels=list(sw.keys()),values=list(sw.values()),
                        marker_colors=sec_colors,hole=.4,textinfo="label+percent"))
                    fig.update_layout(height=280,margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                    st.plotly_chart(fig,use_container_width=True)

            # Evolución histórica — filtrada por chart_years, siempre desde capital inicial
            pr_full,wl_full,dd_full,bw_full,bdd_full=wdd(wnorm,st.session_state.returns,st.session_state.bench_rets,capital)

            # Filtrar al rango visual seleccionado
            cy = int(chart_years.replace("y",""))
            cutoff = pr_full.index.max() - pd.DateOffset(years=cy)
            pr = pr_full.loc[pr_full.index >= cutoff]

            # Recalcular wealth desde capital inicial para el rango visible
            wl = np.exp(pr.cumsum()) * capital
            dd = wl / wl.cummax() - 1
            bw, bdd = {}, {}
            for n in bw_full:
                br = st.session_state.bench_rets[n].loc[pr.index].fillna(0) if n in st.session_state.bench_rets else pd.Series(0,index=pr.index)
                bw[n] = np.exp(br.cumsum()) * capital
                bdd[n] = bw[n] / bw[n].cummax() - 1

            fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.65,.35],vertical_spacing=.04,
                             subplot_titles=[f"Evolución de capital · últimos {cy} años (${capital:,.0f})","Drawdown"])
            fig.add_trace(go.Scatter(x=wl.index,y=wl.values,name="Portafolio",
                                    line=dict(color=C_RV,width=2.5)),row=1,col=1)
            for i,(n,v) in enumerate(bw.items()):
                fig.add_trace(go.Scatter(x=v.index,y=v.values,name=n,
                    line=dict(color=BC[i%len(BC)],dash="dash",width=1.5)),row=1,col=1)
            fig.add_trace(go.Scatter(x=dd.index,y=dd.values,name="DD Portafolio",
                                    fill="tozeroy",fillcolor="rgba(214,96,77,0.3)",
                                    line=dict(color=C_OPT,width=1.5)),row=2,col=1)
            for i,(n,ddb) in enumerate(bdd.items()):
                fig.add_trace(go.Scatter(x=ddb.index,y=ddb.values,name=f"DD {n}",
                    line=dict(color=BC[i%len(BC)],dash="dot",width=1)),row=2,col=1)
            _prof=RiskProfile.for_split(eq_t,fi_t)
            fig.add_hline(y=-_prof.max_drawdown,line_dash="dash",line_color="black",
                         row=2,col=1,annotation_text=f"Límite {_prof.max_drawdown:.0%}")
            fig.update_yaxes(tickprefix="$",tickformat=",.0f",row=1,col=1)
            fig.update_yaxes(tickformat=".0%",row=2,col=1)
            fig.update_layout(height=520,margin=dict(l=0,r=0,t=25,b=0),
                             legend=dict(orientation="h",y=-0.08,font=dict(size=10)))
            st.plotly_chart(fig,use_container_width=True)

            # Métricas históricas del rango visible
            ann_r=np.exp(pr.mean()*PPY)-1; ann_v=pr.std(ddof=1)*np.sqrt(PPY)
            from scipy.stats import norm as _norm
            mu_h=pr.mean()*PPY; sig_h=ann_v; z=_norm.ppf(0.05)
            var95=-(mu_h+z*sig_h); cvar95=-(mu_h-sig_h*_norm.pdf(z)/0.05)
            st.caption(f"**Histórico:** Ret {ann_r:.2%} · Vol {ann_v:.2%} · DD máx {dd.min():.2%} · VaR 95% {var95:.2%} · CVaR 95% {cvar95:.2%}")

# ═══════════════════ TAB 4 ════════════════════════════════════════════════════
with tab4:
    if not(st.session_state.optimized and st.session_state.result): st.info("⬅️ Optimiza primero.")
    else:
        res=st.session_state.result
        wnorm=st.session_state.manual_weights if st.session_state.manual_weights is not None else res.weights

        # MONTE CARLO
        st.subheader("🎲 ¿Cuánto podría valer tu portafolio?")
        c1,c2,c3=st.columns(3)
        mh=c1.selectbox("Años",[1,2,3,5,10],index=2); mn=c2.selectbox("Precisión",[1000,5000,10000],index=1,format_func=lambda x:f"{x:,}")
        mt=c3.number_input("Meta (USD)",value=int(capital*1.2),step=10_000,format="%d")
        if st.button("▶️ Proyectar",type="primary",use_container_width=True):
            with st.spinner(f"Simulando…"): st.session_state["mc"]=monte_carlo(wnorm,res.bl_returns,res.cov_matrix,capital,mh,PPY,mn,mt)
        if "mc" in st.session_state and st.session_state["mc"]:
            mc=st.session_state["mc"]; gain=mc.median_path[-1]-mc.capital
            c1,c2,c3=st.columns(3)
            c1.metric("💰 Proyectado",f"${mc.median_path[-1]:,.0f}",delta=f"+${gain:,.0f} ({gain/mc.capital:+.1%})")
            c2.metric("🛡️ No perder",f"{100-mc.prob_loss*100:.0f}%")
            c3.metric("🎯 Alcanzar meta",f"{mc.prob_target:.0%}",delta=f"${mc.target:,.0f}",delta_color="off")
            st.success(f"En **{mc.horizon_years:.0f} año(s)**, tu inversión de ${mc.capital:,.0f} probablemente valdrá "
                       f"entre **${mc.percentiles[5][-1]:,.0f}** y **${mc.percentiles[95][-1]:,.0f}**, "
                       f"con un valor más probable de **${mc.median_path[-1]:,.0f}**.")

            terminal = mc.terminal
            idx_best = int(np.argmax(terminal))
            idx_worst = int(np.argmin(terminal))
            idx_median = int(np.argsort(terminal)[len(terminal)//2])
            x = mc.dates

            # ── Métricas clave MC ──
            p5_val  = mc.percentiles[5][-1]
            p50_val = mc.percentiles[50][-1]
            p95_val = mc.percentiles[95][-1]
            gbm_min = terminal[idx_worst]
            gbm_max = terminal[idx_best]

            # ── GRÁFICO 1: Monte Carlo — bandas de percentiles ───────────
            st.markdown("#### 📊 Distribución de resultados — Bandas de confianza")
            st.caption("Este gráfico resume el **rango probable** de tu inversión. "
                       "Las bandas muestran dónde caería tu capital en el 50%, 80% y 90% de los escenarios simulados. "
                       "La línea central (mediana) es el resultado más representativo.")
            fig1=go.Figure()
            for lo,hi,cl,nm in [(5,95,"rgba(46,94,140,0.08)","90% de escenarios"),
                                (10,90,"rgba(46,94,140,0.12)","80%"),
                                (25,75,"rgba(46,94,140,0.18)","50%")]:
                fig1.add_trace(go.Scatter(x=list(x)+list(x[::-1]),
                    y=list(mc.percentiles[hi])+list(mc.percentiles[lo][::-1]),
                    fill="toself",fillcolor=cl,line=dict(width=0),name=nm))
            fig1.add_trace(go.Scatter(x=x,y=mc.median_path,name="Mediana (P50)",
                                     line=dict(color=C_RV,width=2.5)))
            fig1.add_hline(y=mc.capital,line_dash="dot",line_color="gray",
                          annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target!=mc.capital:
                fig1.add_hline(y=mc.target,line_dash="dot",line_color=C_RF,
                              annotation_text=f"Meta ${mc.target:,.0f}")
            fig1.update_yaxes(tickprefix="$",tickformat=",.0f")
            fig1.update_layout(height=380,margin=dict(l=0,r=0,t=5,b=0),
                              legend=dict(orientation="h",y=-0.12))
            st.plotly_chart(fig1,use_container_width=True)

            # Interpretación MC
            st.info(
                f"**¿Cómo leerlo?** De {len(terminal):,} simulaciones, el 90% terminó entre "
                f"**${p5_val:,.0f}** y **${p95_val:,.0f}**. "
                f"El valor más probable (mediana) es **${p50_val:,.0f}** "
                f"({'ganancia' if p50_val>mc.capital else 'pérdida'} de "
                f"**{abs(p50_val/mc.capital-1):.1%}**). "
                f"La banda más oscura (50% central) es donde se concentra la mayoría de resultados — "
                f"si la banda es angosta, el portafolio tiene comportamiento más predecible."
            )

            # ── GRÁFICO 2: GBM — trayectorias individuales ───────────────
            st.markdown("#### 🔀 Trayectorias individuales — Movimiento Browniano Geométrico")
            st.caption("Mientras el gráfico anterior resume los rangos, este muestra **caminos concretos** "
                       "que podría seguir tu inversión semana a semana. Cada línea gris es un escenario posible.")
            n_show = st.slider("Trayectorias a mostrar",10,200,50,10,
                               help="Cuántas simulaciones individuales dibujar de fondo.",
                               key="gbm_paths")

            fig2=go.Figure()
            rng_vis = np.random.default_rng(0)
            sample_idx = rng_vis.choice(len(terminal), size=min(n_show, len(terminal)), replace=False)
            for si in sample_idx:
                fig2.add_trace(go.Scatter(x=x,y=mc.paths[si],mode="lines",
                    line=dict(color="rgba(150,150,150,0.15)",width=0.5),
                    showlegend=False,hoverinfo="skip"))
            fig2.add_trace(go.Scatter(x=x,y=mc.paths[idx_best],
                name=f"🚀 Mejor: ${gbm_max:,.0f} ({gbm_max/mc.capital-1:+.1%})",
                line=dict(color="#2CA02C",width=2.5)))
            fig2.add_trace(go.Scatter(x=x,y=mc.paths[idx_worst],
                name=f"😟 Peor: ${gbm_min:,.0f} ({gbm_min/mc.capital-1:+.1%})",
                line=dict(color="#D6604D",width=2.5)))
            fig2.add_trace(go.Scatter(x=x,y=mc.paths[idx_median],
                name=f"📊 Mediana: ${terminal[idx_median]:,.0f} ({terminal[idx_median]/mc.capital-1:+.1%})",
                line=dict(color=C_RV,width=3)))
            fig2.add_hline(y=mc.capital,line_dash="dot",line_color="gray",
                          annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target!=mc.capital:
                fig2.add_hline(y=mc.target,line_dash="dot",line_color=C_RF,
                              annotation_text=f"Meta ${mc.target:,.0f}")
            fig2.update_yaxes(tickprefix="$",tickformat=",.0f")
            fig2.update_layout(height=420,margin=dict(l=0,r=0,t=5,b=0),
                              legend=dict(orientation="h",y=-0.12))
            st.plotly_chart(fig2,use_container_width=True)

            # Interpretación GBM
            st.info(
                f"**¿Cómo leerlo?** Cada línea gris es una posible evolución semanal de tu inversión. "
                f"La mejor trayectoria alcanzó **${gbm_max:,.0f}** y la peor cayó a **${gbm_min:,.0f}**. "
                f"Nota que estos extremos son más amplios que las bandas P5/P95 del gráfico anterior "
                f"(${p5_val:,.0f} – ${p95_val:,.0f}), porque aquí ves los extremos absolutos de "
                f"{len(terminal):,} simulaciones, no solo el rango donde cae el 90%."
            )

            # ── Métricas resumen ──
            sc1,sc2,sc3=st.columns(3)
            sc1.metric("😟 Si va mal (P5)",f"${p5_val:,.0f}",delta=f"{p5_val/mc.capital-1:+.1%}")
            sc2.metric("📊 Más probable (P50)",f"${p50_val:,.0f}",delta=f"{p50_val/mc.capital-1:+.1%}")
            sc3.metric("🚀 Si va bien (P95)",f"${p95_val:,.0f}",delta=f"{p95_val/mc.capital-1:+.1%}")

            # ── Explicación comparativa ──
            with st.expander("❓ ¿Por qué los rangos del primer gráfico y del segundo son diferentes?"):
                st.markdown(
                    f"""**Ambos gráficos usan las mismas {len(terminal):,} simulaciones.** La diferencia está en qué muestran:

**Gráfico 1 (Bandas)** → Muestra **percentiles estadísticos** (P5 a P95). Descarta el 5% de escenarios más extremos por arriba y por abajo. Es como decir: *"en 9 de cada 10 escenarios, tu inversión terminará entre ${p5_val:,.0f} y ${p95_val:,.0f}"*. Es la visión más útil para planificación.

**Gráfico 2 (Trayectorias)** → Muestra **caminos individuales**, incluyendo el mejor y el peor absoluto de todas las simulaciones. La trayectoria peor (${gbm_min:,.0f}) y la mejor (${gbm_max:,.0f}) son outliers — eventos de probabilidad menor al 0.02% — pero existen en el universo de posibilidades.

**¿Cuál usar?** Las bandas del gráfico 1 son tu referencia para tomar decisiones. El gráfico 2 muestra la volatilidad real del camino: aunque termines en la mediana, el viaje puede tener subidas y bajadas fuertes. Si las trayectorias grises te parecen muy erráticas, el portafolio podría beneficiarse de más renta fija."""
                )

        # STRESS
        st.divider()
        st.subheader("🔥 Pruebas de estrés")
        ret_st=st.session_state.returns_full if st.session_state.returns_full is not None else st.session_state.returns
        bench_st=st.session_state.bench_full if st.session_state.bench_full is not None else (st.session_state.bench_rets if isinstance(st.session_state.bench_rets,dict) else {})
        if ret_st is not None: st.caption(f"📅 {ret_st.index.min().strftime('%Y-%m-%d')} → {ret_st.index.max().strftime('%Y-%m-%d')}")
        if st.button("▶️ Correr estrés",use_container_width=True):
            pb=list(bench_st.values())[0] if bench_st else None
            with st.spinner("…"): st.session_state["stress"]=stress_test(wnorm,ret_st,CRISIS_PERIODS,capital,{FICO_TK:FICO},PPY,pb)
        if "stress" in st.session_state and st.session_state["stress"]:
            stres=st.session_state["stress"]; avail=[s for s in stres if s.available]
            if not avail: st.warning("Sin datos. Amplía la historia.")
            else:
                worst=min(avail,key=lambda s:s.port_return); beats=sum(1 for s in avail if s.port_return>s.benchmark_return)
                st.info(f"💡 Peor: **{worst.name}** ({worst.port_return:+.2%}). Superas benchmark en **{beats}/{len(avail)}**.")
                fig=go.Figure()
                fig.add_trace(go.Bar(x=[s.name for s in avail],y=[s.port_return for s in avail],name="Portafolio",marker_color=C_OPT))
                fig.add_trace(go.Bar(x=[s.name for s in avail],y=[s.benchmark_return for s in avail],name="Benchmark",marker_color="#888"))
                fig.update_yaxes(tickformat=".1%"); fig.update_layout(barmode="group",height=300,margin=dict(l=0,r=0,t=5,b=0),legend=dict(orientation="h",y=1.08))
                st.plotly_chart(fig,use_container_width=True)
                for s in avail:
                    ic="🔴" if s.port_return<0 else "🟢"; diff=s.port_return-s.benchmark_return
                    cA,cB=st.columns([3,1])
                    with cA: st.markdown(f"{ic} **{s.name}** · {s.start} → {s.end}"); st.caption(s.description)
                    with cB: st.metric("Impacto",f"{s.port_return:+.2%}",delta=f"${s.port_loss:,.0f}",delta_color="off")
                    st.caption(f"→ {'Mejor' if diff>0 else 'Peor'} que benchmark ({diff:+.2%}). DD: {s.max_drawdown:.2%}. "
                               + (f"Peor: {s.asset_returns.sort_values().index[0]} ({s.asset_returns.sort_values().iloc[0]:+.2%})" if not s.asset_returns.empty else ""))
                    st.divider()
            miss=[s for s in stres if not s.available]
            if miss:
                with st.expander(f"ℹ️ {len(miss)} crisis fuera del rango"):
                    for s in miss: st.caption(f"**{s.name}** ({s.start}→{s.end})")
