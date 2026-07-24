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
FICO = ForcedAsset(ret_annual=0.065,vol_annual=0.010,beta=0.30,sector="Factoring",region="Perú",moneda="USD",instrumento="Fondo")
PERFILES = {"Conservador (30/70)":(0.30,0.70),"Moderado-bajo (40/60)":(0.40,0.60),
            "Moderado (50/50)":(0.50,0.50),"Crecimiento (60/40)":(0.60,0.40),"Agresivo (70/30)":(0.70,0.30)}
P_DESC = {"Conservador (30/70)":"Preservar capital.","Moderado-bajo (40/60)":"Leve crecimiento.",
          "Moderado (50/50)":"Balance.","Crecimiento (60/40)":"Mayor exposición.","Agresivo (70/30)":"Máxima RV."}
EJ = ["AAPL","MSFT","NVDA","JNJ","KO","QQQ"]
C_RV,C_RF,C_OPT = "#2E5E8C","#2CA02C","#D6604D"
BC = ["#888","#E377C2","#FF7F0E","#9467BD","#17BECF"]

for k,v in {"tickers":[],"benchmarks":["^GSPC"],"views":[],"optimized":False,"result":None,
            "manual_weights":None,"returns":None,"bench_rets":None,"betas":None,"sectors":None,
            "returns_full":None,"bench_full":None,"last_period":None,"data_range":""}.items():
    st.session_state.setdefault(k,v)

# ═══════════════════ BACKEND ══════════════════════════════════════════════════
@st.cache_data(show_spinner=False,ttl=600)
def dl_eq(tickers,period="5y"):
    import yfinance as yf
    raw=yf.download(tickers,period=period,interval="1wk",auto_adjust=True,progress=False)
    if raw is None or raw.empty: return None
    px=raw["Close"].copy() if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":list(tickers)[0]})
    px=px.dropna(how="all").ffill(); px.index=pd.to_datetime(px.index).tz_localize(None)
    return np.log(px/px.shift(1)).replace([np.inf,-np.inf],np.nan).dropna(how="all")

@st.cache_data(show_spinner=False,ttl=600)
def dl_bk(tks,period="5y"):
    import yfinance as yf
    out={}
    for b in tks:
        b=b.strip().upper()
        if not b: continue
        try:
            raw=yf.download(b,period=period,interval="1wk",auto_adjust=True,progress=False)
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
        r=requests.get("https://query2.finance.yahoo.com/v1/finance/search",params={"q":q,"quotesCount":6,"newsCount":0},
                       headers={"User-Agent":"Mozilla/5.0"},timeout=5)
        return [{"tk":x["symbol"],"nm":x.get("shortname") or x.get("longname",""),"tp":x.get("quoteType","")} for x in r.json().get("quotes",[]) if x.get("symbol")]
    except: return []

def do_opt(tickers,views_cfg,eq_t,fi_t,pb):
    betas=st.session_state.betas.copy(); betas[FICO_TK]=FICO.beta
    ok=[t for t in tickers if t in st.session_state.returns.columns]; aa=set(ok)|{FICO_TK}
    views=[View(kind="absolute",asset=v["asset"],q=v["q"],confidence=v["confidence"]) if v["type"]=="absolute"
           else View(kind="relative",long=v["long"],short=v["short"],q=v["q"],confidence=v["confidence"])
           for v in views_cfg if (v["type"]=="absolute" and v.get("asset") in aa) or
                                  (v["type"]!="absolute" and v.get("long") in aa and v.get("short") in aa)]
    return run_profile(returns=st.session_state.returns,equity_assets=ok,forced_assets={FICO_TK:FICO},
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
    if not tks or not bks: return False
    lr=dl_eq(tuple(tks),period)
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
    period=st.selectbox("Historia",["1y","2y","3y","5y","10y","max"],index=3)
    with st.expander("⚙️ Avanzado"):
        _p=RiskProfile.for_split(eq_t,fi_t)
        st.caption(f"RF: {FICO_TK} · {FICO.ret_annual:.2%} | Beta: {_p.beta_min:.2f}–{_p.beta_max:.2f} | DD máx: {_p.max_drawdown:.0%}")
        if st.button("🗑️ Limpiar caché",use_container_width=True): st.cache_data.clear(); st.toast("✓")

# Auto-descarga si período cambió
if st.session_state.tickers and st.session_state.benchmarks and st.session_state.last_period and st.session_state.last_period!=period:
    with st.spinner(f"Actualizando a {period}…"): run_dl(period)

# ═══════════════════ MAIN ═════════════════════════════════════════════════════
st.title("Optimizador de portafolios")
tab1,tab2,tab3,tab4=st.tabs(["1 · Activos","2 · Expectativas","3 · Portafolio","4 · Proyecciones"])

# ═══════════════════ TAB 1 ════════════════════════════════════════════════════
with tab1:
    col_s,col_t=st.columns([4,1])
    with col_t: add_to=st.radio("Añadir como",["🔵 Activo","📊 Benchmark"])
    with col_s: q=st.text_input("🔍 Buscar",placeholder="Visa, Apple, ^GSPC, QQQ…")
    if q.strip():
        res=search_yf(q.strip())
        if res:
            cols=st.columns(min(len(res),3))
            for i,r in enumerate(res):
                with cols[i%len(cols)]:
                    if st.button(f"➕ {r['tk']} — {r['nm'][:18]}",key=f"a_{r['tk']}",use_container_width=True):
                        tk=r['tk']; tgt="tickers" if add_to=="🔵 Activo" else "benchmarks"
                        if tk not in st.session_state[tgt]: st.session_state[tgt].append(tk); st.toast(f"✓ {tk}")
    if not st.session_state.tickers:
        if st.button("🚀 Cargar ejemplo",type="primary"):
            st.session_state.tickers=list(EJ); st.session_state.views=[]

    la,lb=st.columns(2)
    with la:
        st.caption(f"**🔵 Activos ({len(st.session_state.tickers)})**")
        for i,t in enumerate(st.session_state.tickers):
            c1,c2=st.columns([5,1]); c1.write(t)
            if c2.button("✕",key=f"ra{i}"):
                rm=st.session_state.tickers.pop(i)
                st.session_state.views=[v for v in st.session_state.views if v.get("asset")!=rm and v.get("long")!=rm and v.get("short")!=rm]
    with lb:
        st.caption(f"**📊 Benchmarks ({len(st.session_state.benchmarks)})**")
        for i,b in enumerate(st.session_state.benchmarks):
            c1,c2=st.columns([5,1]); c1.write(b)
            if c2.button("✕",key=f"rb{i}"): st.session_state.benchmarks.pop(i)

    if st.session_state.data_range: st.success(f"📦 {st.session_state.data_range} ({st.session_state.last_period})")
    if st.button("📥 Descargar datos",type="primary",use_container_width=True,
                 disabled=not(st.session_state.tickers and st.session_state.benchmarks)):
        with st.spinner("Descargando…"):
            if run_dl(period): st.success(f"✅ {st.session_state.data_range}")
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
                r=do_opt(st.session_state.tickers,st.session_state.views,eq_t,fi_t,pb)
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

            # Evolución histórica — grande, con DD de portafolio Y benchmark
            pr,wl,dd,bw,bdd=wdd(wnorm,st.session_state.returns,st.session_state.bench_rets,capital)
            fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.65,.35],vertical_spacing=.04,
                             subplot_titles=[f"Evolución de capital (${capital:,.0f})","Drawdown"])
            fig.add_trace(go.Scatter(x=wl.index,y=wl.values,name="Portafolio",
                                    line=dict(color=C_RV,width=2.5)),row=1,col=1)
            for i,(n,v) in enumerate(bw.items()):
                fig.add_trace(go.Scatter(x=v.index,y=v.values,name=n,
                    line=dict(color=BC[i%len(BC)],dash="dash",width=1.5)),row=1,col=1)
            # Drawdown portafolio
            fig.add_trace(go.Scatter(x=dd.index,y=dd.values,name="DD Portafolio",
                                    fill="tozeroy",fillcolor="rgba(214,96,77,0.3)",
                                    line=dict(color=C_OPT,width=1.5)),row=2,col=1)
            # Drawdown benchmarks
            for i,(n,ddb) in enumerate(bdd.items()):
                fig.add_trace(go.Scatter(x=ddb.index,y=ddb.values,name=f"DD {n}",
                    line=dict(color=BC[i%len(BC)],dash="dot",width=1)),row=2,col=1)
            # Línea límite DD del perfil
            _prof=RiskProfile.for_split(eq_t,fi_t)
            fig.add_hline(y=-_prof.max_drawdown,line_dash="dash",line_color="black",
                         row=2,col=1,annotation_text=f"Límite {_prof.max_drawdown:.0%}")
            fig.update_yaxes(tickprefix="$",tickformat=",.0f",row=1,col=1)
            fig.update_yaxes(tickformat=".0%",row=2,col=1)
            fig.update_layout(height=520,margin=dict(l=0,r=0,t=25,b=0),
                             legend=dict(orientation="h",y=-0.08,font=dict(size=10)))
            st.plotly_chart(fig,use_container_width=True)

            # Métricas históricas inline
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
            fig=go.Figure(); x=mc.dates
            for lo,hi,cl,nm in [(5,95,"rgba(46,94,140,0.08)","90% de escenarios"),(10,90,"rgba(46,94,140,0.12)","80%"),(25,75,"rgba(46,94,140,0.18)","50%")]:
                fig.add_trace(go.Scatter(x=list(x)+list(x[::-1]),y=list(mc.percentiles[hi])+list(mc.percentiles[lo][::-1]),fill="toself",fillcolor=cl,line=dict(width=0),name=nm))
            fig.add_trace(go.Scatter(x=x,y=mc.median_path,name="Más probable",line=dict(color=C_RV,width=2.5)))
            fig.add_hline(y=mc.capital,line_dash="dot",line_color="gray",annotation_text=f"Inversión ${mc.capital:,.0f}")
            if mc.target!=mc.capital: fig.add_hline(y=mc.target,line_dash="dot",line_color=C_RF,annotation_text=f"Meta ${mc.target:,.0f}")
            fig.update_yaxes(tickprefix="$",tickformat=",.0f"); fig.update_layout(height=380,margin=dict(l=0,r=0,t=5,b=0),legend=dict(orientation="h",y=-0.12))
            st.plotly_chart(fig,use_container_width=True)
            sc1,sc2,sc3=st.columns(3)
            sc1.metric("😟 Si va mal (P5)",f"${mc.percentiles[5][-1]:,.0f}",delta=f"{mc.percentiles[5][-1]/mc.capital-1:+.1%}")
            sc2.metric("📊 Más probable (P50)",f"${mc.percentiles[50][-1]:,.0f}",delta=f"{mc.percentiles[50][-1]/mc.capital-1:+.1%}")
            sc3.metric("🚀 Si va bien (P95)",f"${mc.percentiles[95][-1]:,.0f}",delta=f"{mc.percentiles[95][-1]/mc.capital-1:+.1%}")

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
