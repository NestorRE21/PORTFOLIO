# -*- coding: utf-8 -*-
"""
optimizer.py — Motor Black-Litterman modular (Coril SAB)
=========================================================

Motor autocontenido, sin dependencias de Streamlit ni de yfinance.
Recibe retornos ya calculados y devuelve pesos óptimos por perfil.

Flujo de uso:
    from optimizer import (
        RiskProfile, ForcedAsset, BLConfig,
        estimate_covariance, equilibrium_returns,
        black_litterman, mean_variance_optimize,
        run_profile,
    )

    result = run_profile(
        returns        = df_log_returns,      # DataFrame (fechas × activos) log-ret
        equity_assets  = ["AAPL", "MSFT", ...],
        forced_assets  = {"FICCMP13": ForcedAsset(ret_annual=0.0625, ...)},
        profile        = RiskProfile.for_split(0.50, 0.50),
        views          = [ ... ],
        config         = BLConfig(),
    )
    result.weights     # pd.Series de pesos que suman 1

Diseño:
    - Todo anualizado con periods_per_year (52 semanal, 252 diario).
    - Retornos de equilibrio vía reverse optimization: Π = λ·Σ·w_mkt.
    - Covarianza Ledoit-Wolf con regularización PSD (Higham).
    - Optimización Mean-Variance: max wᵀμ − (λ/2)·wᵀΣw con restricciones hard.
    - FICO forzado inyectado como parámetro, no hardcodeado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except ImportError:                       # fallback si sklearn no está
    _HAS_SKLEARN = False

EPS = 1e-10


# =============================================================================
# CONFIGURACIÓN Y TIPOS
# =============================================================================

class ProfileName(str, Enum):
    """Perfiles base que definen restricciones de beta y drawdown."""
    CONSERVADOR = "Conservador"
    MODERADO    = "Moderado"
    AGRESIVO    = "Agresivo"


# Restricciones por perfil base (heredadas de usd__clientes.py)
_PROFILE_BETA_DD = {
    ProfileName.CONSERVADOR: {"beta_min": 0.40, "beta_max": 0.90, "max_dd": 0.20},
    ProfileName.MODERADO:    {"beta_min": 0.60, "beta_max": 1.10, "max_dd": 0.30},
    ProfileName.AGRESIVO:    {"beta_min": 0.80, "beta_max": 1.90, "max_dd": 0.45},
}


@dataclass
class RiskProfile:
    """
    Un perfil de inversión: split equity/fico + restricciones de beta/drawdown.

    Los 5 perfiles comerciales (A-E) se construyen con for_split(), que asigna
    el perfil base correcto según el % de renta variable.
    """
    equity_target: float
    fico_target:   float
    base:          ProfileName
    beta_min:      float
    beta_max:      float
    max_drawdown:  float
    label:         str = ""

    @staticmethod
    def _base_for_equity(equity_pct: float) -> ProfileName:
        """Mapea % equity -> perfil base (regla FIVE_PROFILE de usd__clientes)."""
        if equity_pct <= 0.40 + EPS:
            return ProfileName.CONSERVADOR
        if equity_pct <= 0.60 + EPS:
            return ProfileName.MODERADO
        return ProfileName.AGRESIVO

    @classmethod
    def for_split(cls, equity_target: float, fico_target: float,
                  label: str = "") -> "RiskProfile":
        if abs(equity_target + fico_target - 1.0) > 1e-6:
            raise ValueError(
                f"equity_target + fico_target debe sumar 1.0 "
                f"(recibido {equity_target + fico_target:.4f})."
            )
        base = cls._base_for_equity(equity_target)
        r    = _PROFILE_BETA_DD[base]
        return cls(
            equity_target = equity_target,
            fico_target   = fico_target,
            base          = base,
            beta_min      = r["beta_min"],
            beta_max      = r["beta_max"],
            max_drawdown  = r["max_dd"],
            label         = label or f"{equity_target:.0%}/{fico_target:.0%}",
        )


@dataclass
class ForcedAsset:
    """Activo de renta fija con retorno/vol forzados (no estimados de datos)."""
    ret_annual:  float
    vol_annual:  float
    beta:        float = 0.30
    sector:      str   = "Renta Fija"
    region:      str   = "Perú"
    moneda:      str   = "USD"
    instrumento: str   = "Fondo de inversión"
    asset_class: str   = "Renta Fija"


@dataclass
class View:
    """
    View del analista. Dos tipos:
      - absolute: asset rendirá q anual.
      - relative: long rendirá q anual más que short.
    confidence en (0, 1]: 1.0 = certeza total, 0.1 = muy incierta.
    """
    kind:        str                      # "absolute" | "relative"
    q:           float                    # retorno esperado anual
    confidence:  float = 0.5
    asset:       Optional[str] = None     # para absolute
    long:        Optional[str] = None     # para relative
    short:       Optional[str] = None     # para relative
    name:        str = ""


@dataclass
class BLConfig:
    """Parámetros globales del motor."""
    rf_annual:        float = 0.02
    periods_per_year: int   = 52          # 52 semanal, 252 diario
    tau:              float = 0.05        # incertidumbre del equilibrio
    min_lambda:       float = 0.50
    max_lambda:       float = 10.0
    fallback_lambda:  float = 2.50
    max_weight_equity: float = 0.25       # cap por activo RV
    gamma_beta:       float = 5.0         # peso de la penalización de beta (suave)
    ridge:            float = 1e-8        # regularización PSD


@dataclass
class ProfileResult:
    """Salida de run_profile()."""
    weights:        pd.Series
    exp_return:     float
    volatility:     float
    sharpe:         float
    beta:           float
    equity_weight:  float
    fico_weight:    float
    bl_returns:     pd.Series
    equilibrium:    pd.Series
    cov_matrix:     pd.DataFrame
    profile:        RiskProfile
    success:        bool
    message:        str = ""
    feasible:       bool = True          # validación matemática (no el flag del solver)
    feasibility_report: str = ""


# =============================================================================
# ÁLGEBRA NUMÉRICA
# =============================================================================

def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Proyección espectral a PSD (Higham): simetriza y colapsa autovalores <ε."""
    m = np.asarray(matrix, dtype=float)
    m = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(m)
    vals = np.maximum(vals, epsilon)
    psd  = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (psd + psd.T)


def solve_psd(A: np.ndarray, B: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Resuelve A·X = B de forma estable; pseudo-inversa si A es singular."""
    A = nearest_psd(A, epsilon=ridge)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ B


# =============================================================================
# COVARIANZA
# =============================================================================

def estimate_covariance(returns: pd.DataFrame, periods_per_year: int = 52,
                        ridge: float = 1e-8) -> pd.DataFrame:
    """
    Covarianza anualizada con Ledoit-Wolf shrinkage.
    Se ajusta sobre retornos periódicos y se anualiza × periods_per_year.
    Fallback a covarianza muestral si sklearn no está disponible.
    """
    clean = (returns.replace([np.inf, -np.inf], np.nan)
                    .dropna(how="all")
                    .fillna(0.0))
    if clean.empty:
        raise ValueError("Matriz de retornos vacía tras limpieza.")

    X = clean.to_numpy(dtype=float)

    if _HAS_SKLEARN:
        lw  = LedoitWolf().fit(X)
        cov = lw.covariance_
    else:
        cov = np.cov(X, rowvar=False, ddof=1)

    cov_annual = nearest_psd(cov * periods_per_year, epsilon=ridge)
    return pd.DataFrame(cov_annual, index=clean.columns, columns=clean.columns)


def inject_forced_assets(returns: pd.DataFrame,
                         forced_assets: dict[str, ForcedAsset],
                         periods_per_year: int = 52,
                         seed: int = 42) -> pd.DataFrame:
    """
    Añade columnas de retornos sintéticos para los activos forzados (FICO).
    Retorno periódico constante + ruido mínimo para estabilidad de la covarianza.
    """
    if not forced_assets:
        return returns
    rng    = np.random.default_rng(seed)
    frames = {}
    for ticker, fa in forced_assets.items():
        per_ret = np.log(1.0 + fa.ret_annual) / periods_per_year
        per_vol = fa.vol_annual / np.sqrt(periods_per_year)
        noise   = rng.normal(0.0, per_vol * 0.05, size=len(returns))
        frames[ticker] = pd.Series(per_ret + noise, index=returns.index, name=ticker)
    forced_df = pd.DataFrame(frames)
    return pd.concat([returns, forced_df], axis=1)


# =============================================================================
# CALIBRACIÓN DE LAMBDA
# =============================================================================

def calibrate_lambda(benchmark_returns: pd.Series, config: BLConfig) -> float:
    """
    λ = (E[R_bmk] − Rf) / Var[R_bmk], anualizado, con clamp y fallback.
    Si no hay benchmark, usa fallback_lambda.
    """
    if benchmark_returns is None or len(benchmark_returns) < 2:
        return config.fallback_lambda

    bmk       = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()
    ret_ann   = bmk.mean() * config.periods_per_year
    var_ann   = bmk.var(ddof=1) * config.periods_per_year
    if not np.isfinite(var_ann) or var_ann <= EPS:
        return config.fallback_lambda

    raw = (ret_ann - config.rf_annual) / var_ann
    if np.isfinite(raw) and config.min_lambda <= raw <= config.max_lambda:
        return float(raw)
    return config.fallback_lambda


# =============================================================================
# RETORNOS DE EQUILIBRIO (reverse optimization)
# =============================================================================

def market_weights(assets: Sequence[str], equity_assets: Sequence[str],
                   fico_assets: Sequence[str], equity_target: float,
                   fico_target: float,
                   market_caps: Optional[dict] = None) -> pd.Series:
    """
    Pesos de mercado para el equilibrio. Sin market caps: equipondera dentro
    de cada bucket, respetando el split del perfil.
    """
    eq = list(equity_assets)
    fi = list(fico_assets)
    w  = {}
    if market_caps:
        eq_tot = sum(market_caps.get(a, 0.0) for a in eq) or 1.0
        fi_tot = sum(market_caps.get(a, 0.0) for a in fi) or 1.0
        for a in eq: w[a] = equity_target * market_caps.get(a, 0.0) / eq_tot
        for a in fi: w[a] = fico_target  * market_caps.get(a, 0.0) / fi_tot
    else:
        for a in eq: w[a] = equity_target / max(len(eq), 1)
        for a in fi: w[a] = fico_target  / max(len(fi), 1)
    s = pd.Series(w).reindex(assets).fillna(0.0)
    return s / s.sum() if s.sum() > EPS else s


def equilibrium_returns(cov: pd.DataFrame, w_mkt: pd.Series,
                        risk_aversion: float, rf_annual: float) -> pd.Series:
    """Π = λ·Σ·w_mkt (exceso) + Rf → retorno total de equilibrio."""
    assets    = list(cov.index)
    w         = w_mkt.reindex(assets).fillna(0.0).to_numpy()
    pi_excess = risk_aversion * (cov.to_numpy() @ w)
    return pd.Series(pi_excess + rf_annual, index=assets, name="equilibrium")


# =============================================================================
# VIEWS → P, Q, OMEGA
# =============================================================================

def build_views(assets: Sequence[str], views: Sequence[View]):
    """
    Construye P (K×N), Q (K), y el vector de confianza.
    Q en escala anual. confidence en (0,1].
    """
    assets = list(assets)
    idx    = {a: i for i, a in enumerate(assets)}
    n      = len(assets)
    P, Q, conf, names = [], [], [], []

    def _check(a):
        if a not in idx:
            raise ValueError(f"Activo '{a}' de la view no está en el universo.")

    for i, v in enumerate(views):
        row = np.zeros(n)
        if v.kind == "absolute":
            _check(v.asset)
            row[idx[v.asset]] = 1.0
        elif v.kind == "relative":
            _check(v.long); _check(v.short)
            row[idx[v.long]]  =  1.0
            row[idx[v.short]] = -1.0
        else:
            raise ValueError(f"Tipo de view no soportado: '{v.kind}'.")

        c = float(v.confidence)
        if not (0.0 < c <= 1.0):
            raise ValueError(f"confidence debe estar en (0,1], recibido {c}.")

        P.append(row)
        Q.append(float(v.q))
        conf.append(c)
        names.append(v.name or f"View_{i+1}")

    if not P:
        return (np.zeros((0, n)), np.zeros(0), np.zeros(0), [])
    return np.vstack(P), np.array(Q), np.array(conf), names


def build_omega(P: np.ndarray, cov: np.ndarray, tau: float,
                confidence: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """
    Ω diagonal escalada por confianza (parametrización tipo He-Litterman).
    Alta confianza → Ω pequeña → la view pesa más.
    scale = (1 − c) / c  →  c=1 ⇒ 0 (view casi cierta); c→0 ⇒ ∞ (view ignorada).
    """
    view_var = np.diag(P @ cov @ P.T)
    scale    = (1.0 - confidence) / np.maximum(confidence, EPS)
    # +ε evita Ω exactamente 0 (singular) cuando confidence=1
    omega    = np.diag(tau * view_var * scale + ridge)
    return nearest_psd(omega, epsilon=ridge)


# =============================================================================
# NÚCLEO BLACK-LITTERMAN
# =============================================================================

def black_litterman(cov: pd.DataFrame, pi: pd.Series, P: np.ndarray,
                    Q: np.ndarray, confidence: np.ndarray, config: BLConfig):
    """
    Posterior Black-Litterman.
      μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹·Π + PᵀΩ⁻¹·Q]
      Σ_BL = Σ + M
    Devuelve (ret_bl: Series, sigma_bl: DataFrame).
    """
    assets = list(cov.index)
    n      = len(assets)
    Sigma  = cov.to_numpy(dtype=float)
    Pi     = pi.reindex(assets).to_numpy(dtype=float)
    tau    = config.tau
    ridge  = config.ridge

    if P.shape[0] == 0:                    # sin views → posterior = equilibrio
        return (pi.copy(),
                pd.DataFrame(nearest_psd(Sigma * (1 + tau), ridge),
                             index=assets, columns=assets))

    # Q en exceso sobre Rf, ajustado por exposición neta de cada view
    net_exp  = P @ np.ones(n)
    Q_excess = Q - net_exp * config.rf_annual
    Pi_excess = Pi - config.rf_annual

    Omega    = build_omega(P, Sigma, tau, confidence, ridge)
    tau_cov  = nearest_psd(tau * Sigma, ridge)

    inv_tau_cov = solve_psd(tau_cov, np.eye(n), ridge)
    inv_omega   = solve_psd(Omega, np.eye(P.shape[0]), ridge)

    precision = nearest_psd(inv_tau_cov + P.T @ inv_omega @ P, ridge)
    M         = solve_psd(precision, np.eye(n), ridge)
    mu_excess = M @ (inv_tau_cov @ Pi_excess + P.T @ inv_omega @ Q_excess)

    sigma_bl  = nearest_psd(Sigma + M, ridge)
    ret_bl    = mu_excess + config.rf_annual

    return (pd.Series(ret_bl, index=assets, name="ret_bl"),
            pd.DataFrame(sigma_bl, index=assets, columns=assets))


# =============================================================================
# OPTIMIZACIÓN MEAN-VARIANCE CON RESTRICCIONES
# =============================================================================

def mean_variance_optimize(mu: pd.Series, cov: pd.DataFrame,
                           equity_assets: Sequence[str],
                           fico_assets: Sequence[str],
                           profile: RiskProfile, risk_aversion: float,
                           betas: pd.Series, config: BLConfig):
    """
    max  wᵀμ − (λ/2)·wᵀΣw − γ·(wᵀβ − β_target)²
    s.a. Σw = 1
         Σw_equity = equity_target        (bucket hard)
         Σw_fico   = fico_target
         0 ≤ w_equity ≤ max_weight_equity
         0 ≤ w_fico   ≤ 1

    El beta ya NO es restricción hard: se controla con una penalización
    cuadrática suave centrada en β_target = (beta_min + beta_max) / 2.
    γ = config.gamma_beta gradúa cuánto se castiga desviarse del centro.

    Solver: cvxpy (QP convexo) si está disponible; si no, scipy SLSQP.
    Devuelve (weights: Series, success: bool, message: str).
    """
    assets = list(mu.index)
    n      = len(assets)
    eq_set = set(equity_assets)
    fi_set = set(fico_assets)

    mu_np  = mu.to_numpy(dtype=float)
    S      = nearest_psd(cov.to_numpy(dtype=float), config.ridge)
    beta_v = betas.reindex(assets).fillna(1.0).to_numpy(dtype=float)

    eq_idx = np.array([i for i, a in enumerate(assets) if a in eq_set])
    fi_idx = np.array([i for i, a in enumerate(assets) if a in fi_set])
    if len(eq_idx) == 0: raise ValueError("Sin activos de renta variable.")
    if len(fi_idx) == 0: raise ValueError("Sin activos de renta fija.")

    beta_target = 0.5 * (profile.beta_min + profile.beta_max)
    gamma       = config.gamma_beta
    lo_hi       = [(0.0, config.max_weight_equity) if a in eq_set else (0.0, 1.0)
                   for a in assets]

    # ── Intento primario: cvxpy ──────────────────────────────────────────────
    try:
        import cvxpy as cp

        w = cp.Variable(n)
        objective = (mu_np @ w
                     - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(S))
                     - gamma * cp.square(beta_v @ w - beta_target))
        cons = [
            cp.sum(w) == 1.0,
            cp.sum(w[eq_idx]) == profile.equity_target,
            cp.sum(w[fi_idx]) == profile.fico_target,
            w >= 0.0,
        ]
        for i, (lo, hi) in enumerate(lo_hi):
            cons.append(w[i] <= hi)

        prob = cp.Problem(cp.Maximize(objective), cons)
        prob.solve()

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            wv = np.asarray(w.value, dtype=float)
            wv[np.abs(wv) < 1e-8] = 0.0
            if wv.sum() > EPS:
                wv = wv / wv.sum()
            return (pd.Series(wv, index=assets, name=profile.label),
                    prob.status == "optimal", f"cvxpy: {prob.status}")
        # si cvxpy no converge, cae al respaldo scipy
    except ImportError:
        pass

    # ── Respaldo: scipy SLSQP con la misma penalización ──────────────────────
    def neg_utility(w):
        beta_pen = gamma * (float(w @ beta_v) - beta_target) ** 2
        return -(w @ mu_np) + 0.5 * risk_aversion * (w @ S @ w) + beta_pen

    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w, ix=eq_idx: w[ix].sum() - profile.equity_target},
        {"type": "eq", "fun": lambda w, ix=fi_idx: w[ix].sum() - profile.fico_target},
    ]
    # Multi-start: el equiponderado puede ser un punto estacionario que atrapa
    # a SLSQP. Se prueban varios x0 (incluido uno sesgado por retorno) y se
    # elige la mejor solución factible.
    def _seed_equal():
        x = np.zeros(n)
        x[eq_idx] = profile.equity_target / len(eq_idx)
        x[fi_idx] = profile.fico_target   / len(fi_idx)
        return x

    def _seed_return_tilt():
        # sesga el bucket equity hacia los de mayor mu, respetando el cap
        x = np.zeros(n)
        x[fi_idx] = profile.fico_target / len(fi_idx)
        mu_eq   = mu_np[eq_idx]
        order   = np.argsort(mu_eq)[::-1]
        restante = profile.equity_target
        cap      = config.max_weight_equity
        for k in order:
            take = min(cap, restante)
            x[eq_idx[k]] = take
            restante -= take
            if restante <= EPS:
                break
        return x

    best_w, best_obj, best_ok, best_msg = None, np.inf, False, ""
    for x0 in (_seed_return_tilt(), _seed_equal()):
        res = minimize(neg_utility, x0, method="SLSQP",
                       bounds=lo_hi, constraints=constraints,
                       options={"maxiter": 5000, "ftol": 1e-12, "disp": False})
        w = np.asarray(res.x, dtype=float)
        w[np.abs(w) < 1e-8] = 0.0
        if w.sum() > EPS:
            w = w / w.sum()
        obj = neg_utility(w)
        if obj < best_obj:
            best_w, best_obj, best_ok, best_msg = w, obj, bool(res.success), res.message

    return (pd.Series(best_w, index=assets, name=profile.label),
            best_ok, f"scipy: {best_msg}")


# =============================================================================
# ORQUESTADOR
# =============================================================================

def run_profile(returns: pd.DataFrame, equity_assets: Sequence[str],
                forced_assets: dict[str, ForcedAsset], profile: RiskProfile,
                views: Optional[Sequence[View]] = None,
                config: Optional[BLConfig] = None,
                benchmark_returns: Optional[pd.Series] = None,
                betas: Optional[pd.Series] = None) -> ProfileResult:
    """
    Pipeline completo para un perfil:
      1. Inyecta FICO forzado.
      2. Estima covarianza Ledoit-Wolf.
      3. Calibra λ y calcula equilibrio Π.
      4. Aplica views vía Black-Litterman.
      5. Optimiza Mean-Variance con restricciones.
    """
    config = config or BLConfig()
    views  = views or []

    equity_assets = [a for a in equity_assets if a in returns.columns]
    fico_assets   = list(forced_assets.keys())
    if not equity_assets:
        raise ValueError("No hay activos de renta variable válidos en 'returns'.")

    # 1. Inyectar FICO
    full = inject_forced_assets(returns[equity_assets], forced_assets,
                                config.periods_per_year)
    assets = list(full.columns)

    # 2. Covarianza
    cov = estimate_covariance(full, config.periods_per_year, config.ridge)

    # 3. Equilibrio
    lam   = calibrate_lambda(benchmark_returns, config)
    w_mkt = market_weights(assets, equity_assets, fico_assets,
                           profile.equity_target, profile.fico_target)
    pi    = equilibrium_returns(cov, w_mkt, lam, config.rf_annual)

    # Forzar el retorno del FICO en el ancla de equilibrio
    for ticker, fa in forced_assets.items():
        if ticker in pi.index:
            pi.loc[ticker] = fa.ret_annual

    # 4. Black-Litterman
    P, Q, conf, _ = build_views(assets, views)
    ret_bl, sigma_bl = black_litterman(cov, pi, P, Q, conf, config)

    # Reforzar retorno FICO tras BL (no debe moverse por views de equity)
    for ticker, fa in forced_assets.items():
        if ticker in ret_bl.index:
            ret_bl.loc[ticker] = fa.ret_annual

    # 5. Betas
    if betas is None:
        betas = pd.Series(1.0, index=assets)
        for ticker, fa in forced_assets.items():
            betas.loc[ticker] = fa.beta
    betas = betas.reindex(assets).fillna(1.0)

    # 6. Optimización Mean-Variance
    weights, ok, msg = mean_variance_optimize(
        ret_bl, sigma_bl, equity_assets, fico_assets,
        profile, lam, betas, config)

    # 7. Métricas
    w_np     = weights.to_numpy()
    mu_np    = ret_bl.reindex(weights.index).to_numpy()
    S_np     = sigma_bl.reindex(index=weights.index, columns=weights.index).to_numpy()
    exp_ret  = float(w_np @ mu_np)
    vol      = float(np.sqrt(max(w_np @ S_np @ w_np, EPS)))
    sharpe   = (exp_ret - config.rf_annual) / vol if vol > EPS else float("nan")
    beta_p   = float(w_np @ betas.reindex(weights.index).fillna(1.0).to_numpy())
    eq_w     = float(weights[[a for a in weights.index if a in equity_assets]].sum())
    fi_w     = float(weights[[a for a in weights.index if a in fico_assets]].sum())

    # 8. Validación de factibilidad (matemática, no el flag del solver).
    #    scipy puede reportar mensajes engañosos aunque la solución sea válida;
    #    aquí verificamos directamente lo que de verdad importa.
    checks = []
    tol = 1e-4
    if abs(float(weights.sum()) - 1.0) > tol:
        checks.append(f"pesos suman {weights.sum():.4f}, no 1")
    if abs(eq_w - profile.equity_target) > 1e-2:
        checks.append(f"bucket RV {eq_w:.2%} ≠ target {profile.equity_target:.0%}")
    if abs(fi_w - profile.fico_target) > 1e-2:
        checks.append(f"bucket RF {fi_w:.2%} ≠ target {profile.fico_target:.0%}")
    if (weights < -tol).any():
        checks.append("hay pesos negativos")
    eq_weights = weights[[a for a in weights.index if a in equity_assets]]
    if len(eq_weights) and eq_weights.max() > config.max_weight_equity + tol:
        checks.append(f"cap RV excedido: {eq_weights.max():.2%} > "
                      f"{config.max_weight_equity:.0%}")
    feasible = len(checks) == 0
    feas_report = "OK" if feasible else " | ".join(checks)

    return ProfileResult(
        weights=weights.sort_values(ascending=False),
        exp_return=exp_ret, volatility=vol, sharpe=sharpe, beta=beta_p,
        equity_weight=eq_w, fico_weight=fi_w,
        bl_returns=ret_bl, equilibrium=pi, cov_matrix=sigma_bl,
        profile=profile, success=ok, message=msg,
        feasible=feasible, feasibility_report=feas_report,
    )


# =============================================================================
# FRONTERA EFICIENTE (para graficar; no altera la selección de run_profile)
# =============================================================================

def efficient_frontier(mu: pd.Series, cov: pd.DataFrame,
                       equity_assets: Sequence[str],
                       fico_assets: Sequence[str],
                       profile: RiskProfile, config: BLConfig,
                       n_points: int = 40):
    """
    Barre la frontera eficiente respetando los buckets del perfil.
    Para cada nivel de retorno objetivo, minimiza la varianza.
    Devuelve DataFrame con columnas ['ret', 'vol'] (ambos anualizados).

    Uso típico: graficar la frontera y superponer el punto óptimo que
    devuelve run_profile(). No cambia ninguna lógica del motor.
    """
    assets = list(mu.index)
    n      = len(assets)
    eq_set = set(equity_assets)
    fi_set = set(fico_assets)

    mu_np = mu.to_numpy(dtype=float)
    S     = nearest_psd(cov.to_numpy(dtype=float), config.ridge)

    eq_idx = np.array([i for i, a in enumerate(assets) if a in eq_set])
    fi_idx = np.array([i for i, a in enumerate(assets) if a in fi_set])

    bounds = [(0.0, config.max_weight_equity) if a in eq_set else (0.0, 1.0)
              for a in assets]
    base_cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w, ix=eq_idx: w[ix].sum() - profile.equity_target},
        {"type": "eq", "fun": lambda w, ix=fi_idx: w[ix].sum() - profile.fico_target},
    ]

    x0 = np.zeros(n)
    x0[eq_idx] = profile.equity_target / max(len(eq_idx), 1)
    x0[fi_idx] = profile.fico_target   / max(len(fi_idx), 1)

    # Semilla sesgada por retorno: evita que SLSQP se atasque en el
    # equiponderado al buscar los extremos de la frontera.
    x0_tilt = np.zeros(n)
    x0_tilt[fi_idx] = profile.fico_target / max(len(fi_idx), 1)
    mu_eq   = mu_np[eq_idx]
    order   = np.argsort(mu_eq)[::-1]
    restante, cap = profile.equity_target, config.max_weight_equity
    for k in order:
        take = min(cap, restante)
        x0_tilt[eq_idx[k]] = take
        restante -= take
        if restante <= EPS:
            break

    # Rango de retornos factibles con multi-start (equal + tilt)
    def _extreme(sign):
        best = None
        for seed in (x0, x0_tilt):
            r = minimize(lambda w: sign * (w @ mu_np), seed, method="SLSQP",
                         bounds=bounds, constraints=base_cons,
                         options={"maxiter": 2000, "ftol": 1e-10})
            val = float(r.x @ mu_np)
            if best is None or (sign * val < sign * best[0]):
                best = (val, r.x)
        return best

    r_min = _extreme(+1)[0]
    r_max = _extreme(-1)[0]
    if abs(r_max - r_min) < EPS:
        return pd.DataFrame({"ret": [r_min], "vol": [np.sqrt(x0 @ S @ x0)]})

    pts = []
    for r_target in np.linspace(r_min, r_max, n_points):
        cons = base_cons + [
            {"type": "eq", "fun": lambda w, rt=r_target: (w @ mu_np) - rt}
        ]
        best_w = None
        for seed in (x0_tilt, x0):
            res = minimize(lambda w: w @ S @ w, seed, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 2000, "ftol": 1e-12})
            if res.success or res.status == 9:
                w = np.clip(res.x, 0, None)
                if w.sum() > EPS:
                    w = w / w.sum()
                obj = float(w @ S @ w)
                if best_w is None or obj < best_w[1]:
                    best_w = (w, obj)
        if best_w is not None:
            w = best_w[0]
            pts.append((float(w @ mu_np), float(np.sqrt(max(w @ S @ w, EPS)))))

    if not pts:
        return pd.DataFrame(columns=["ret", "vol"])
    df = pd.DataFrame(pts, columns=["ret", "vol"]).sort_values("vol")
    return df.reset_index(drop=True)
