# -*- coding: utf-8 -*-
"""
projections.py — Monte Carlo y Stress Testing para portafolios
===============================================================

Módulo autocontenido, sin dependencias de Streamlit.
Recibe pesos, retornos esperados (BL), covarianza y retornos históricos.

Uso:
    from projections import monte_carlo, stress_test, CRISIS_PERIODS

    mc = monte_carlo(weights, mu_bl, cov_bl, capital, horizon_years=3)
    st = stress_test(weights, returns, CRISIS_PERIODS, capital)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# PERÍODOS DE CRISIS PARA STRESS TESTING
# =============================================================================

@dataclass
class CrisisPeriod:
    """Define un período histórico de crisis."""
    name:  str
    start: str       # fecha inicio 'YYYY-MM-DD'
    end:   str       # fecha fin 'YYYY-MM-DD'
    description: str = ""


# Crisis predefinidas (el usuario puede añadir las suyas)
CRISIS_PERIODS = [
    CrisisPeriod("COVID-19", "2020-02-19", "2020-03-23",
                 "Caída por pandemia global. S&P 500 -34% en 23 días."),
    CrisisPeriod("Taper Tantrum 2013", "2013-05-22", "2013-06-24",
                 "Bernanke señala reducción de QE. Emergentes golpeados."),
    CrisisPeriod("Sell-off Q4 2018", "2018-10-01", "2018-12-24",
                 "Temores de subida de tasas + guerra comercial."),
    CrisisPeriod("Flash Crash 2020", "2020-02-19", "2020-02-28",
                 "Primera semana de pánico COVID, antes del fondo."),
    CrisisPeriod("Russia-Ukraine 2022", "2022-02-18", "2022-03-08",
                 "Invasión de Ucrania. Commodities y mercados volátiles."),
    CrisisPeriod("SVB / Banking 2023", "2023-03-08", "2023-03-15",
                 "Colapso de Silicon Valley Bank. Contagio bancario."),
    CrisisPeriod("Yen Carry Unwind 2024", "2024-07-31", "2024-08-05",
                 "Subida del BOJ + unwind de carry trade en yen."),
]


# =============================================================================
# MONTE CARLO
# =============================================================================

@dataclass
class MonteCarloResult:
    """Resultado de la simulación Monte Carlo."""
    paths:        np.ndarray       # (n_sims, n_steps+1) trayectorias de capital
    dates:        pd.DatetimeIndex # eje temporal
    percentiles:  dict             # {p: array} para cada percentile
    median_path:  np.ndarray
    mean_path:    np.ndarray
    terminal:     np.ndarray       # valores terminales (n_sims,)
    prob_loss:    float            # P(capital final < capital inicial)
    prob_target:  float            # P(capital final >= target) si se define
    var_terminal: float            # VaR 95% del valor terminal
    cvar_terminal: float           # CVaR 95% del valor terminal
    capital:      float
    target:       float
    horizon_years: float


def monte_carlo(weights: pd.Series, mu_bl: pd.Series, cov_bl: pd.DataFrame,
                capital: float = 100_000, horizon_years: float = 3.0,
                periods_per_year: int = 52, n_sims: int = 5000,
                target: Optional[float] = None,
                seed: int = 42) -> MonteCarloResult:
    """
    Simulación Monte Carlo del portafolio.

    Genera n_sims trayectorias de capital usando retornos multivariados
    normales con μ y Σ del Black-Litterman (semanales).

    Parameters
    ----------
    weights : pd.Series — pesos del portafolio (incluye FICO)
    mu_bl : pd.Series — retornos esperados anuales BL
    cov_bl : pd.DataFrame — covarianza anual BL
    capital : float — inversión inicial
    horizon_years : float — horizonte en años
    periods_per_year : int — 52 semanal, 252 diario
    n_sims : int — número de simulaciones
    target : float — capital objetivo (para calcular prob_target)
    seed : int — semilla para reproducibilidad
    """
    if target is None:
        target = capital * 1.10  # default: 10% de ganancia

    assets = list(weights.index)
    w = weights.reindex(assets).fillna(0.0).to_numpy()

    # Retornos y covarianza por período (semanal)
    mu_period = mu_bl.reindex(assets).fillna(0.0).to_numpy() / periods_per_year
    cov_period = cov_bl.reindex(index=assets, columns=assets).fillna(0.0).to_numpy() / periods_per_year

    # Regularizar covarianza para Cholesky
    cov_period = 0.5 * (cov_period + cov_period.T)
    eigvals = np.linalg.eigvalsh(cov_period)
    if eigvals.min() < 1e-10:
        cov_period += np.eye(len(assets)) * (1e-8 - min(eigvals.min(), 0))

    n_steps = int(horizon_years * periods_per_year)

    rng = np.random.default_rng(seed)
    # Generar retornos multivariados: (n_sims, n_steps, n_assets)
    sims = rng.multivariate_normal(mu_period, cov_period, size=(n_sims, n_steps))

    # Retorno del portafolio por período: (n_sims, n_steps)
    port_rets = sims @ w

    # Trayectorias de capital: (n_sims, n_steps+1)
    cum_rets = np.cumsum(port_rets, axis=1)       # log-retornos acumulados
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = capital
    paths[:, 1:] = capital * np.exp(cum_rets)

    # Percentiles
    pcts = {5: None, 10: None, 25: None, 50: None, 75: None, 90: None, 95: None}
    for p in pcts:
        pcts[p] = np.percentile(paths, p, axis=0)

    terminal = paths[:, -1]
    median_path = pcts[50]
    mean_path = paths.mean(axis=0)

    # Probabilidades
    prob_loss   = float((terminal < capital).mean())
    prob_target = float((terminal >= target).mean())

    # VaR y CVaR del valor terminal (pérdida respecto a capital inicial)
    losses = capital - terminal                  # positivo = pérdida
    var_95  = float(np.percentile(losses, 95))
    tail    = losses[losses >= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    # Eje temporal
    start = pd.Timestamp.today()
    dates = pd.date_range(start, periods=n_steps + 1,
                          freq="W" if periods_per_year == 52 else "B")

    return MonteCarloResult(
        paths=paths, dates=dates, percentiles=pcts,
        median_path=median_path, mean_path=mean_path,
        terminal=terminal, prob_loss=prob_loss, prob_target=prob_target,
        var_terminal=var_95, cvar_terminal=cvar_95,
        capital=capital, target=target, horizon_years=horizon_years,
    )


# =============================================================================
# STRESS TESTING
# =============================================================================

@dataclass
class StressResult:
    """Resultado de un escenario de stress."""
    name:         str
    description:  str
    start:        str
    end:          str
    port_return:  float       # retorno total del portafolio en el período
    port_loss:    float       # pérdida en USD (negativo = ganancia)
    max_drawdown: float       # drawdown máximo dentro del período
    benchmark_return: float   # retorno del benchmark en el mismo período
    asset_returns: pd.Series  # retorno por activo en el período
    available:    bool        # ¿había datos para este período?
    n_periods:    int         # semanas/días en el período


def stress_test(weights: pd.Series, returns: pd.DataFrame,
                crises: Sequence[CrisisPeriod], capital: float = 100_000,
                forced_assets: Optional[dict] = None,
                periods_per_year: int = 52,
                benchmark: Optional[pd.Series] = None) -> list[StressResult]:
    """
    Aplica escenarios históricos de crisis al portafolio actual.

    Para cada crisis, toma los retornos reales del período y calcula
    el impacto en el portafolio con los pesos actuales.
    """
    forced_assets = forced_assets or {}
    results = []

    for crisis in crises:
        mask = (returns.index >= crisis.start) & (returns.index <= crisis.end)
        period_rets = returns.loc[mask]

        if len(period_rets) < 1:
            results.append(StressResult(
                name=crisis.name, description=crisis.description,
                start=crisis.start, end=crisis.end,
                port_return=0.0, port_loss=0.0, max_drawdown=0.0,
                benchmark_return=0.0, asset_returns=pd.Series(dtype=float),
                available=False, n_periods=0,
            ))
            continue

        # Retorno por activo en el período
        asset_rets = {}
        for a in weights.index:
            if a in period_rets.columns:
                asset_rets[a] = float(period_rets[a].sum())  # log-ret acumulado
            elif a in forced_assets:
                fa = forced_assets[a]
                per_ret = np.log(1 + fa.ret_annual) / periods_per_year
                asset_rets[a] = per_ret * len(period_rets)
            else:
                asset_rets[a] = 0.0

        asset_rets_s = pd.Series(asset_rets)

        # Retorno del portafolio
        w = weights.reindex(asset_rets_s.index).fillna(0.0)
        port_cum_ret = float((w * asset_rets_s).sum())
        port_loss = capital * (np.exp(port_cum_ret) - 1)

        # Drawdown dentro del período
        daily_port = pd.Series(0.0, index=period_rets.index)
        for a in weights.index:
            if a in period_rets.columns:
                daily_port += weights[a] * period_rets[a].fillna(0.0)
            elif a in forced_assets:
                fa = forced_assets[a]
                daily_port += weights[a] * (np.log(1 + fa.ret_annual) / periods_per_year)
        wealth = np.exp(daily_port.cumsum())
        dd = float((wealth / wealth.cummax() - 1).min())

        # Benchmark
        bmk_ret = 0.0
        if benchmark is not None:
            bmk_mask = benchmark.index.isin(period_rets.index)
            if bmk_mask.any():
                bmk_ret = float(benchmark.loc[bmk_mask].sum())

        results.append(StressResult(
            name=crisis.name, description=crisis.description,
            start=crisis.start, end=crisis.end,
            port_return=np.exp(port_cum_ret) - 1,
            port_loss=port_loss,
            max_drawdown=dd,
            benchmark_return=np.exp(bmk_ret) - 1,
            asset_returns=asset_rets_s.apply(lambda x: np.exp(x) - 1),
            available=True,
            n_periods=len(period_rets),
        ))

    return results
