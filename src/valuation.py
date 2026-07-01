"""
Logika perhitungan valuasi:
Graham, DCF (dengan fallback EV/Sales), EV/EBITDA, dan fair value gabungan.
"""
import numpy as np
import pandas as pd

from .config import (
    RISK_FREE_RATE, EQUITY_RISK_PREM, TERMINAL_GROWTH, DCF_YEARS,
    WACC_MIN, WACC_MAX, WEIGHT_DCF, WEIGHT_GRAHAM, WEIGHT_EV_EBITDA,
    MOS_MURAH, MOS_MAHAL,
)
from .mappings import (
    GRAHAM_CONSTANTS, GRAHAM_K_DEFAULT,
    EV_EBITDA_TARGETS, EV_EBITDA_DEFAULT,
    EV_SALES_TARGETS, EV_SALES_DEFAULT,
)
from .utils import safe_div


def get_graham_constants(sector, industry):
    key = (sector, industry)
    if key in GRAHAM_CONSTANTS:
        return GRAHAM_CONSTANTS[key]
    # Fallback level 2: cocok sektor saja (ambil entry pertama)
    for (s, i), v in GRAHAM_CONSTANTS.items():
        if s == sector:
            return v
    # Fallback level 3: tidak ditemukan sama sekali → gunakan K=22.5 (Graham original)
    print(f"      ⚠️  Graham fallback K=22.5 untuk sector='{sector}' industry='{industry}'")
    return {"K": GRAHAM_K_DEFAULT, "target_pe": 15.0, "target_pb": 1.5}

def get_valid_methods(sector):
    if sector == "Financial Services":
        return {"graham": True, "dcf": False, "ev_ebitda": False}
    elif sector == "Technology":
        return {"graham": True, "dcf": True, "ev_ebitda": True}
    elif sector == "Energy":
        return {"graham": True, "dcf": False, "ev_ebitda": True}
    elif sector == "Real Estate":
        return {"graham": True, "dcf": True, "ev_ebitda": False}
    else:
        return {"graham": True, "dcf": True, "ev_ebitda": True}

# ════════════════════════════════════════════════════════════════

def calc_graham_value(eps, bvps, sector, industry):
    constants = get_graham_constants(sector, industry)
    if constants is None:
        return np.nan, np.nan
    K, target_pb = constants["K"], constants["target_pb"]
    if sector == "Financial Services":
        return (target_pb * bvps, K) if bvps > 0 else (np.nan, K)
    if eps <= 0 or bvps <= 0:
        return np.nan, K
    return np.sqrt(K * eps * bvps), K

def calc_dcf_value(fcf_total_series, shares, beta, debt, cash,
                   revenue_total=np.nan, sector=""):
    """
    DCF berbasis FCF historis.
    Jika FCF positif < 2 tahun (growth/loss-making), fallback ke EV/Sales.
    Returns: (value_per_share, wacc, cagr_or_note)
      cagr_or_note: float jika DCF normal, string "EV/Sales" jika fallback
    """
    wacc = np.clip(RISK_FREE_RATE + beta * EQUITY_RISK_PREM, WACC_MIN, WACC_MAX)

    # ── Cek apakah FCF cukup untuk DCF ───────────────────────────
    has_fcf = False
    if fcf_total_series is not None and len(fcf_total_series) >= 2:
        clean = fcf_total_series.dropna()
        clean = clean[clean > 0]
        if len(clean) >= 2:
            has_fcf = True

    if has_fcf:
        # ── DCF Normal ───────────────────────────────────────────
        n    = len(clean) - 1
        if n <= 0:
            return np.nan, wacc, np.nan
        cagr = np.clip((clean.iloc[-1] / clean.iloc[0]) ** (1/n) - 1, -0.20, wacc * 2)
        base = clean.iloc[-1]
        pv   = sum(base * (1+cagr)**t / (1+wacc)**t for t in range(1, DCF_YEARS+1))
        tv   = base * (1+cagr)**DCF_YEARS * (1+TERMINAL_GROWTH) / (wacc - TERMINAL_GROWTH)
        ev   = pv + tv / (1+wacc)**DCF_YEARS
        val  = safe_div(ev - debt + cash, shares)
        return (val if val > 0 else np.nan), wacc, cagr

    else:
        # ── Fallback: EV/Sales ───────────────────────────────────
        # Pakai ketika FCF tidak cukup positif (startup, growth, loss-making)
        if pd.isna(revenue_total) or revenue_total <= 0 or shares <= 0:
            return np.nan, wacc, np.nan
        mult    = EV_SALES_TARGETS.get(sector, EV_SALES_DEFAULT)
        net_dbt = (debt - cash) if not pd.isna(debt) and not pd.isna(cash) else 0
        ev_val  = mult * revenue_total - net_dbt
        val     = safe_div(ev_val, shares)
        # cagr diganti string marker agar tahu ini EV/Sales
        return (val if val > 0 else np.nan), wacc, "EV/Sales"

def calc_ev_ebitda_value(ebitda, debt, cash, shares, sector):
    if ebitda <= 0 or shares <= 0:
        return np.nan, np.nan
    mult = EV_EBITDA_TARGETS.get(sector, EV_EBITDA_DEFAULT)
    val  = safe_div(mult * ebitda - (debt - cash), shares)
    return (val if val > 0 else np.nan), mult

def calc_weighted_fair_value(g_val, d_val, ev_val):
    vals    = {"graham": g_val, "dcf": d_val, "ev": ev_val}
    weights = {"graham": WEIGHT_GRAHAM, "dcf": WEIGHT_DCF, "ev": WEIGHT_EV_EBITDA}
    valid   = {k: v for k, v in vals.items()
               if v is not None and not (isinstance(v, float) and np.isnan(v))}
    if not valid:
        return np.nan
    total_w = sum(weights[k] for k in valid)
    return sum(v * weights[k] / total_w for k, v in valid.items())

def get_signal(price, fair_value):
    if not fair_value or np.isnan(fair_value) or price <= 0:
        return "N/A"
    up = (fair_value - price) / price
    return "🟢 MURAH" if up >= MOS_MURAH else ("🔴 MAHAL" if up <= MOS_MAHAL else "🟡 WAJAR")

def fmt_cagr(cagr_val):
    """Format CAGR: float → persen, string 'EV/Sales' → tampilkan apa adanya."""
    if cagr_val is None:
        return np.nan
    if isinstance(cagr_val, str):
        return cagr_val          # "EV/Sales"
    if pd.isna(cagr_val):
        return np.nan
    return round(cagr_val * 100, 2)

