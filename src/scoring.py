"""
Sistem fundamental scoring (skala 1–10) berbasis SCORING_CONFIG.
Termasuk penentuan tipe saham (bank / karya / umum) dan skala EV/EBITDA.
"""
import numpy as np
import pandas as pd

from .utils import safe_div


# Ditentukan otomatis dari yfinance sector + industry

def get_stock_type(sector, industry):
    """
    bank  → Financial Services / Banks
    karya → Industrials / Engineering & Construction (BUMN Karya)
    bank juga → Financial Services / Credit Services, Insurance, dll
    umum  → semua lainnya
    """
    if sector == "Financial Services":
        return "bank"
    if sector == "Industrials" and industry == "Engineering & Construction":
        return "karya"
    return "umum"

# Definisi scoring: setiap entry = satu rasio yang dinilai
# format: {
#   "rasio"     : nama kolom di data,
#   "label"     : nama tampilan,
#   "tipe"      : "umum" | "bank" | "karya" | "semua" | list tipe,
#   "bobot_umum": float (0-1),  # bobot untuk tipe umum
#   "bobot_bank": float (0-1),  # bobot untuk tipe bank (0 = tidak dipakai)
#   "skala"     : [(batas_bawah, batas_atas, skor), ...]  inklusif kiri
#                 batas_atas=None → tak terbatas ke atas
#                 urutan: dari buruk ke bagus
# }

SCORING_CONFIG = [
    # ════════════════════════════════════════════════════════
    # PROFITABILITAS
    # ════════════════════════════════════════════════════════
    {
        "rasio": "GPM %", "label": "GPM (%)", "tipe": "umum",
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(None,10,1),(10,20,2),(20,30,3),(30,50,4),(50,None,5)],
    },
    {
        "rasio": "GPM %", "label": "GPM Karya (%)", "tipe": "karya",
        "bobot_umum": 0.05, "bobot_bank": 0.00,
        "skala": [(None,5,1),(5,10,2),(10,15,3),(15,20,4),(20,None,5)],
    },
    {
        "rasio": "OPM %", "label": "OPM (%)", "tipe": "umum",
        "bobot_umum": 0.08, "bobot_bank": 0.00,
        "skala": [(None,5,1),(5,10,2),(10,15,3),(15,25,4),(25,None,5)],
    },
    {
        "rasio": "OPM %", "label": "OPM Karya (%)", "tipe": "karya",
        "bobot_umum": 0.07, "bobot_bank": 0.00,
        "skala": [(None,3,1),(3,6,2),(6,10,3),(10,15,4),(15,None,5)],
    },
    {
        "rasio": "NPM %", "label": "NPM (%)", "tipe": ["umum","karya"],
        "bobot_umum": 0.08, "bobot_bank": 0.00,
        "skala": [(None,0,1),(0,5,2),(5,10,3),(10,15,4),(15,None,5)],
    },
    {
        "rasio": "NPM %", "label": "NPM Bank (%)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.08,
        "skala": [(None,10,1),(10,15,2),(15,20,3),(20,30,4),(30,None,5)],
    },
    {
        "rasio": "ROE %", "label": "ROE (%)", "tipe": ["umum","karya"],
        "bobot_umum": 0.12, "bobot_bank": 0.00,
        "skala": [(None,5,1),(5,8,2),(8,12,3),(12,20,4),(20,None,5)],
    },
    {
        "rasio": "ROE %", "label": "ROE Bank (%)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.12,
        "skala": [(None,5,1),(5,8,2),(8,12,3),(12,20,4),(20,None,5)],
    },
    {
        "rasio": "ROA %", "label": "ROA (%)", "tipe": ["umum","karya"],
        "bobot_umum": 0.04, "bobot_bank": 0.00,
        "skala": [(None,1,1),(1,3,2),(3,5,3),(5,10,4),(10,None,5)],
    },
    {
        "rasio": "ROA %", "label": "ROA Bank (%)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.08,
        "skala": [(None,0.5,1),(0.5,1.0,2),(1.0,1.5,3),(1.5,2.5,4),(2.5,None,5)],
    },
    # FCF / Net Income — kualitas laba
    {
        "rasio": "FCF_to_NI", "label": "FCF/Net Income", "tipe": ["umum","karya"],
        "bobot_umum": 0.05, "bobot_bank": 0.00,
        "skala": [(None,0,1),(0,0.3,2),(0.3,0.7,3),(0.7,1.2,4),(1.2,None,5)],
    },
    {
        "rasio": "FCF_to_NI", "label": "FCF/Net Income Bank", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.05,
        "skala": [(None,0,1),(0,0.3,2),(0.3,0.7,3),(0.7,1.2,4),(1.2,None,5)],
    },

    # ════════════════════════════════════════════════════════
    # SOLVABILITAS
    # ════════════════════════════════════════════════════════
    {
        "rasio": "DER (x)", "label": "DER (x)", "tipe": "umum",
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(2.0,None,1),(1.5,2.0,2),(1.0,1.5,3),(0.5,1.0,4),(None,0.5,5)],
        "inverse": True,
    },
    {
        "rasio": "DER (x)", "label": "DER Karya (x)", "tipe": "karya",
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(4.0,None,1),(3.0,4.0,2),(2.0,3.0,3),(1.0,2.0,4),(None,1.0,5)],
        "inverse": True,
    },
    {
        "rasio": "DAR (x)", "label": "DAR (x)", "tipe": ["umum","karya"],
        "bobot_umum": 0.02, "bobot_bank": 0.00,
        "skala": [(0.8,None,1),(0.6,0.8,2),(0.4,0.6,3),(0.2,0.4,4),(None,0.2,5)],
        "inverse": True,
    },
    {
        "rasio": "Interest_Coverage", "label": "Interest Cov. (x)", "tipe": "umum",
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(None,1.0,1),(1.0,2.0,2),(2.0,5.0,3),(5.0,10.0,4),(10.0,None,5)],
    },
    {
        "rasio": "Interest_Coverage", "label": "Interest Cov. Karya (x)", "tipe": "karya",
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(None,1.0,1),(1.0,2.0,2),(2.0,5.0,3),(5.0,10.0,4),(10.0,None,5)],
    },
    # Net Debt / EBITDA
    {
        "rasio": "NetDebt_EBITDA", "label": "Net Debt/EBITDA (x)", "tipe": "umum",
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(5.0,None,1),(3.0,5.0,2),(2.0,3.0,3),(1.0,2.0,4),(None,1.0,5)],
        "inverse": True,
    },
    {
        "rasio": "NetDebt_EBITDA", "label": "Net Debt/EBITDA Karya (x)", "tipe": "karya",
        "bobot_umum": 0.05, "bobot_bank": 0.00,
        "skala": [(8.0,None,1),(6.0,8.0,2),(4.0,6.0,3),(2.0,4.0,4),(None,2.0,5)],
        "inverse": True,
    },
    # Risk Factor & Asset Turnover Bank
    {
        "rasio": "Risk Factor", "label": "Risk Factor", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.17,
        "skala": [(50,None,1),(40,50,2),(30,40,3),(20,30,4),(None,20,5)],
        "inverse": True,
    },
    {
        "rasio": "Asset Turnover (x)", "label": "Asset Turnover Bank (x)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.17,
        "skala": [(None,0.05,1),(0.05,0.08,2),(0.08,0.12,3),(0.12,0.15,4),(0.15,None,5)],
    },

    # ════════════════════════════════════════════════════════
    # VALUASI
    # ════════════════════════════════════════════════════════
    {
        "rasio": "PER (x)", "label": "PER (x)", "tipe": ["umum","karya"],
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(25,None,1),(20,25,2),(15,20,3),(10,15,4),(None,10,5)],
        "inverse": True,
    },
    {
        "rasio": "PER (x)", "label": "PER Bank (x)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.08,
        "skala": [(25,None,1),(20,25,2),(15,20,3),(10,15,4),(None,10,5)],
        "inverse": True,
    },
    {
        "rasio": "PBV (x)", "label": "PBV (x)", "tipe": ["umum","karya"],
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(3.0,None,1),(2.0,3.0,2),(1.0,2.0,3),(0.5,1.0,4),(None,0.5,5)],
        "inverse": True,
    },
    {
        "rasio": "PBV (x)", "label": "PBV Bank (x)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.10,
        "skala": [(4.5,None,1),(3.5,4.5,2),(2.5,3.5,3),(1.5,2.5,4),(None,1.5,5)],
        "inverse": True,
    },
    # EV/EBITDA — recycle dari valuation, skala per sektor di-handle di scoring func
    {
        "rasio": "EV_EBITDA_score_input", "label": "EV/EBITDA (x)", "tipe": ["umum","karya"],
        "bobot_umum": 0.05, "bobot_bank": 0.00,
        "skala": [(20,None,1),(15,20,2),(10,15,3),(7,10,4),(None,7,5)],
        "inverse": True,
    },
    {
        "rasio": "Dividend Yield %", "label": "Div. Yield (%)", "tipe": ["umum","karya"],
        "bobot_umum": 0.06, "bobot_bank": 0.00,
        "skala": [(None,0,1),(0,2,2),(2,4,3),(4,6,4),(6,None,5)],
    },
    {
        "rasio": "Dividend Yield %", "label": "Div. Yield Bank (%)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.05,
        "skala": [(None,0,1),(0,2,2),(2,4,3),(4,6,4),(6,None,5)],
    },

    # ════════════════════════════════════════════════════════
    # PERTUMBUHAN
    # ════════════════════════════════════════════════════════
    {
        "rasio": "Revenue_CAGR3", "label": "Revenue CAGR 3Y (%)", "tipe": ["umum","karya"],
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(None,0,1),(0,5,2),(5,10,3),(10,20,4),(20,None,5)],
    },
    {
        "rasio": "Revenue_CAGR3", "label": "Revenue CAGR 3Y Bank (%)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.05,
        "skala": [(None,0,1),(0,5,2),(5,10,3),(10,20,4),(20,None,5)],
    },
    {
        "rasio": "EPS_CAGR3", "label": "EPS CAGR 3Y (%)", "tipe": ["umum","karya"],
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(None,-10,1),(-10,0,2),(0,8,3),(8,20,4),(20,None,5)],
    },
    {
        "rasio": "EPS_CAGR3", "label": "EPS CAGR 3Y Bank (%)", "tipe": "bank",
        "bobot_umum": 0.00, "bobot_bank": 0.05,
        "skala": [(None,-10,1),(-10,0,2),(0,8,3),(8,20,4),(20,None,5)],
    },

    # ════════════════════════════════════════════════════════
    # LIKUIDITAS & AKTIVITAS
    # ════════════════════════════════════════════════════════
    {
        "rasio": "Current Ratio (x)", "label": "Current Ratio (x)", "tipe": "umum",
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(None,0.8,1),(0.8,1.0,2),(1.0,1.5,3),(1.5,2.5,4),(2.5,None,5)],
    },
    {
        "rasio": "Current Ratio (x)", "label": "Current Ratio Karya (x)", "tipe": "karya",
        "bobot_umum": 0.05, "bobot_bank": 0.00,
        "skala": [(None,1.0,1),(1.0,1.2,2),(1.2,1.5,3),(1.5,2.0,4),(2.0,None,5)],
    },
    {
        "rasio": "Inventory Turnover", "label": "Inventory TO (x)", "tipe": "umum",
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(None,2,1),(2,3,2),(3,5,3),(5,8,4),(8,None,5)],
    },
    {
        "rasio": "Asset Turnover (x)", "label": "Asset Turnover (x)", "tipe": ["umum","karya"],
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(None,0.3,1),(0.3,0.5,2),(0.5,0.8,3),(0.8,1.2,4),(1.2,None,5)],
    },
    {
        "rasio": "Receivables TO", "label": "Receivables TO (x)", "tipe": "umum",
        "bobot_umum": 0.02, "bobot_bank": 0.00,
        "skala": [(None,4,1),(4,6,2),(6,9,3),(9,12,4),(12,None,5)],
    },
    {
        "rasio": "Receivables TO", "label": "Receivables TO Karya (x)", "tipe": "karya",
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(None,2,1),(2,3,2),(3,4,3),(4,6,4),(6,None,5)],
    },
    # CCC — hanya consumer sector, tipe "consumer" dihandle khusus
    {
        "rasio": "CCC", "label": "Cash Conv. Cycle (hari)", "tipe": "consumer",
        "bobot_umum": 0.03, "bobot_bank": 0.00,
        "skala": [(180,None,1),(90,180,2),(30,90,3),(0,30,4),(None,0,5)],
        "inverse": True,
    },
]

# Verifikasi total bobot per stock_type (CCC dikecualikan — conditional)
def _verify_weights():
    def applies(tipe, stype):
        if tipe == "consumer": return False   # skip — conditional
        if tipe == "semua":    return True
        if isinstance(tipe, list): return stype in tipe
        return tipe == stype
    for stype in ["umum", "bank", "karya"]:
        total = sum(
            (c["bobot_bank"] if stype == "bank" else c["bobot_umum"])
            for c in SCORING_CONFIG
            if applies(c["tipe"], stype)
            and (c["bobot_bank"] if stype == "bank" else c["bobot_umum"]) > 0
        )
        # Umum: CCC = 3% untuk Consumer, tapi karena conditional bobot lain normalize
        status = "✅" if abs(total - 1.0) < 0.001 else f"⚠️  diff={total-1.0:+.4f}"
        print(f"  {status}  bobot {stype:5s} = {total:.4f}")




def score_from_skala(value, skala, inverse=False):
    """
    Kembalikan skor 1-5 berdasarkan skala.
    Skala format: list of (lo, hi, skor) — lo=None berarti -inf, hi=None berarti +inf.
    """
    if value is None or pd.isna(value):
        return np.nan
    for lo, hi, s in skala:
        lo_ok = (lo is None) or (value >= lo)
        hi_ok = (hi is None) or (value <  hi)
        if lo_ok and hi_ok:
            return s
    return np.nan

# EV/EBITDA skala per sektor untuk scoring
EV_EBITDA_SCORE_SKALA = {
    "Technology":         [(40,None,1),(30,40,2),(20,30,3),(12,20,4),(None,12,5)],
    "Energy":             [(10,None,1),(8,10,2),(5,8,3),(3,5,4),(None,3,5)],
    "Basic Materials":    [(10,None,1),(8,10,2),(5,8,3),(3,5,4),(None,3,5)],
    # default umum untuk sektor lain
}
EV_EBITDA_SCORE_SKALA_DEFAULT = [(20,None,1),(15,20,2),(10,15,3),(7,10,4),(None,7,5)]

# Sektor consumer untuk CCC
CONSUMER_SECTORS = {"Consumer Defensive", "Consumer Cyclical"}

def calc_fundamental_score(data_row, stock_type, sector=""):
    """
    data_row : dict rasio untuk 1 periode, termasuk kunci baru:
               FCF_to_NI, NetDebt_EBITDA, Revenue_CAGR3, EPS_CAGR3,
               CCC, EV_EBITDA_score_input
    stock_type: "bank" | "karya" | "umum"
    sector    : normalized yfinance sector (untuk CCC & EV/EBITDA skala)
    Returns   : (score_1to10, dict komponen)
    """
    components     = {}
    total_weighted = 0.0
    total_weight   = 0.0
    is_consumer    = sector in CONSUMER_SECTORS

    for cfg in SCORING_CONFIG:
        tipe = cfg["tipe"]

        # ── Cek applicable ────────────────────────────────────
        if tipe == "consumer":
            # CCC: hanya untuk sektor consumer DAN stock_type umum
            applicable = (is_consumer and stock_type == "umum")
        elif tipe == "semua":
            applicable = True
        elif isinstance(tipe, list):
            applicable = stock_type in tipe
        else:
            applicable = (tipe == stock_type)

        if not applicable:
            continue

        bobot = cfg["bobot_bank"] if stock_type == "bank" else cfg["bobot_umum"]
        if bobot == 0:
            continue

        raw_val = data_row.get(cfg["rasio"], np.nan)
        try:
            raw_val = float(raw_val)
        except:
            raw_val = np.nan

        # ── Skala override untuk EV/EBITDA ────────────────────
        if cfg["rasio"] == "EV_EBITDA_score_input":
            skala = EV_EBITDA_SCORE_SKALA.get(sector, EV_EBITDA_SCORE_SKALA_DEFAULT)
        else:
            skala = cfg["skala"]

        # ── Pre-processing khusus ──────────────────────────────
        # FCF/NI: cap atas 3.0 agar tidak bias dari net income sangat kecil
        if cfg["rasio"] == "FCF_to_NI" and not pd.isna(raw_val):
            raw_val = min(raw_val, 3.0)
        # NetDebt/EBITDA: jika negatif (net cash) → skor otomatis 5
        if cfg["rasio"] == "NetDebt_EBITDA" and not pd.isna(raw_val) and raw_val < 0:
            raw_val = -1.0  # masuk bucket (None, 1.0) → skor 5
        # CAGR: konversi ke persen jika masih dalam desimal
        if cfg["rasio"] in ("Revenue_CAGR3", "EPS_CAGR3") and not pd.isna(raw_val):
            if abs(raw_val) < 5:   # kemungkinan masih desimal (0.15 bukan 15)
                raw_val = raw_val * 100

        skor = score_from_skala(raw_val, skala, inverse=cfg.get("inverse", False))

        components[cfg["label"]] = {
            "raw":        raw_val,
            "skor_1to5":  skor,
            "bobot":      bobot,
            "kontribusi": skor * bobot if not pd.isna(skor) else 0.0,
        }
        if not pd.isna(skor):
            total_weighted += skor * bobot
            total_weight   += bobot

    # Normalize & skala ke 1-10
    normalized_5 = safe_div(total_weighted, total_weight, default=np.nan)
    score_1to10  = round(normalized_5 * 2, 2) if not pd.isna(normalized_5) else np.nan
    return score_1to10, components

def score_label(score):
    if pd.isna(score):     return "N/A"
    if score >= 8.0:       return "⭐ SANGAT BAGUS"
    if score >= 6.5:       return "✅ BAGUS"
    if score >= 5.0:       return "🟡 CUKUP"
    if score >= 3.5:       return "⚠️ KURANG"
    return "❌ BURUK"

