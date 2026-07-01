"""
Konfigurasi global & parameter valuasi.

CATATAN KEAMANAN:
- SPREADSHEET_ID diambil dari environment variable, dengan fallback ke nilai
  default (ID bukan rahasia — akses tetap dikontrol lewat sharing sheet ke
  email service account). Anda tetap bisa meng-override lewat secret bila mau.
- Kredensial service account TIDAK PERNAH ditulis di kode. Lihat src/sheets.py.
"""
import os

# ── Spreadsheet ────────────────────────────────────────────────────────────
# Boleh di-override via Actions secret / env var SPREADSHEET_ID.
SPREADSHEET_ID = os.environ.get(
    "SPREADSHEET_ID",
    "1X2lQYfPC6Dj8WEwbJsXevZgprXw2f2pi_obpTUZFFn8",
)

# ── DCF Parameters ─────────────────────────────────────────────────────────
RISK_FREE_RATE   = 0.065
EQUITY_RISK_PREM = 0.055
TERMINAL_GROWTH  = 0.045
DCF_YEARS        = 5
WACC_MIN         = 0.08
WACC_MAX         = 0.20

# ── Fair Value Weights ─────────────────────────────────────────────────────
WEIGHT_DCF       = 0.50
WEIGHT_GRAHAM    = 0.30
WEIGHT_EV_EBITDA = 0.20

# ── Signal Thresholds ──────────────────────────────────────────────────────
MOS_MURAH        = 0.20
MOS_MAHAL        = -0.10

# ── Robustness / retry (untuk CI/CD) ───────────────────────────────────────
YF_MAX_RETRIES   = int(os.environ.get("YF_MAX_RETRIES", "3"))
YF_RETRY_DELAY   = float(os.environ.get("YF_RETRY_DELAY", "5"))   # detik
