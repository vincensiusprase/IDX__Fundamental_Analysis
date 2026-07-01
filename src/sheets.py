"""
Modul output/penyimpanan: otentikasi Google Sheets + upload DataFrame.

KEAMANAN — cara kredensial dibaca (berurutan):
  1. Env var GCP_SA_KEY / GOOGLE_SERVICE_ACCOUNT_JSON  -> isi JSON service account
     (dipakai di GitHub Actions; disimpan sebagai repository secret).
  2. File service_account.json di root repo (dipakai saat run lokal;
     WAJIB masuk .gitignore, jangan pernah di-commit).

Tidak ada kredensial yang ditulis di dalam kode.
"""
import os
import json

import numpy as np
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials

from .config import SPREADSHEET_ID

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SERVICE_ACCOUNT_FILE = os.environ.get("SERVICE_ACCOUNT_FILE", "service_account.json")


def get_client():
    """Kembalikan gspread client terotentikasi, atau raise bila kredensial tak ada."""
    raw = os.environ.get("GCP_SA_KEY") or os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        creds = Credentials.from_service_account_info(json.loads(raw), scopes=SCOPES)
    elif os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    else:
        raise RuntimeError(
            "Kredensial tidak ditemukan. Set env var GCP_SA_KEY (isi JSON service "
            "account) atau sediakan file service_account.json di root repo."
        )
    return gspread.authorize(creds)


def open_spreadsheet(gc):
    """Buka spreadsheet berdasarkan SPREADSHEET_ID."""
    return gc.open_by_key(SPREADSHEET_ID)


def upload_sheet(sh, df, name, rows=1000, cols=60):
    """Tulis DataFrame ke worksheet `name` (dibuat bila belum ada)."""
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=rows, cols=cols)
    ws.clear()
    set_with_dataframe(ws, df, include_index=False)
    ws.freeze(rows=1)
    print(f"   ✅ Sheet '{name}': {len(df)} baris, {len(df.columns)} kolom")


def sort_and_clean(df):
    """Urutkan per Ticker & Year (TTM di atas) dan bersihkan inf/-inf."""
    df = df.copy()
    df["_sy"] = df["Year"].apply(lambda x: 9999 if str(x) == "TTM" else int(x))
    df = df.sort_values(["Ticker", "_sy"], ascending=[True, False]).drop(columns=["_sy"])
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)
