#!/usr/bin/env python3
"""
Entrypoint pipeline valuasi IDX.

Menjalankan seluruh sektor (atau sebagian lewat env var SECTORS), meng-upload
hasil ke Google Sheets, dan mengembalikan exit code:
  0  -> minimal satu sektor sukses
  1  -> tidak ada sektor yang sukses (workflow ditandai gagal)

Contoh:
    python main.py                       # semua sektor
    SECTORS="TECHNO,ENERGY" python main.py   # sebagian sektor
"""
import os
import sys
import traceback

from src.sectors import SECTORS
from src.sheets import get_client, open_spreadsheet
from src.pipeline import run_sector


def _selected_sectors():
    raw = os.environ.get("SECTORS", "").strip()
    if not raw:
        return list(SECTORS.keys())
    picked = [s.strip().upper() for s in raw.split(",") if s.strip()]
    invalid = [s for s in picked if s not in SECTORS]
    if invalid:
        print(f"⚠️  Sektor tidak dikenal diabaikan: {invalid}")
    return [s for s in picked if s in SECTORS]


def main():
    codes = _selected_sectors()
    if not codes:
        print("❌ Tidak ada sektor valid untuk diproses.")
        return 1

    print(f"🔐 Otentikasi Google Sheets...")
    try:
        gc = get_client()
        sh = open_spreadsheet(gc)
    except Exception as e:
        print(f"❌ Gagal otentikasi / buka spreadsheet: {e}")
        return 1
    print("✅ Terhubung ke spreadsheet.\n")

    print(f"🚀 Memproses {len(codes)} sektor: {', '.join(codes)}")
    summaries = []
    for code in codes:
        try:
            summaries.append(run_sector(code, SECTORS[code], sh))
        except Exception as e:
            # Satu sektor gagal tidak menghentikan sektor lain.
            print(f"❌ Sektor {code} error: {e}")
            traceback.print_exc()
            summaries.append({"sector": code, "rows": 0, "status": "ERROR"})

    ok = [s for s in summaries if s["status"] == "OK"]
    print("\n" + "=" * 60)
    print("RINGKASAN")
    for s in summaries:
        print(f"  {s['status']:6s} | {s['sector']:8s} | {s['rows']} baris")
    print(f"Sukses {len(ok)}/{len(summaries)} sektor.")
    print("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
