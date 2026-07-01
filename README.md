# 📊 IDX Valuation Pipeline

Sistem valuasi fundamental multi-metode untuk emiten Bursa Efek Indonesia (IDX),
di-refactor agar modular, aman, dan siap jalan otomatis via GitHub Actions.

## Struktur

```
.
├── main.py                     # entrypoint (jalankan semua/sebagian sektor)
├── requirements.txt
├── .gitignore                  # memblokir commit kredensial
├── .env.example
├── src/
│   ├── config.py               # parameter DCF, bobot, threshold, id spreadsheet
│   ├── sectors.py              # daftar saham + nama sheet per sektor (950 saham)
│   ├── mappings.py             # peta sektor/industri, konstanta Graham, target EV
│   ├── utils.py                # PENGAMBILAN DATA: helper ekstraksi nilai yfinance
│   ├── valuation.py            # PERHITUNGAN: Graham, DCF/EV-Sales, EV/EBITDA, fair value
│   ├── scoring.py              # PERHITUNGAN: fundamental score 1–10
│   ├── backtest.py             # PERHITUNGAN: enrichment backtest historis
│   ├── sheets.py               # OUTPUT/PENYIMPANAN: auth + upload Google Sheets
│   └── pipeline.py             # orkestrasi: tarik data yfinance + rangkai semua
└── .github/workflows/
    ├── main.yml                # workflow single-job (semua sektor)
    └── matrix.yml.example      # varian paralel per sektor (rekomendasi)
```

**Menggantikan 11 file `IDX*_VALUATION.py` yang 99% identik.** Untuk mengubah
daftar saham, cukup edit `src/sectors.py` — logika perhitungan tidak disentuh.

## Setup GitHub Actions (langkah demi langkah)

### 1. Buat Service Account Google
1. Buka <https://console.cloud.google.com> → buat / pilih project.
2. **APIs & Services → Enable APIs** → aktifkan **Google Sheets API** dan **Google Drive API**.
3. **IAM & Admin → Service Accounts → Create Service Account**.
4. Setelah dibuat, buka service account → tab **Keys → Add Key → JSON**. File JSON terunduh.
5. Salin **email service account** (mis. `bot@project.iam.gserviceaccount.com`).

### 2. Share spreadsheet ke service account
Buka spreadsheet tujuan → **Share** → tempel email service account tadi → beri
akses **Editor**. Tanpa langkah ini, upload akan gagal walau kredensial benar.

### 3. Simpan kredensial sebagai secret (JANGAN commit file JSON)
Di repo GitHub: **Settings → Secrets and variables → Actions**.
- Tab **Secrets → New repository secret**
  - Name: `GCP_SA_KEY`
  - Value: **tempel seluruh isi file JSON** service account.
- (Opsional) Tab **Variables → New variable**
  - Name: `SPREADSHEET_ID`, Value: id spreadsheet Anda.

### 4. Commit kode & aktifkan workflow
`main.yml` sudah berisi jadwal **Sabtu 13:30 WIB** (`cron: "30 6 * * 6"`, UTC) dan
tombol manual. Setelah di-push, buka tab **Actions** → pilih workflow → **Run workflow**
untuk uji jalan pertama tanpa menunggu jadwal.

> **Catatan jadwal:** GitHub Actions memakai UTC (WIB = UTC+7). Jika yang Anda maksud
> **01:30 WIB Sabtu**, gantilah cron menjadi `"30 18 * * 5"` (Jumat 18:30 UTC).

### Run lokal
```bash
pip install -r requirements.txt
# taruh file service_account.json di root repo (sudah di-gitignore)
python main.py                       # semua sektor
SECTORS="TECHNO,ENERGY" python main.py   # sebagian sektor
```

## Error handling
- **Per-ticker**: gagal 1 emiten (data kosong / koneksi) hanya dilewati.
- **Per-sektor**: gagal 1 sektor tidak menggagalkan sektor lain.
- **Retry**: pengambilan history harga otomatis dicoba ulang (`YF_MAX_RETRIES`).
- **Exit code**: workflow hanya ditandai gagal bila **tidak ada** sektor yang sukses.

## ⚠️ Catatan keamanan (repo publik)
| Item | Status | Tindakan |
|------|--------|----------|
| Service account JSON | 🔴 Rahasia | **Wajib** lewat secret `GCP_SA_KEY`, jangan commit. `.gitignore` sudah memblokir `service_account.json`. |
| Spreadsheet ID | 🟡 Bukan rahasia | Sekadar identifier; akses tetap dikontrol lewat sharing. Boleh dipublik, atau sembunyikan via `vars.SPREADSHEET_ID`. |
| API Key | 🟢 Tidak dipakai | Pipeline pakai OAuth service account, bukan API key statis. |

Jika file JSON pernah ter-commit sebelumnya: **rotasi/hapus key itu** di Google
Cloud Console (buat key baru) — menghapus dari commit terbaru saja tidak cukup
karena masih ada di riwayat git.
