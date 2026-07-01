"""Enrichment backtest historis untuk tiap baris valuasi (skip TTM)."""
import numpy as np
import pandas as pd


def get_trading_day(hist_df, target_date):
    """
    Ambil hari trading terdekat >= target_date.
    hist_df index harus sudah tz-naive.
    """
    h = hist_df.copy()
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h = h.sort_index()
    future = h[h.index >= target_date]
    if future.empty:
        return None, None
    return future.index[0], future.iloc[0]


def calc_vwap_first_week(hist_df, start_date):
    """
    VWAP = rata-rata Close × Volume selama 5 hari trading pertama mulai start_date.
    Jika Volume tidak tersedia, pakai rata-rata Close saja.
    """
    h = hist_df.copy()
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    h = h.sort_index()
    week = h[h.index >= start_date].head(5)
    if week.empty:
        return np.nan
    if 'Volume' in week.columns and week['Volume'].sum() > 0:
        vwap = (week['Close'] * week['Volume']).sum() / week['Volume'].sum()
    else:
        vwap = week['Close'].mean()
    return round(vwap, 2)


def run_backtest(df_val, hist_cache):
    """
    Enrichment backtest untuk setiap baris df_val.
    - Skip jika Year == 'TTM'
    - Window: 1 Apr (T+1) → 31 Des (T+1)
    - Entry: VWAP 5 hari trading pertama April T+1
    - High/Low: dari harga Close dalam window
    - Close: harga Close hari trading terakhir Desember T+1
    - Return cols: to_high, to_close, to_yearend (same as to_close window ini),
                   max_drawdown dari entry ke low

    hist_cache: dict {ticker: history_df} sudah diambil sebelumnya
    """
    BT_COLS = [
        'BT_Entry_Price', 'BT_Entry_Date',
        'BT_High_Price',  'BT_High_Date',
        'BT_Low_Price',   'BT_Low_Date',
        'BT_Close_Price', 'BT_Close_Date',
        'BT_Return_to_High (%)',
        'BT_Return_to_Close (%)',
        'BT_Max_Drawdown (%)',
        'BT_Data_Status',
    ]

    results = []
    today = pd.Timestamp.today().normalize()

    for _, row in df_val.iterrows():
        rec = {c: np.nan for c in BT_COLS}
        rec['BT_Data_Status'] = 'N/A'

        year = row['Year']

        # Skip TTM
        if str(year) == 'TTM':
            rec['BT_Data_Status'] = 'SKIP_TTM'
            results.append(rec)
            continue

        ticker = row['Ticker']
        try:
            yr = int(year)
        except:
            results.append(rec)
            continue

        # Window: 1 Apr T+1 → 31 Des T+1
        window_start = pd.Timestamp(yr + 1, 4,  1)
        window_end   = pd.Timestamp(yr + 1, 12, 31)

        # Belum terjadi
        if window_start > today:
            rec['BT_Data_Status'] = 'FUTURE'
            results.append(rec)
            continue

        # Ambil history dari cache
        hist = hist_cache.get(ticker)
        if hist is None or hist.empty:
            rec['BT_Data_Status'] = 'NO_DATA'
            results.append(rec)
            continue

        h = hist.copy()
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        h = h.sort_index()

        # Slice window
        window_data = h[(h.index >= window_start) & (h.index <= window_end)]

        if window_data.empty:
            rec['BT_Data_Status'] = 'NO_WINDOW_DATA'
            results.append(rec)
            continue

        # Partial window (window_end belum terjadi = masih dalam tahun berjalan)
        is_partial = window_end > today

        # ── Entry: VWAP 5 hari trading pertama April ──
        entry_price = calc_vwap_first_week(h, window_start)
        first_day, _ = get_trading_day(h, window_start)
        rec['BT_Entry_Price'] = entry_price
        rec['BT_Entry_Date']  = first_day.strftime('%Y-%m-%d') if first_day is not None else np.nan

        if pd.isna(entry_price) or entry_price <= 0:
            rec['BT_Data_Status'] = 'NO_ENTRY'
            results.append(rec)
            continue

        # ── High dalam window ──
        high_idx   = window_data['Close'].idxmax()
        high_price = window_data['Close'].max()
        rec['BT_High_Price'] = round(high_price, 2)
        rec['BT_High_Date']  = high_idx.strftime('%Y-%m-%d')

        # ── Low dalam window ──
        low_idx   = window_data['Close'].idxmin()
        low_price = window_data['Close'].min()
        rec['BT_Low_Price'] = round(low_price, 2)
        rec['BT_Low_Date']  = low_idx.strftime('%Y-%m-%d')

        # ── Close: hari trading terakhir Desember T+1 ──
        dec_data = window_data[window_data.index.month == 12]
        if not dec_data.empty:
            close_price = dec_data['Close'].iloc[-1]
            close_date  = dec_data.index[-1]
        else:
            # Fallback: hari terakhir window yang tersedia
            close_price = window_data['Close'].iloc[-1]
            close_date  = window_data.index[-1]

        rec['BT_Close_Price'] = round(close_price, 2)
        rec['BT_Close_Date']  = close_date.strftime('%Y-%m-%d')

        # ── Returns ──
        def pct(end, start):
            if pd.isna(start) or start == 0: return np.nan
            return round((end - start) / start * 100, 2)

        rec['BT_Return_to_High (%)']  = pct(high_price,  entry_price)
        rec['BT_Return_to_Close (%)'] = pct(close_price, entry_price)
        rec['BT_Max_Drawdown (%)']    = pct(low_price,   entry_price)  # negatif = drawdown

        rec['BT_Data_Status'] = 'PARTIAL' if is_partial else 'COMPLETE'
        results.append(rec)

    return pd.DataFrame(results, index=df_val.index)

